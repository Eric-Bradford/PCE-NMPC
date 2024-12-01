from pylab import *
import numpy as np
import math
from PCE_problem_definition import *
from PCE_class import *
from casadi import *
from scipy.stats import multivariate_normal
from scipy.io import savemat
from pyDOE import *
import sobol_seq as sobol_seq

class PCE_NMPC:
    def __init__(self):
        # Variable definitions
        self.xd, self.xa, self.xu, self.u, self.ODEeq, self.Aeq, self.Obj_M, \
        self.Obj_L, self.R, self.ng, self.gfcn, self.G, self.pg, self.u_min, \
        self.u_max, self.states, self.algebraics, self.inputs, \
        self.hfcn, self.Acoeff0, self.L_S, self.Sigma_m, \
        self.gpfcn, self.pgp, self.GP, self.ngp, self.Sigma_w, self.nm, \
        self.Agfcn, self.Agpfcn, self.Ahfcn = DAE_system()
        self.tf, self.nk, self.shrinking_horizon, self.deg, self.cp, self.nicp,\
        self.simulation_time, self.opts, self.number_of_repeats, self.ac_N, \
        self.manner_N, self.PCorder_N, self.ac_S, self.manner_S, self.PCorder_S, \
        self.morder = specifications()
        self.h = self.tf/self.nk/self.nicp
        self.nd, self.na = SX.size(self.xd)[0], SX.size(self.xa)[0] 
        self.nu , self.nun = SX.size(self.u)[0], SX.size(self.xu)[0]
        self.PCE_S = PCE(self.nun+self.nd,self.ac_S,self.manner_S,self.PCorder_S)   
        self.PCE_N = PCE(self.nun+self.nd,self.ac_N,self.manner_N,self.PCorder_N)
        self.ps_N, self.ws_N, self.ns_N = self.PCE_N.Sample_design()
        self.PSIfcn_N, self.L_N = self.PCE_S.PSI_fcn()
        self.aoptfcn_N, self.PCE_meanfcn_N, self.PCE_variancefcn_N, self.chebychevfcn \
        = self.PCE_N.PCE_function_definitions()
        self.Acoeff_solver = self.state_estimator()
        self.ns_S  = 10000
        self.ws_S  = SX.ones(self.ns_S)/self.ns_S
        self.ps_Sa = sobol_seq.i4_sobol_generate(2*(self.nun+self.nd),self.ns_S)
        for i in range(2*(self.nun+self.nd)):
            self.ps_Sa[:,i] = norm(loc=0.,scale=1.).ppf(self.ps_Sa[:,i]) 
        self.ps_S = SX(DM(self.ps_Sa[:,:self.nun+self.nd]))
        self.w_s  = SX(DM(self.ps_Sa[:,self.nun+self.nd:2*(self.nun+self.nd)]))
        
        # Internal function calls
        self.C, self.D                  = self.collocation_points()
        self.ffcn, self.Afcn, self.Bfcn = self.model_fcn()
        self.NV, self.V, self.vars_lb, self.vars_ub, self.vars_init, self.XD, \
        self.XA, self.U, self.con, self.v, self.ucl = self.NLP_specification() 
        self.clfcn, self.satfcn = self.closed_loop_function()
        self.vars_init, self.vars_lb, self.vars_ub, self.g, self.lbg, \
        self.ubg, self.XD, self.XA, self.U, self.cfcn, self.tffcn, self.Cov, self.testfcn, self.UCLfcn \
                                          = self.set_constraints()
        self.Obj, self.g, self.lbg, self.ubg, self.mean_gfcn, self.variance_gfcn, self.g_Ffcn, self.stdgfcn = self.set_pconstraints_objective()
        self.solver                          = self.create_solver()
        self.integrator                      = self.integrator_def()
        
    def collocation_points(self):
        deg, cp, nk, h = self.deg, self.cp, self.nk, self.h
        C = np.zeros((deg+1,deg+1)) # Coefficients of the collocation equation
        D = np.zeros(deg+1)         # Coefficients of the continuity equation
        
        # All collocation time points
        tau = SX.sym("tau") # Collocation point
        tau_root = [0] + collocation_points(deg,cp)
        T = np.zeros((nk,deg+1))
        for i in range(nk):
            for j in range(deg+1):
                T[i][j] = h*(i + tau_root[j])
        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        for j in range(deg+1):
            L = 1
            for j2 in range(deg+1):
                if j2 != j:
                    L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
            lfcn = Function('lfcn', [tau],[L])
        
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = lfcn(1.0)
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            tfcn = Function('tfcn', [tau],[tangent(L,tau)])
            for j2 in range(deg+1):
                C[j][j2] = tfcn(tau_root[j2]) 
            
        return C, D    
    
    def model_fcn(self):
        xd, xa, u, xu, ODEeq, Aeq = self.xd, self.xa, self.u, self.xu, self.ODEeq, self.Aeq
        t                         = SX.sym("t")               
        p_s                       = SX.sym("p_s")      
        nun, nd, na, nk           = self.nun, self.nd, self.na, self.nk
        xddot, nu                 = SX.sym("xddot",nd), self.nu  
        
        res   = []
        for i in range(nd):
            res = vertcat(res,ODEeq[i]*p_s*t - xddot[i]) 
        
        for i in range(na):
            res = vertcat(res,Aeq[i])
        
        ffcn = Function('ffcn', [t,xddot,xd,xa,xu,u,p_s],[res])
        
        res   = []
        for i in range(nd):
            res = vertcat(res,jacobian(ODEeq[i]*p_s*t,vertcat(xd,xu)))
        
        for i in range(nun):
            a   = SX.zeros(nun+nd)
            res = vertcat(res,a.T)
        
        A        = SX.sym('A',nd+nun,nd+nun)
        exprA1   = (SX.eye(nd+nun) + 1./2*A + 1./10*mtimes(A,A) + 1./120*mtimes(A,mtimes(A,A)))
        exprA2   = (SX.eye(nd+nun) - 1./2*A + 1./10*mtimes(A,A) - 1./120*mtimes(A,mtimes(A,A)))
        Pade3fcn = Function('Pade3fcn',[A],[mtimes(solve(exprA2,SX.eye(nd+nun)),exprA1)])
        Afcn     = Function('Afcn',[t,xd,xu,u,p_s],[Pade3fcn(res/SX(nk))])
        
        res1  = []
        for i in range(nd):
            res1 = vertcat(res1,jacobian(ODEeq[i]*p_s*t,u))
        for i in range(nun):
            a    = SX.zeros(nu)
            res1 = vertcat(res1,a.T)
        
        Ainv = solve(res/SX(nk),SX.eye(nd+nun))
        Bfcn = Function('Bfcn',[A,t,xd,xu,u,p_s],[mtimes(1./(2*SX(nk))*(SX.eye(nd+nun)+A),res1)])   
    
        return ffcn, Afcn, Bfcn

    def NLP_specification(self):
        xd, xa, u, nk, deg, nicp = self.xd, self.xa, self.u, self.nk, self.deg, self.nicp
        nd, na, nu, nx           = self.nd, self.na, self.nu, self.nd+self.na 
        ng, gfcn                 = self.ng, self.gfcn
        nicp, deg, nm            = self.nicp, self.deg, self.nm
        ns_N, nun, ngp           = self.ns_N, self.nun, self.ngp
        
        # Total number of variables
        NXD = nicp*nk*(deg+1)*nd # Collocated differential states
        NXA = nicp*nk*deg*na     # Collocated algebraic states
        NU  = nk*nu              # Feedforward controls
        NUP = nm*nu              # Policy parametrization
        NUM = (nk-1)*nu*ns_N     # Control matrix
        NV  = (NXD+NXA)*ns_N + NU + NUP + NUM    
        
        # NLP variable vector
        V   = MX.sym("V",NV+ng+ngp*nk+(nun+nd)**2*nk*ns_N+1)
        con = MX.sym("con",ns_N*(nd+nun)+nk+nu+nu*nm*(nk-1))
        
        # All variables with bounds and initial guess
        vars_lb       = np.zeros(NV+ng+ngp*nk+(nun+nd)**2*nk*ns_N+1)
        vars_ub       = np.zeros(NV+ng+ngp*nk+(nun+nd)**2*nk*ns_N+1)
        vars_init     = np.zeros(NV+ng+ngp*nk+(nun+nd)**2*nk*ns_N+1)
        vars_lb[-1], vars_ub[-1], vars_init[-1] = 1., 1e6, 1e4
        
        # differential states, algebraic states and control matrix definition after
        # discredization
        XD      = np.resize(np.array([],dtype=MX),(nk,nicp,deg+1,ns_N)) # NB: same name as above
        XA      = np.resize(np.array([],dtype=MX),(nk,nicp,deg,ns_N))   # NB: same name as above
        U       = np.resize(np.array([],dtype=MX),(nk,ns_N))
        v       = np.resize(np.array([],dtype=MX),nk)
        ucl     = np.resize(np.array([],dtype=MX),nk)
        
        return NV, V, vars_lb, vars_ub, vars_init, XD, XA, U, con, v, ucl

    def closed_loop_function(self):
        nm, nu       = self.nm, self.nu
        u_max, u_min = SX(DM(self.u_max)), SX(DM(self.u_min)) 
        
        # Define closed-loop policy function
        y      = SX.sym("y",nm)
        ymean  = SX.sym("ymean",nm)
        para   = SX.sym('para',nu*nm)
        K      = reshape(para,nu,nm)
        yscale = SX([1.,1e1])
        uscale = SX([1e4,1.])
        clfcn  = Function('clfcn',[y,ymean,para],[\
                          uscale*mtimes(K,(y-ymean)*yscale)])
    
        # Saturation function
        u      = SX.sym("u",nu)
        satfcn = Function('satfcn',[u],[(exp(u)/(exp(u)+1))*(u_max-u_min) + u_min])
    
        return clfcn, satfcn

    def set_constraints(self):
        nk, nicp, deg, C, h  = self.nk, self.nicp, self.deg, self.C, self.h
        ffcn, D, v           = self.ffcn, self.D, self.v
        nd, na, nu, nx       = self.nd, self.na, self.nu, self.nd+self.na
        u_min,u_max          = self.u_min, self.u_max
        ng, gfcn             = self.ng, self.gfcn
        con, V, NV, nm       = self.con, self.V, self.NV, self.nm
        vars_lb, vars_ub     = self.vars_lb, self.vars_ub
        XD, XA, U, vars_init = self.XD, self.XA, self.U, self.vars_init
        ns_N, nun, ngp       = self.ns_N, self.nun, self.ngp
        Sigma_w, Afcn, Bfcn  = SX(DM(self.Sigma_w)), self.Afcn, self.Bfcn
        Ahfcn                = self.Ahfcn
        L_N, aoptfcn_N       = self.PCE_N.L, self.aoptfcn_N
        covariance_fcn       = self.PCE_N.Covariance(nd)
        PCE_meanfcn_N        = self.PCE_meanfcn_N
        Sigma_m, hfcn        = MX(DM(self.Sigma_m)), self.hfcn
        clfcn, satfcn        = self.clfcn, self.satfcn
        
        xd_current    = reshape(con[:ns_N*nd],ns_N,nd)
        xu_current    = reshape(con[ns_N*nd:ns_N*(nd+nun)],ns_N,nun)
        p_s           = con[ns_N*(nd+nun):ns_N*(nd+nun)+nk]
        UCLM          = reshape(con[ns_N*(nd+nun)+nk+nu:ns_N*(nd+nun)+nk+nu+(nk-1)*nm*nu],nk-1,nu*nm)
    
        xD_init          = np.array((nk*nicp*(deg+1))*[[1.56e3,1e1,378.,1e1]])
        xA_init          = np.array((nk*nicp*(deg+1))*[[]])
        u_init           = np.array((nk*nicp*(deg+1))*[[1e4,300.]])
        offset                                = NV+(nun+nd)**2*nk*ns_N
        vars_lb[offset:offset+ng]             = np.ones(ng)*0.
        vars_ub[offset:offset+ng]             = np.ones(ng)*inf
        vars_init[offset:offset+ng]           = np.ones(ng)*1e-1
        vars_lb[offset+ng:offset+ng+ngp*nk]   = np.ones(ngp*nk)*0.
        vars_ub[offset+ng:offset+ng+ngp*nk]   = np.ones(ngp*nk)*inf        
        vars_init[offset+ng:offset+ng+ngp*nk] = np.ones(ngp*nk)*1e-1
        
        offset  = 0
        
        xD_min, xD_max  = np.array([1e2,-inf,100.,-inf]), np.array([inf,inf,600.,inf])
        xA_min, xA_max  = np.array([-inf]*na), np.array([inf]*na)
                
        # Get collocated states and parametrized control
        for k in range(nk):
            for l in range(ns_N):
                # Collocated states
                for i in range(nicp):
                    for j in range(deg+1):
                                  
                        # Get the expression for the state vector
                        XD[k][i][j][l] = V[offset:offset+nd]
                        if j !=0:
                            XA[k][i][j-1][l] = V[offset+nd:offset+nd+na]
                        # Add the initial condition
                        index = (deg+1)*(nicp*k+i) + j
                        if k==0 and j==0 and i==0:
                            vars_init[offset:offset+nd] = xD_init[index,:]
                            
                            vars_lb[offset:offset+nd] = xD_min
                            vars_ub[offset:offset+nd] = xD_max                    
                            offset += nd
                        else:
                            if j!=0:
                                vars_init[offset:offset+nx] = np.append(xD_init[index,:],xA_init[index,:]) 
                                
                                vars_lb[offset:offset+nx] = np.append(xD_min,xA_min)
                                vars_ub[offset:offset+nx] = np.append(xD_max,xA_max)
                                offset += nx
                            else:
                                vars_init[offset:offset+nd] = xD_init[index,:]
                                
                                vars_lb[offset:offset+nd] = xD_min
                                vars_ub[offset:offset+nd] = xD_max
                                offset += nd
            
                # Control matrix
                if k != 0:
                    U[k][l]                     = V[offset:offset+nu]
                    vars_lb[offset:offset+nu]   = u_min
                    vars_ub[offset:offset+nu]   = u_max
                    vars_init[offset:offset+nu] = u_init[index,:]
                    offset                     += nu
                
            # Parametrized controls
            v[k]                        = V[offset:offset+nu]
            if k == 0:
                vars_lb[offset:offset+nu]   = u_min 
                vars_ub[offset:offset+nu]   = u_max 
            else:
                vars_lb[offset:offset+nu]   = u_min - 0.05*(u_max-u_min)  
                vars_ub[offset:offset+nu]   = u_max + 0.05*(u_max-u_min)
            vars_init[offset:offset+nu] = u_init[index,:]
            offset                     += nu 
    
        # Closed-loop control policy parameters np.array([-inf]*nu) np.array([ inf]*nu)
        ucl                            = V[offset:offset+nu*nm]
        vars_lb[offset:offset+nu*nm]   = np.array([-3e-2,-15.,-1e-2,-3.]) 
        vars_ub[offset:offset+nu*nm]   = np.array([ 3e-2, 15., 1e-2, 3.])               
        vars_init[offset:offset+nu*nm] = np.array([0.015,-7.5, 4e-3,-1.25])
        offset                        += nu*nm
        
        assert(offset==NV)
        
        # Nonlinear constraints 
        g   = []
        lbg = []
        ubg = []
        
        # Mean of y
        Meany = MX.zeros(nk,nm)
        for k in range(nk):
            for j in range(nm):
                yvector = MX.zeros(ns_N)
                for l in range(ns_N):
                    yvector[l] = hfcn(XD[k][0][0][l],xu_current[l,:])[j]
                acoeffy = aoptfcn_N(yvector)
                meany   = PCE_meanfcn_N(acoeffy)
                Meany[k,j] = meany      
        
        p_sa     = SX.sym('p_sa')
        va       = SX.sym('va',nu)
        yreala   = SX.sym('yreala',nm)
        ymeana   = SX.sym('ymeana',nm)
        ucla     = SX.sym('ucla',nm*nu)
        ufcn     = Function('ufcn',[va,yreala,ymeana,ucla,p_sa],[va + p_sa*clfcn(yreala,ymeana,ucla)])
        u1       = SX.sym('u1',nu)
        u2       = SX.sym('u2',nu)
        udifffcn = Function('udifffcn',[u1,u2],[u1-u2])
        for k in range(nk):
            for l in range(ns_N):
                if k == 0:
                    U[k][l] = v[k]
                else:
                    yreal = hfcn(XD[k][0][0][l],xu_current[l,:])
                    ymean = Meany[k,:]
                    g += [udifffcn(U[k][l],ufcn(v[k],yreal,ymean,ucl,p_s[k]))]
                    lbg.append(np.array([0.]*nu))
                    ubg.append(np.array([0.]*nu))
       
        # Covariance definition       
        Aa1         = SX.sym('Aa1',nd+nun,nd+nun)
        Ba1         = SX.sym('Ba1',nd+nun,nu)
        Sigma_wa    = SX.sym('Sigma_wa',nd+nun,nd+nun)
        Sigma_xa    = SX.sym('Sigma_xa',nd+nun,nd+nun)
        p_sa        = SX.sym('p_sa')
        Ha          = SX.sym('Ha',nm,nd+nun)
        Sigma_ua    = SX.sym('Sigma_ua',nu,nu)
        Sigma_uxa   = SX.sym('Sigma_uxa',nu,nd+nun)
        Sigma_xua   = SX.sym('Sigma_xua',nd+nun,nu)
        Ka          = SX.sym('Ka',nu,nm)
        Sigma_ma    = SX.sym('Sigma_ma',nm,nm)
        Sigma_ya    = SX.sym('Sigma_ya',nm,nm)
        Sigma_yfcn  = Function('Sigma_yfcn',[Ha,Sigma_xa,Sigma_ma],\
                               [mtimes(mtimes(Ha,Sigma_xa),Ha.T) + Sigma_ma])
        Sigma_ufcn  = Function('Sigma_ufcn',[Ka,Sigma_ya],\
                               [mtimes(mtimes(Ka,Sigma_ya),Ka.T)])
        Sigma_uxfcn = Function('Sigma_uxfcn',[Aa1,Ka,Ha,Sigma_xa],[\
                               mtimes(Ka,mtimes(Ha,Sigma_xa))])
        Sigma_xufcn = Function('Sigma_xufcn',[Aa1,Ka,Ha,Sigma_xa],[\
                               mtimes(mtimes(Sigma_xa,Ha.T),Ka.T)])
        Amtimesfcn  = Function('Amtimesfcn',[Sigma_xa,Aa1,Sigma_wa],
                              [mtimes(mtimes(Aa1,Sigma_xa),Aa1.T) + Sigma_wa])
        Feedbackfcn = Function('Feedbackfcn',[Sigma_xa,Sigma_ua,Sigma_uxa,Sigma_xua,Aa1,Ba1],\
        [Sigma_xa + mtimes(mtimes(Ba1,Sigma_ua),Ba1.T) + mtimes(mtimes(Ba1,Sigma_uxa),Aa1.T) +\
        mtimes(mtimes(Aa1,Sigma_xua),Ba1.T)]) 
        Sigma_xa2   = SX.sym('Sigma_xa2',nd+nun,nd+nun)
        Aminusfcn   = Function('Aminusfcn',[Sigma_xa,Sigma_xa2],[\
        reshape(Sigma_xa,(nd+nun)**2,1) - reshape(Sigma_xa2,(nd+nun)**2,1)])
    
        Cov        = np.resize(np.array([],dtype=MX),(nk+1,ns_N))
        offset     = NV
        for k in range(nk):
            for l in range(ns_N):
                Cov[0][l]   = MX.eye(nd+nun)*1e-8
                Cov[k+1][l] = reshape(V[offset:(nd+nun)**2+offset],nun+nd,nun+nd)      
                vars_lb[offset:(nd+nun)**2+offset]   = np.ones((nd+nun)**2)*-inf
                vars_ub[offset:(nd+nun)**2+offset]   = np.ones((nd+nun)**2)*inf
                vars_init[offset:(nd+nun)**2+offset] = np.zeros((nd+nun)**2)
                A1 = Afcn(V[-1],XD[k][0][0][l],xu_current[l,:],U[k][l],p_s[k])
                B1 = Bfcn(A1,V[-1],XD[k][0][0][l],xu_current[l,:],U[k][l],p_s[k])
                H  = Ahfcn(XD[k][0][0][l],xu_current[l,:])
                if k == 0:
                    K = MX.zeros(nu,nm)
                else:
                    K = reshape(ucl,nu,nm)
                Sigmax   = Amtimesfcn(Cov[k][l],A1,MX(DM(Sigma_w)))
                Sigma_y  = Sigma_yfcn(H,Cov[k][l],Sigma_m)
                Sigma_u  = Sigma_ufcn(K,Sigma_y)
                Sigma_ux = Sigma_uxfcn(A1,K,H,Cov[k][l])
                Sigma_xu = Sigma_xufcn(A1,K,H,Cov[k][l])
                Sigmaxp  = Feedbackfcn(Sigmax,Sigma_u,Sigma_ux,Sigma_xu,A1,B1)
                g       += [Aminusfcn(Sigmaxp,Cov[k+1][l])]
                lbg.append(np.zeros((nd+nun)**2))
                ubg.append(np.zeros((nd+nun)**2))
                offset += (nd+nun)**2
                                  
        # Initial value constraint     
        for l in range(ns_N):
            g   +=  [XD[0][0][0][l] - xd_current[l,:].T]
            lbg.append(np.zeros(nd)) 
            ubg.append(np.zeros(nd)) 
            
        # For all finite elements
        for k in range(nk):
            for l in range(ns_N):
                for i in range(nicp):
                    # For all collocation points
                    for j in range(1,deg+1):                
                        # Get an expression for the state derivative at the collocation point
                        xp_jk = 0
                        for j2 in range (deg+1):
                            xp_jk += C[j2][j]*XD[k][i][j2][l] # get the time derivative of the differential states (eq 10.19b)
                        
                        # Add collocation equations to the NLP
                        fk = ffcn(V[-1],xp_jk/h,XD[k][i][j][l],XA[k][i][j-1][l],xu_current[l,:],U[k][l],p_s[k])
                        g += [fk[:nd]]           # impose system dynamics (for the differential states (eq 10.19b))
                        lbg.append(np.zeros(nd)) # equality constraints
                        ubg.append(np.zeros(nd)) # equality constraints
                        g += [fk[nd:]]                               # impose system dynamics (for the algebraic states (eq 10.19b))
                        lbg.append(np.zeros(na)) # equality constraints
                        ubg.append(np.zeros(na)) # equality constraints
                                                                               
                    np.resize(np.array([],dtype=SX),(nk,nicp,deg))
                    # Get an expression for the state at the end of the finite element
                    if k > 0:
                        xf_k = 0
                        for j in range(deg+1):
                            xf_k += D[j]*XD[k-1][i][j][l]
                        
                        # Add continuity equation to NLP
                        if i==nicp-1:
                            g += [XD[k][0][0][l] - xf_k]
                        else:
                            g += [XD[k-1][i+1][0][l] - xf_k]
                    
                        lbg.append(np.zeros(nd))
                        ubg.append(np.zeros(nd))
        
        UCLmat = MX.zeros(nk-1,nu*nm)
        for i in range(nk-1):
            UCLmat[i,:] = ucl
            
        cfcn    = Function('cfcn' ,[V],[v[0]])
        tffcn   = Function('tffcn',[V],[V[-1]])
        testfcn = Function('testfcn',[V,con],[A1,B1,Cov[k][l],Sigmax,Sigma_u,Sigma_y,Sigma_ux,Sigma_xu,Sigmaxp])
        UCLfcn  = Function('UCLfcn',[V,con],[UCLmat])
        
        return vars_init, vars_lb, vars_ub, g, lbg, ubg, XD, XA, U, cfcn, tffcn, Cov, testfcn, UCLfcn

    def set_pconstraints_objective(self):
        G, R, V, NV      = self.G, self.R, self.V, self.NV
        nk, nicp, deg    = self.nk, self.nicp, self.deg
        U, XD, XA, ns_N  = self.U, self.XD, self.XA, self.ns_N
        nd, na, nu, nx   = self.nd, self.na, self.nu, self.nd+self.na
        nun, gfcn        = self.nun, self.gfcn
        gpfcn, GP, pgp   = self.gpfcn, self.GP, self.pgp
        ng, con, Obj_L, Obj_M = self.ng, self.con, self.Obj_L, self.Obj_M
        PCE_meanfcn_N, PCE_variancefcn_N, chebychevfcn, pg  = \
        self.PCE_meanfcn_N, self.PCE_variancefcn_N, self.chebychevfcn, self.pg
        g, ubg, lbg, ngp = self.g, self.ubg, self.lbg, self.ngp
        lambdav          = V[NV+(nun+nd)**2*nk*ns_N:NV+(nun+nd)**2*nk*ns_N+ng+ngp*nk]
        p_s              = con[ns_N*(nd+nun):ns_N*(nd+nun)+nk]
        u_previous       = con[ns_N*(nd+nun)+nk:ns_N*(nd+nun)+nk+nu]
        Obj_F            = MX.zeros(ns_N)
        xu_current       = reshape(con[ns_N*nd:ns_N*(nd+nun)],ns_N,nun)        
        Cov              = self.Cov
        Agfcn, Agpfcn    = self.Agfcn, self.Agpfcn
        offset           = NV+(nun+nd)**2*nk*ns_N
        stdg             = V[offset:offset+ng]
        stdgp            = V[offset+ng:offset+ng+ngp*nk]
        aoptfcn_N        = self.aoptfcn_N

        # Define terminal nonlinear constraints
        g_F      = MX.zeros(ns_N,ng)
        for l in range(ns_N):
            for j in range(ng):
                g_F[l,j] = gfcn(XD[nk-1][nicp-1][deg][l],XA[nk-1][nicp-1][deg-1][l],\
                xu_current[l,:],U[-1][l])[j]
        g_Ffcn   = Function('g_Ffcn',[V,con],[g_F])
        
        A          = SX.sym('A',ng,nd+nun)
        Sigma_x    = SX.sym('Sigma_x',nd+nun,nd+nun)
        diagvarfcn = Function('diagvarfcn',[A,Sigma_x],[diag(mtimes(mtimes(A,Sigma_x),A.T))]) 
        g_Fvar     = MX.zeros(ns_N,ng)
        for l in range(ns_N):
            for j in range(ng):
                Ag = Agfcn(XD[nk-1][nicp-1][deg][l],xu_current[l,:],U[-1][l])
                g_Fvar[l,j] = diagvarfcn(Ag,Cov[-1][l])[j]
                
        # Define path nonlinear constraints
        g_FP     = np.resize(np.array([],dtype=MX),nk)
        for k in range(nk):
            g_Fp = MX.zeros(ns_N,ngp)
            for l in range(ns_N):
                for j in range(ngp):
                    g_Fp[l,j] = gpfcn(XD[k][nicp-1][deg][l],XA[k][nicp-1][deg-1][l],\
                       xu_current[l,:],U[k][l])[j]
            g_FP[k] = g_Fp        
        
        Ap          = SX.sym('A',ngp,nd+nun)
        diagvarfcnp = Function('diagvarfcnp',[Ap,Sigma_x],[diag(mtimes(mtimes(Ap,Sigma_x),Ap.T))]) 
        g_FPvar = np.resize(np.array([],dtype=MX),nk)  
        for k in range(nk):
            g_Fpvar = MX.zeros(ns_N,ngp)
            for l in range(ns_N):
                for j in range(ngp):
                    Agp = Agpfcn(XD[k][nicp-1][deg][l],xu_current[l,:],U[k][l])
                    g_Fpvar[l,j] = diagvarfcnp(Agp,Cov[k][l])[j]
            g_FPvar[k] = g_Fpvar
                    
        std1        = SX.sym('std1')
        variance_g1 = SX.sym('variance_g1')
        squareminusfcn = Function('squareminusfcn',[std1,variance_g1],[variance_g1 + SX(1e-8) - std1**2])
        for j in range(ng):
            a_g_F      = aoptfcn_N(g_F[:,j])
            a_g_Fvar   = aoptfcn_N(g_Fvar[:,j])
            mean_g     = PCE_meanfcn_N(a_g_F)
            variance_g = PCE_variancefcn_N(a_g_F) + PCE_meanfcn_N(a_g_Fvar)
            g         += [squareminusfcn(stdg[j],variance_g)]
            lbg.append([0])
            ubg.append([0])
            g         += [chebychevfcn(mean_g,stdg[j],pg[j],MX(1))]    
            lbg.append([-inf]) 
            ubg.append([0.])
        mean_gfcn = Function('mean_gfcn',[V,con],[mean_g])
        variance_gfcn = Function('variance_gfcn',[V,con],[variance_g])
        
        i = 0
        for k in range(nk):
            for j in range(ngp):
                a_g_FP          = aoptfcn_N(g_FP[k][:,j])
                a_g_FPvar       = aoptfcn_N(g_FPvar[k][:,j])
                mean_gp         = PCE_meanfcn_N(a_g_FP)
                variance_gp     = PCE_variancefcn_N(a_g_FP) + PCE_meanfcn_N(a_g_FPvar)
                g              += [squareminusfcn(stdgp[i],variance_gp)]
                lbg.append([0])
                ubg.append([0])
                g         += [chebychevfcn(mean_gp,stdgp[i],pgp[j],p_s[k])]   
                lbg.append([-inf]) 
                ubg.append([0.])
                i         += 1
        
        for l in range(ns_N):            
            # Control penality
            u1     = SX.sym('u1',nu)
            u2     = SX.sym('u2',nu)
            dufcn  = Function('dufcn',[u1,u2],[mtimes(mtimes(transpose(u2-u1),R),u2-u1)])
            deltau = MX.zeros(1)
            for k in range(nk-1):
                if k == 0:
                    deltau += dufcn(u_previous,U[k][l])*p_s[k]
                else:
                    deltau += dufcn(U[k][l],U[k+1][l])*p_s[k]
            Obj_F[l] += deltau
        
            # Lagrange term of objective
            lagrange = MX.zeros(1)
            for k in range(nk): 
                lagrange += Obj_L(XD[k][nicp-1][deg][l],XA[k][nicp-1][deg-1][l],U[k][l],0.)
            Obj_F[l] += lagrange    
                
            # Mayer term of objective
            Obj_F[l] += Obj_M(XD[nk-1][nicp-1][deg][l],XA[nk-1][nicp-1][deg-1][l],U[-1][l],V[-1])     
        
        a_Obj   = aoptfcn_N(Obj_F)
        Obj     = PCE_meanfcn_N(a_Obj)
        stdgfcn = Function('stdgfcn',[V,con],[deltau])       
        
        return Obj, g, lbg, ubg, mean_gfcn, variance_gfcn, g_Ffcn, stdgfcn
         
    def create_solver(self):
        V, con, Obj, g, opts = self.V, self.con, self.Obj, self.g, self.opts  
       
        # Define NLP
        nlp = {'x':V, 'p':con, 'f':Obj, 'g':vertcat(*g)} 
            
        # Allocate an NLP solver
        solver = nlpsol("solver", "ipopt", nlp, opts)   
       
        return solver
    
    def integrator_def(self):
        xd, xa, u, ODEeq, Aeq = self.xd, self.xa, self.u, self.ODEeq, self.Aeq
        xu, nu, nun           = self.xu, self.nu, self.nun
        p     = SX.sym('p',self.nu+self.nun+1)
        uNMPC = p[:nu]
        xus   = p[nu:nun+nu]
        dt    = p[-1]
        
        ODE = []
        for i in range(self.nd):
            ODE = vertcat(ODE,substitute(ODEeq[i]*dt,vertcat(u,xu),vertcat(uNMPC,xus))) 
        
        A = []
        for i in range(self.na):
            A   = vertcat(A,substitute(Aeq[i],vertcat(u,xu),vertcat(uNMPC,xus)))   
    
        dae = {'x':xd, 'z':xa, 'p':p, 'ode':ODE, 'alg':A} 
        I   = integrator('I', 'idas', dae, {'abstol':1e-10,'reltol':1e-10})
        
        return I
    
    def simulator(self,xd_previous,uNMPC,t0,tf,xu_real):
        I          = self.integrator
        dt         = tf-t0
        pdef       = np.array(vertcat(uNMPC,xu_real,dt))
        res        = I(x0=xd_previous,p=pdef)
        xd_current = np.array(res['xf'])
        xa_current = np.array(res['zf'])
    
        return xd_current, xa_current
        
    def xu_s(self,Acoeff):
        PCE_S    = self.PCE_S
        PCE_N    = self.PCE_N
        ps_S, L_S, ns_S, ws_S  = self.ps_S, self.L_S, self.ns_S, self.ws_S
        ps_N, L_N, ns_N, ws_N  = self.ps_N, self.L_N, self.ns_N, self.ws_N
        nun, nd  = self.nun, self.nd
        
        xu_S = SX.sym('xu_S',ns_S,nun+nd)
        for i in range(ns_S):
            xu_S[i,:] = PCE_S.xu_fcn(Acoeff)(ps_S[i,:])
        
        xu_N = SX.sym('xu_N',ns_N,nun+nd)
        for i in range(ns_N):
            xu_N[i,:] = PCE_S.xu_fcn(Acoeff)(ps_N[i,:])
        
        return xu_S, xu_N
    
    def xd_s(self,Acoeff,u_past,deltat):
        PCE_S   = self.PCE_S
        nd, nun = self.nd, self.nun                 
        ps_S, L_S, ns_S, ws_S  = self.ps_S, self.L_S, self.ns_S, self.ws_S
        ps_N, L_N, ns_N, ws_N  = self.ps_N, self.L_N, self.ns_N, self.ws_N
        tf, t0 = deltat[-1][-1][-1], 0.
        xu_fcn = PCE_S.xu_fcn(Acoeff)
        
        xdxu_S = SX.sym('xdxu_S',ns_S,nd)
        for i in range(ns_S):
            xda      = DM(xu_fcn(ps_S[i,:]))
            xda, xaa = self.simulator(xda[:nd],u_past[-1],t0,tf,xda[nd:nd+nun])
            xdxu_S[i,:] = xda
        
        xdxu_N = SX.sym('xdxu_N',ns_N,nd)
        for i in range(ns_N):
            xda         = DM(xu_fcn(ps_N[i,:]))
            xda, xaa    = self.simulator(xda[:nd],u_past[-1],t0,tf,xda[nd:nd+nun])
            xdxu_N[i,:] = xda       
        
        return xdxu_S, xdxu_N
    
    def p_moments(self,xdxu_S,xu_S,yd,xd_current,xu_current,Acoeffold):
        nun, hfcn , yd, L_S     = self.nun, self.hfcn, np.array(yd), self.L_S
        Sigma_m, nd, Sigma_w    = self.Sigma_m, self.nd, self.Sigma_w 
        morder, w_s, ns_S, ws_S = self.morder, self.w_s, self.ns_S, self.ws_S     
        xdxu_S, xu_S            = np.array(DM(xdxu_S)), np.array(DM(xu_S))
        diagw, w_s              = np.array(diag(Sigma_w)).T, np.array(DM(w_s))
        ws_S, Sigma_m, nm       = np.array(DM(ws_S)), np.array(Sigma_m), self.nm
        Acoeff1                 = DM(Acoeffold)
        
        alpha  = np.array([0.])
        xt_s   = np.zeros((ns_S,nun+nd))
        likyd  = np.zeros(ns_S)
        for i in range(ns_S):
            xt_s[i,:]   = np.concatenate((xdxu_S[i,:],xu_S[i,nd:])) + w_s[i,:]*np.sqrt(diagw)
            xt_s[i,:nd] = xt_s[i,:nd].clip(min=0.)
            mean_m      = np.array(DM(hfcn(xt_s[i,:nd],xt_s[i,nd:nd+nun]))).flatten()
            mn          = multivariate_normal(mean_m,Sigma_m)
            likyd[i]    = mn.pdf(yd.T) 
            alpha      += ws_S[i]*likyd[i]       
        alpha = fmax(fabs(alpha),1e-14)
        
        exps     = (p for p in product(range(morder+1), repeat=nun+nd) if sum(p) <= morder)
        exps     = list(exps)
        pmoments = np.zeros((len(exps)))
        prod     = np.zeros((ns_S))
        for i in range(len(exps)):
            for j in range(ns_S):
                prod[j] = np.prod(xt_s[j,:]**np.array(exps[i]))*likyd[j]
            pmoments[i] = (1./alpha)*(np.dot(np.transpose(ws_S),prod))
        
        x0guess      = np.concatenate((xd_current,np.array(xu_current).flatten()))
        Acoeff1[:,0] = x0guess
        args         = {}
        args["lbx"]  = np.ones((nun+nd)*L_S+len(exps))*-inf
        args["ubx"]  = np.ones((nun+nd)*L_S+len(exps))*inf
        args["p"]    = pmoments
        args["lbg"]  = np.zeros(len(exps))
        args["ubg"]  = np.zeros(len(exps))
        args["x0"]   = np.concatenate((np.reshape(DM(Acoeff1),((nun+nd)*L_S)),np.zeros(len(exps))))
        
        return args, pmoments
    
    def state_estimator(self): 
        nun, PCE_S, nd, morder = self.nun, self.PCE_S, self.nd, self.morder
        L_S, momentsfcn        = self.L_S, PCE_S.Moments(morder)
        exps = (p for p in product(range(morder+1), repeat=nun+nd) if sum(p) <= morder)
        exps = list(exps)
        g           = []

        diffmomentsa = SX.sym('diffmomentsa',len(exps))
        Acoeffv      = MX.sym('Acoeffv',nun+nd,L_S)
        B            = reshape(Acoeffv,L_S*(nun+nd),1)
        pmomentsa    = SX.sym('pmomentsa',len(exps))
        mtimesfcn    = Function('mtimesfcn',[diffmomentsa],[norm_2(diffmomentsa)**2])
        pmoments     = MX.sym('pmoments',len(exps))
        diffmoments  = MX.sym('diffmoments',len(exps))
        momentsPCE   = SX.sym('momentsPCE',len(exps))
        difffcn      = Function('difffcn',[momentsPCE,pmomentsa],[(momentsPCE-pmomentsa)/pmomentsa])
        diffmomentse = difffcn(momentsfcn(B),pmoments)
        diff1        = SX.sym('diff1',len(exps))
        diff2        = SX.sym('diff2',len(exps))
        diff12fcn    = Function('diff12fcn',[diff1,diff2],[diff1-diff2])
        g           += [diff12fcn(diffmomentse,diffmoments)]
        Obj          = mtimesfcn(diffmoments)
        A            = vertcat(B,diffmoments)
        Objfcn       = Function('Objfcn',[A],[Obj])
        
        opts                                       = {}
        opts["expand"]                             = True
        opts["ipopt.max_iter"]                     = 1000
        opts["ipopt.tol"]                          = 1e-12
        opts['ipopt.linear_solver']                = 'ma57'
        opts['ipopt.warm_start_init_point']        = "yes" 
        
        nlp    = {'x':A, 'p':pmoments, 'f':Objfcn(A), 'g':vertcat(*g)} 
        solver = nlpsol("solver","ipopt",nlp,opts)
        
        return solver
    
    def initialization(self):
        tf, deltat, nu, nd = self.tf, self.tf/self.nk, self.nu, self.nd
        number_of_repeats, na, ng  = self.number_of_repeats, self.na, self.ng
        nun, L_S, nk, ns_S = self.nun, self.L_S, self.nk, self.ns_S
        newvars_lb, newvars_ub = DM(self.vars_lb), DM(self.vars_ub)

        time_loop  = []
        U_pasts    = np.zeros((number_of_repeats,int(math.ceil(tf/deltat)),nu))
        Xd_pasts   = np.zeros((int(math.ceil(tf/deltat))*100+1,number_of_repeats,nd))
        Xa_pasts   = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,na)) 
        Con_pasts  = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,ng))
        AC_pasts   = np.zeros((nk+1,number_of_repeats,nun+nd,L_S))
        xu_pasts   = np.zeros((nk+1,number_of_repeats,nun)) 
        xuS_pasts  = np.zeros((nk+1,number_of_repeats,nun+nd,ns_S))
        t_pasts    = np.zeros((int(math.ceil(tf/deltat))*100+1,number_of_repeats))
        timeNMPC   = np.zeros((nk,number_of_repeats))
        timePCEs   = np.zeros((nk,number_of_repeats))
        t_past     = [self.u_max]
        time_loop  = []
        newvars_lb[self.NV-self.nu*self.nm:self.NV] = np.ones(self.nu*self.nm)*0.    
        newvars_ub[self.NV-self.nu*self.nm:self.NV] = np.ones(self.nu*self.nm)*0.
        
        return U_pasts, Xd_pasts, Xa_pasts, Con_pasts, AC_pasts, time_loop, \
    t_pasts, xu_pasts, xuS_pasts, timeNMPC, timePCEs, newvars_lb, newvars_ub   
    
    def initialization_loop(self):
        PCE_S, Acoeff0 = self.PCE_S, self.Acoeff0
        lbg, ubg, ng   = self.lbg, self.ubg, self.ng
        vars_lb, vars_ub, vars_init = self.vars_lb, self.vars_ub, self.vars_init
        tf, deltat, nu, nd = self.tf, self.tf/self.nk, self.nu, self.nd
        number_of_repeats, na  = self.number_of_repeats, self.na
        
        arg = {} 
        arg["lbg"] = np.concatenate(lbg)
        arg["ubg"] = np.concatenate(ubg)
        arg["lbx"] = vars_lb
        arg["ubx"] = vars_ub
        arg["x0"] =  vars_init
        
        u_nmpc    = np.array(self.u_max)
        t_past    = []
        u_past    = []
        tk        = -1
        t0i       = np.array([[0.]]) 
        tfi       = 0. 
        xu_fcn0   = PCE_S.xu_fcn(Acoeff0)

        return arg, u_past, xu_fcn0, t_past, tk, t0i, tfi, u_nmpc
    
    def update_inputs(self,xu_N,tk,u_nmpc,UCLM):
        nd, nk, nu, nun, ns_N = self.nd, self.nk, self.nu, self.nun, self.ns_N
        nm                    = self.nm
        tk                   += 1
        p                     = np.zeros(ns_N*(nd+nun)+nk+nu+(nk-1)*nm*nu)
        if tk == 0:
            UCLMc = UCLM
        else:
            UCLMc = vertcat(UCLM[tk:,:],DM.zeros(tk,nu*nm))
        UCLMa     = reshape(UCLMc,(nk-1)*nu*nm,1)
        
        if self.shrinking_horizon:
            a = np.concatenate((np.ones(nk-tk),np.zeros(tk)))
        else:
            a = np.ones(nk)
                
        p[:ns_N*nd]              = np.array(DM(reshape(xu_N[:,:nd],nd*ns_N,1)))[:,0]
        p[ns_N*nd:ns_N*(nd+nun)] = np.array(DM(reshape(xu_N[:,nd:],nun*ns_N,1)))[:,0]
        p[ns_N*(nd+nun):ns_N*(nd+nun)+nk]       = a
        p[ns_N*(nd+nun)+nk:ns_N*(nd+nun)+nk+nu] = np.array(u_nmpc).flatten()
        p[ns_N*(nd+nun)+nk+nu:ns_N*(nd+nun)+nk+nu+(nk-1)*nm*nu] = np.array(UCLMa)[:,0]
        
        return p, tk
    
    def collect_data(self,t_past,u_past,time_taken,start,end,t0i,u_nmpc):
        t_past += [t0i[0][0]] 
        u_past += [u_nmpc]
        time_taken += [end-start]
        
        return t_past, u_past, time_taken
    
    def generate_data(self,Xd_pasts,Xa_pasts,Con_pasts,U_pasts,un,loopend,\
                      time_loop,u_past,xu_pasts,deltat,t_pasts,ws):
        simulation_time  = self.simulation_time
        t_pasts[0,un]    = 0.
        xds              = Xd_pasts[0,un,:]
        t0is             = 0. # start time of integrator
        tfis             = 0. # end time of integrator
        l = 0
        time_loop       += [loopend]
        nu, nk, nd       = self.nu, self.nk, self.nd
        
        for k in range(nk):
                for i in range(nu):
                    U_pasts[un][k][i] = u_past[k][i]
        
        for k in range(nk):
            for o in range(100):
                l += 1
                tfis += deltat[k]/100
                xds, xas = self.simulator(xds,u_past[k],t0is,tfis,xu_pasts[k,un,:])
                Xd_pasts[l,un,:]    = xds[:,0]
                Xa_pasts[l-1,un,:]  = xas[:,0]
                Con_pasts[l-1,un,:] = np.array(DM(self.gfcn(xds,xas,xu_pasts[k,un,:],u_past[k]))).flatten()
                t0is += deltat[k][0][0]/100
                t_pasts[l,un]       = t0is 
            xds = xds.flatten() + ws[k][:nd]
        
        return Xd_pasts, Xa_pasts, Con_pasts, U_pasts, time_loop, t_pasts
    
    def plot_graphs(self,t_past,t_pasts,Xd_pasts,Xa_pasts,U_pasts,Con_pasts):
        states              = self.states
        algebraics          = self.algebraics
        inputs              = self.inputs
        number_of_repeats   = self.number_of_repeats
        nd, na, nu, ng      = self.nd, self.na, self.nu, self.ng
        for j in range(nd):
            plt.figure(j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[:,i],Xd_pasts[:,i,j],'-')
            plt.ylabel(states[j])
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
     
        for j in range(na):
            plt.figure(nd+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:,i],Xa_pasts[:,i,j],'-')
            plt.ylabel(algebraics[j])
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        
        for k in range(nu):
            plt.figure(nd+na+k)
            t_pastp = np.sort(np.concatenate([list(xrange(self.nk+1))]*2))
            plt.clf()
            for j in range(number_of_repeats):
                u_pastpF = []
                for i in range(len(U_pasts[j])):
                    u_pastpF += [U_pasts[j][i][k]]*2
                plt.plot(t_pastp[1:-1],u_pastpF,'-')
            plt.ylabel(inputs[k])
            plt.xlabel('time')
            plt.xlim([0,self.nk])  
        
        for j in range(ng):
            plt.figure(nd+na+nu+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:,i],Con_pasts[:,i,j],'-')
            plt.ylabel('g'+str(j))
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        
        return
    
    def save_results(self,Xd_pasts,Xa_pasts,U_pasts,Con_pasts,t_pasts,AC_pasts,xu_pasts,xuS_pasts,timeNMPC,timePCEs):
        states              = self.states
        algebraics          = self.algebraics
        inputs              = self.inputs
        number_of_repeats   = self.number_of_repeats
        nd, na, nu, ng,nun  = self.nd, self.na, self.nu, self.ng, self.nun
        
        savemat('Xd_matrix',{"Xd_matrix":Xd_pasts})
        
        savemat('Xa_matrix',{"Xa_matrix":Xa_pasts})
        
        savemat('U_matrix',{"U_matrix":U_pasts})
        
        savemat('Con_matrix',{"Con_matrix":Con_pasts})
        
        savemat('t_matrix',{"t_matrix":t_pasts})
        
        savemat('AC_matrix',{"AC_matrix":AC_pasts})
         
        savemat('xu_pasts',{"xu_pasts":xu_pasts})
        
        savemat('xuS_pasts',{"xuS_pasts":xuS_pasts})
        
        savemat('timeNMPC',{"timeNMPC":timeNMPC})
        
        savemat('timePCEs',{"timePCEs":timePCEs})
        
        return