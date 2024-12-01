# SNMPC model and problem setup
import numpy as np
from casadi import *
import math 
from itertools import product
from PCE_class import *
from scipy.special import binom

def specifications():
    
    # NMPC algorithm
    tf = 1.                  # time horizon
    nk = 8                   # number of control intervals 
    shrinking_horizon = True # shrinking horizon or receding horizon
    
    # NMPC PCE specification
    ac_N      = 2     # accuracy of sparse Gauss hermite for NMPC online
    manner_N  = 1     # manner of sparse Gauss hermite for NMPC online 
    PCorder_N = 2     # Polynomial chaos expanion order for NMPC online
    
    # State estimator Gauss hermite specification
    ac_S      = 2     # accuracy of sparse Gauss hermite for state estimator
    manner_S  = 1     # manner of sparse Gauss hermite for state estimator
    PCorder_S = 2     # Polynomial chaos expanion order for state estimator
    
    # Discredization using direct collocation 
    deg  = 5                     # Degree of interpolating polynomial
    cp   = "radau"               # Type of collocation points
    nicp = 1                     # Number of (intermediate) collocation points per control interval
    
    # Simulation
    simulation_time   = 1.  # simulation time
    number_of_repeats = 1   # number of Monte Carlo simulations  
    
    # NLP solver
    opts                                       = {}
    opts["expand"]                             = True
    opts["ipopt.max_iter"]                     = 8000
    opts["ipopt.tol"]                          = 1e-8
    opts["ipopt.linear_solver"]                = 'ma27'
    opts["ipopt.max_cpu_time"]                 = 10000 
    opts["ipopt.mu_strategy"]                  = "adaptive"
    
    # State estimator
    morder = 3 # number of total moments considered in state estimator               

    return tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time, opts, \
 number_of_repeats, ac_N, manner_N, PCorder_N, ac_S, manner_S, PCorder_S, morder

def DAE_system():
    
    tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time, opts, \
    number_of_repeats, ac_N, manner_N, PCorder_N, ac_S, manner_S, PCorder_S, \
    m_order = specifications()
    
    # Specifications
    Nm      =  2        # Moments considered for polymerization
    ndstate =  4        # Number of states
    nastate =  0        # Number of algebraic equations
    nu      =  2        # Number of inputs
    nicp    =  1        # Number of (intermediate) collocation points per control interval
    tf      =  1.       # Final time
    nun     =  2        # Number of uncertain parameters 
    
    # Parameters           
    xhi             = 0.      # - 
    VR              = 17.     # m3 
    Tb              = 373.15  # K
    Tf              = 298.15  # K
    Tad             = 523.15  # K
    MWPO            = 58.08   # kg/kmol
    MWalc           = 90.08   # kg/kmol
    MWH2O           = 18.02   # kg/kmol
    MWcatalyst      = 56.11   # kg/kmol
    Alc0            = 1e1     # kmol
    PO0             = 1e1     # kmol
    T0              = 378.15  # K
    Ah              = 2.4e8   # m3/kmol/s
    Eh              = 8.24e4  # kJ/kmol
    Ai              = 4e9     # m3/kmol/s
    Ei              = 7.8e4   # kJ/kmol
    Ep              = 6.9e4   # kJ/kmol
    At              = 9.5e8   # m3/kmol/s
    Et              = 1.05e5  # kJ/kmol
    CPf             = 2.25    # kJ/kg/K
    CPb             = 2.05    # kJ/kg/K
    R               = 8.314   # kJ/kmol/K
    ncT             = 1.      # kmol
    negativedeltaHP = 9.2e4   # kJ/kmol
    
    # SX definitions
    t         =  SX.sym("t")        
    u         =  SX.sym("u",nu)   
    xd        =  SX.sym("xd",ndstate) 
    xu        =  SX.sym("xu",nun)
    mass      =  xd[0]
    VM        =  xd[1]
    Temp      =  xd[2]
    momentX1  =  xd[3]
    xa        =  SX.sym("xa",nastate) 
    xddot     =  SX.sym("xddot",ndstate) 
    dm        =  SX.sym("dm",1)
    dVM       =  SX.sym("dVM",1)
    dTemp     =  SX.sym("dTemp",1)
    dmomentX  =  SX.sym("dVX",Nm)
    F         =  u[0]/1e6
    Tw        =  u[1]
    algebraic =  SX.sym('algebraic',nastate)
    
    # Initial conditions
    x_initial       = SX.zeros(ndstate)
    x_initial[1]    = SX(PO0)
    x_initial[2]    = SX(T0)
    x_initial[3]    = SX(Alc0)
    x_initial[0]    = SX(Alc0*MWalc) + SX(PO0*MWPO) + ncT*SX(MWcatalyst)
    initial_V       = x_initial[0]*(10**(-3) + 1000*7.576*10**(-10)*(398.15-298.15))
    nN2             = (1.*10.**5*(VR-initial_V))/(8.314*373.)
    momentX         = SX.sym('momentX',2)
    momentX[0]      = SX(Alc0)
    momentX[1]      = momentX1  
    
    # Define uncertain parameters with corresponding coefficients of PCE expansion
    PCE_S          = PCE(nun+ndstate,ac_S,manner_S,PCorder_S)
    PSIfcn_S, L_S  = PCE_S.PSI_fcn()
    xi             = SX.sym('xi',nun+ndstate)
    Acoeff0        = SX.zeros(nun+ndstate,L_S)
    Acoeff0[0,0]   = x_initial[0] # kg  
    Acoeff0[0,1]   = SX(0.)       # kg^2        
    Acoeff0[1,0]   = x_initial[1] # kmol
    Acoeff0[1,3]   = SX(1.)       # kmol^2 
    Acoeff0[2,0]   = x_initial[2] # K
    Acoeff0[2,6]   = SX(4.)       # K^2
    Acoeff0[3,0]   = x_initial[3] # kmol
    Acoeff0[3,10]  = SX(0.5)      # kmol^2
    Acoeff0[4,0]   = 10.          # m3/kmol/s
    Acoeff0[4,15]  = SX(1.)       # (m3/kmol/s)^2 
    Acoeff0[5,0]   = 1.5e1        # kW/K 
    Acoeff0[5,21]  = SX(4.)       # (kW/K)^2
    Ap             = xu[0]*1e6 
    UA             = xu[1]
    
    # Equation definitions    
    Volume          = mass*10**(-3)
    kp              = Ap*exp(-Ep/(R*Temp))
    kt              = At*exp(-Et/(R*Temp))
    PPOsat          = 10**(6.28-(1158./(Temp-36.93)))
    PPGsat          = 10**(8.08-(2692.19/(Temp-14.97)))
    l               = momentX[1]/momentX[0]
    ns              = VM
    polymer_amount  = momentX[0] 
    np1             = polymer_amount
    phis            = ns/(ns+np1*l)
    phiP            = (np1*l)/(ns+np1*l)
    aPO             = exp(log(phis)+(1.-(1./l))*phiP+xhi*phiP**2)
    aW              = 1.
    aPG             = 1.
    VG              = VR - Volume
    PPO             = aPO*PPOsat
    PPG             = aPG*PPGsat
    PN2             = nN2*R*Temp/VG
    Pr              = PPO + PPG + PN2
    momentG         = SX.sym('momentG',Nm)
    ni              = momentX[0]
    momentG[0]      = momentX[0]*ncT/ni
    momentG[1]      = momentX[1]*ncT/ni
    
    # Declare Algebraic equations
    algebraics = []
    Aeq        = []

    # Declare ODE equations (use notation as defined in the strings)
    states      = ['m','VM','Temp','momentX1']
    dm          = F*MWPO
    dVM         = F - (kp*momentG[0] + kt*(momentG[0]))*VM/Volume
    dTemp       = (negativedeltaHP*kp*momentG[0]*VM/Volume-UA*(Temp-Tw) \
                   -F*MWPO*CPf*(Temp-Tf))/(mass*CPb*12)
    
    for k in range(Nm):
        sum1 = 0
        for i in range(0,k+1):
            sum1 += SX(binom(k,i))*momentG[i]
        dmomentX[k] = (kp*sum1 - kp*momentG[k])*VM/Volume
        ODEeq =  [dm,dVM,dTemp,dmomentX[1]] 

    # Define objective (in expectation)
    t           = SX.sym('t')
    Obj_M       = Function('mayer', [xd,xa,u,t],[t]) # Mayer term
    Obj_L       = Function('lagrange', [xd,xa,u,t],[0.])   # Lagrange term
    R           = diag(SX([1e-6,1e-4])) # Weighting of control penality

    # Define control bounds
    inputs = ['F','Tw']
    u_min  =  np.array([0. , 298.15])
    u_max  =  np.array([1e4, 423.15])
    
    # Define terminal constraint functions g(x) <= 0
    Mn        = MWPO*(momentX[1]/momentX[0]) 
    unrct     = MWPO*(VM/mass)*10.**(6)
    gequation = vertcat(-Mn+350.,unrct-1000.)
    ng        = SX.size(gequation)[0]
    gfcn      = Function('gfcn',[xd,xa,xu,u],[gequation])
    pg        = MX([0.05,0.05])
    G         = [2e4,2e4]*diag(SX.ones(ng))
    Ag        = vertcat(jacobian(gequation[0],vertcat(xd,xu)),\
                        jacobian(gequation[1],vertcat(xd,xu)))
    Agfcn     = Function('Jgfcn',[xd,xu,u],[Ag])
    
    # Define path constraint functions g(x) <= 0
    gpequation = vertcat(Temp-420.)
    ngp        = SX.size(gpequation)[0]
    gpfcn      = Function('gpfcn',[xd,xa,xu,u],[gpequation])
    pgp        = MX([0.05])
    GP         = [4e4]*diag(SX.ones(ngp))
    Agp        = jacobian(gpequation[0],vertcat(xd,xu))      
    Agpfcn     = Function('Jgpfcn',[xd,xu,u],[Agp])
    
    # Measurement model
    measfcn = vertcat(Temp,VM)
    nm      = SX.size(measfcn)[0]
    hfcn    = Function('hfcn', [xd,xu],[measfcn])
    Sigma_m = [0.25,1e-3]*diag(np.ones(nm))
    Ah      = SX.sym('Ah',nm,nun+ndstate)
    for i in range(nm):
        Ah[i,:] = jacobian(measfcn[i],vertcat(xd,xu))
    Ahfcn   = Function('Ahfcn',[xd,xu],[Ah])
    
    # Disturbance noise 
    Sigma_w = np.array(diag(DM([1.,1e-3,1.,2.5e-1,5e-2,2e-1])**2*np.ones(nun+ndstate)))
    
    return xd, xa, xu, u, ODEeq, Aeq, Obj_M, Obj_L, R, ng, gfcn, G, pg, u_min, \
u_max, states, algebraics, inputs, hfcn, Acoeff0, L_S, Sigma_m, \
gpfcn, pgp, GP, ngp, Sigma_w, nm, Agfcn, Agpfcn, Ahfcn   