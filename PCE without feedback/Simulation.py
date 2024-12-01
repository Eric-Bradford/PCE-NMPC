# Closed-loop simulation of defined problem
from PCE_NMPC import *
from PCE_problem_definition import *
import numpy as np
from casadi import *

PCE_NMPC                   = PCE_NMPC()
solver                     = PCE_NMPC.create_solver()
U_pasts, Xd_pasts, Xa_pasts, Con_pasts, AC_pasts, time_loop, t_pasts, xu_pasts, xuS_pasts, \
timeNMPC, timePCEs         = PCE_NMPC.initialization()
number_of_repeats          = PCE_NMPC.number_of_repeats
simulation_time, nun, nd   = PCE_NMPC.simulation_time, PCE_NMPC.nun, PCE_NMPC.nd
time_taken, Sigma_m, nm    = [], PCE_NMPC.Sigma_m, DM.size(PCE_NMPC.Sigma_m)[0] 
Sigma_w, L_S               = PCE_NMPC.Sigma_w, PCE_NMPC.L_S
Acoeff_solver              = PCE_NMPC.Acoeff_solver

for un in range(number_of_repeats):
    deltat, ws, Acoeffnew = [], np.zeros((PCE_NMPC.nk,nun+nd)), PCE_NMPC.Acoeff0
    loopstart = time.time()
    arg, u_past, xu_fcn0, t_past, tk, t0i, tfi, u_nmpc \
    = PCE_NMPC.initialization_loop()
    xi_real = np.expand_dims(np.random.multivariate_normal(np.array([0.]*(nun+nd)),np.eye(nun+nd)),0).T
    xd_current             = DM([1538,9,385,9.5]) # xu_fcn0(SX(xi_real))[:nd]
    xu_current             = DM([7.5,10]) # xu_fcn0(SX(xi_real))[nd:nd+nun]
    AC_pasts[0,un,:,:]     = DM(PCE_NMPC.Acoeff0)
    Xd_pasts[0,un,:]       = np.array(DM(xd_current)).flatten()
    xu_pasts[0,un,:]       = np.array(DM(xu_current)).flatten()
    xu_S, xu_N             = PCE_NMPC.xu_s(PCE_NMPC.Acoeff0)
    xuS_pasts[0,un,:,:]    = DM(xu_S.T)
    while True:
        print('Repeat number: ' + str(un+1) + '  Iteration: ' + str(tk+1))  
        
        # Break when simulation time is reached
        if tk >= PCE_NMPC.nk-1:
            break
     
        # Create sample vectors of xu and xd for NMPC
        xu_S, xu_N     = PCE_NMPC.xu_s(Acoeffnew)
        
        # Parameter to set initial condition of NMPC algorithm and update discrete time tk
        p, tk      = PCE_NMPC.update_inputs(xu_N,tk,u_nmpc)        
        arg["p"]   = p
        
        arg["x0"]  = np.loadtxt('v_opt'+str(tk))

        # Measure computational time taken
        start1 = time.time()
        
        # Solve NMPC problem and extract first control input u_nmpc
        res           = solver(**arg)
        u_nmpc        = PCE_NMPC.cfcn(np.array(res["x"])[:,0])
        tf_nmpc       = PCE_NMPC.tffcn(np.array(res["x"])[:,0])
        deltat       += [np.array(tf_nmpc/PCE_NMPC.nk)]
        tfi          += np.array((tf_nmpc)/PCE_NMPC.nk)
        
        # Save solution for warm-starting
#        np.savetxt('v_opt'+str(tk), np.array(res["x"]))
            
        # Simulate and measure real system  withSpyder has encountered a pro this control input
        xd_current, xa_current = PCE_NMPC.simulator(xd_current,u_nmpc,t0i,tfi,xu_current)
        w          = np.random.multivariate_normal(np.zeros(nd+nun),Sigma_w)
        xd_current = (xd_current.flatten() + w[:nd]).clip(min=0.)
        xu_current = xu_current + w[nd:nd+nun]
        yd         = DM(PCE_NMPC.hfcn(xd_current,xu_current)) + \
        np.random.multivariate_normal(np.zeros(nm),Sigma_m)

        # Collect data
        end1            = time.time()
        timeNMPC[tk,un] = end1 - start1 
        t_past, u_past, time_taken = \
        PCE_NMPC.collect_data(t_past,u_past,time_taken,start1,end1,t0i,u_nmpc)
        t0i += deltat[tk]
        
        # State estimate from measurement
        Acoeffold               = Acoeffnew
        start2                  = time.time()
        xdxu_S, xdxu_N          = PCE_NMPC.xd_s(Acoeffold,u_past,deltat)
        xu_S, xu_N              = PCE_NMPC.xu_s(Acoeffold)
        args, pmoments          = PCE_NMPC.p_moments(xdxu_S,xu_S,yd,xd_current,xu_current,Acoeffold)
        res                     = Acoeff_solver(**args)['x']
        Acoeffnew               = reshape(res[:(nun+nd)*L_S],nun+nd,L_S)
        AC_pasts[tk+1,un,:,:]   = DM(Acoeffnew)
        xu_pasts[tk+1,un,:]     = np.array(xu_current).flatten()
        xu_S, xu_N              = PCE_NMPC.xu_s(Acoeffnew)
        xuS_pasts[tk+1,un,:,:]  = DM(xu_S.T)
        ws[tk,:]                = w
        end2                    = time.time()
        timePCEs[tk,un]         = end2 - start2
        
    # Generate data for plots
    t_past += [t0i[0][0]] 
    loopend = time.time()
    Xd_pasts, Xa_Pasts, Con_pasts, U_pasts, time_loop, t_pasts = \
    PCE_NMPC.generate_data(Xd_pasts,Xa_pasts,Con_pasts,U_pasts,un,loopend,\
                       time_loop,u_past,xu_pasts,deltat,t_pasts,ws)
    
# Plot results
PCE_NMPC.plot_graphs(t_past,t_pasts,Xd_pasts,Xa_Pasts,U_pasts,Con_pasts)

# Save results
PCE_NMPC.save_results(Xd_pasts,Xa_pasts,U_pasts,Con_pasts,t_pasts,AC_pasts,xu_pasts,xuS_pasts,timeNMPC,timePCEs)