from sys import path
import numpy as np
from pyDOE import *
import math
from scipy.stats.distributions import norm
from itertools import product
from casadi import *
from Sparse_Gauss_Hermite import generate_md_points as Sparse_GH
from pyDOE import *
from multinomial_coefficients import multinomial_coefficients as multinomial_coefficients
import scipy as sc

class PCE:
    def __init__(self,nun,ac,manner,PCorder):
        self.nun                  = nun
        self.ac                   = ac
        self.manner               = manner
        self.PCorder              = PCorder     
        self.PSIfcn, self.L       = self.PSI_fcn()
        self.ps, self.ws, self.ns = self.Sample_design()
        self.PSImatrix            = self.Sample_matrix()
        
    def Sample_design(self):
        ps, ws = Sparse_GH(self.ac,self.nun,self.manner)
        ns     = np.size(ws)
        ps     = SX(np.transpose(ps))
        ws     = SX(ws)
        
        return ps, ws, ns
    
    def PSI_fcn(self):
        x       = SX.sym('x')
        He0fcn  = Function('He0fcn' ,[x],[1.])
        He1fcn  = Function('He1fcn' ,[x],[x])
        He2fcn  = Function('He2fcn' ,[x],[x**2  - 1])
        He3fcn  = Function('He3fcn' ,[x],[x**3  - 3*x])
        He4fcn  = Function('He4fcn' ,[x],[x**4  - 6*x**2 + 3])
        He5fcn  = Function('He5fcn' ,[x],[x**5  - 10*x**3 + 15*x])
        He6fcn  = Function('He6fcn' ,[x],[x**6  - 15*x**4 + 45*x**2 - 15])
        He7fcn  = Function('He7fcn' ,[x],[x**7  - 21*x**5 + 105*x**3 - 105*x])
        He8fcn  = Function('He8fcn' ,[x],[x**8  - 28*x**6 + 210*x**4 - 420*x**2  + 105])
        He9fcn  = Function('He9fcn' ,[x],[x**9  - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x])
        He10fcn = Function('He10fcn',[x],[x**10 - 45*x**8 + 640*x**6 - 3150*x**4 + 4725*x**2 - 945])
        Helist  = [He0fcn,He1fcn,He2fcn,He3fcn,He4fcn,He5fcn,He6fcn,He7fcn,He8fcn,He9fcn,He10fcn]
        
        PCorder = self.PCorder
        nun     = self.nun
        
        xi   = SX.sym("xi",nun)
        exps = (p for p in product(range(PCorder+1), repeat=nun) if sum(p) <= PCorder)
        exps = list(exps)
        L    = math.factorial(nun+PCorder)/(math.factorial(nun)*math.factorial(PCorder))
        PSI  = SX.ones(L)
        for i in range(len(exps)):
            for j in range(nun):
                PSI[i] *= Helist[exps[i][j]](xi[j])
        PSIfcn  = Function('PSIfcn',[xi],[PSI]) 
        
        return PSIfcn, L

    def xu_fcn(self,A_coeff):
        nun         = self.nun
        L, PSIfcn   = self.L, self.PSIfcn
        
        xu = SX.sym('xu',nun)
        xi = SX.sym('xi',nun)
        for i in range(nun):
            xu[i] = mtimes(A_coeff[i,:],PSIfcn(xi))
        xu_fcn = Function('xu_fcn',[xi],[xu])
        
        return xu_fcn
            
    def Sample_matrix(self):
        ps, ws, ns = self.ps, self.ws, self.ns
        PSIfcn, L  = self.PSI_fcn()
        PSImatrix  = SX.zeros(ns,L)
        for i in range(ns):
            PSIa = PSIfcn(ps[i,:])
            for j in range(L):
                PSImatrix[i,j] = PSIa[j]    
        
        return PSImatrix

    def Inner_product(self):
        L, nun = self.L, self.nun
        PSIfcn = self.PSI_fcn()[0]
        ps, ws = Sparse_GH(3,nun,1)
        ps, ws, ns = SX(ps.T), SX(ws), np.size(ws)
        
        Inner_product = SX.zeros(L)
        for i in range(L):
            sum1 = 0
            for s in range(ns):
                sum1 += ws[s]*PSIfcn(ps[s,:])[i]**2
            Inner_product[i] = sum1  
            
        return Inner_product
    
    def PCE_function_definitions(self):
        ns, ws, L            = self.ns, self.ws, self.L
        PSImatrix            = self.PSImatrix
        Inner_product        = self.Inner_product() 
        
        F                    = SX.sym('F',ns) 
        Fa                   = F*ws
        aoptfcn              = Function('aoptfcn',[F],[mtimes(Fa.T,PSImatrix)/Inner_product.T])
    
        a                    = SX.sym('a',1,L)
        PCE_meanfcn          = Function('PCE_meanfcn',[a],[a[0]]) 
        PCE_variancefcn      = Function('PCE_variancefcn',[a],\
                               [mtimes(a[1:]**2,Inner_product[1:])])
                
        x             = SX.sym('x')
        p             = SX.sym('p')
        mean          = SX.sym('mean')
        std           = SX.sym('std')
        p_s           = SX.sym('p_s')
        chebychevfcn  = Function('chebychevfcn',[mean,std,p,p_s],\
                        [p_s*mean + p_s*sqrt((1-p)/p)*std])
  
        return aoptfcn, PCE_meanfcn, PCE_variancefcn, chebychevfcn
    
    def Moments(self,morder):
        L, nun     = self.L, self.nun
        exps = (p for p in product(range(morder+1), repeat=nun) if sum(p) <= morder)
        exps       = list(exps)
        moments    = SX.zeros(len(exps))
        A_coeff    = SX.sym('A_coeff',nun,L)
        ps, ws     = Sparse_GH(ceil(morder),nun,2)
        ps, ws, ns = SX(ps.T), SX(ws), np.size(ws)
        xu_fcn     = self.xu_fcn(A_coeff)
        
        xu         = np.resize(np.array([],dtype=SX),ns)
        for s in range(ns):
            xu[s]  = xu_fcn(ps[s,:])
        
        xuv     = SX.sym('xuv',nun)
        prodxuv = SX(1)
        for j in range(nun):
            prodxuv *= xuv[j]      
        prodxufcn = Function('prodxufcn',[xuv],[prodxuv])
        
        for i in range(len(exps)):
            prod = SX.sym('prod',ns)
            for s in range(ns):
                prod[s] = prodxufcn(xu[s]**exps[i])
            moments[i] = mtimes(ws.T,prod)    
        
        A          = reshape(A_coeff,nun*L,1)
        momentsfcn = Function('momentsfcn',[A],[moments])

        return momentsfcn
     
    def Covariance(self,nd):
        nun, L     = self.nun, self.L
        ps, ws     = Sparse_GH(3,nun,2)
        ps, ws, ns = SX(ps.T), SX(ws), np.size(ws)
        A_coeff    = SX.sym('A_coeff',nun,L)
        xu_fcn     = self.xu_fcn(A_coeff)
        
        covariance = SX.zeros(nun,nun)
        for i in range(nun):
            for j in range(nun):
                xumoment = 0
                for s in range(ns):
                    xumoment += ws[s]*(xu_fcn(ps[s,:])[i]*xu_fcn(ps[s,:])[j] \
                    - A_coeff[i,0]*A_coeff[j,0])
                covariance[i,j] = xumoment
        covariance_fcn = Function('covariance_fcn',[A_coeff],[covariance])        
        
        return covariance_fcn