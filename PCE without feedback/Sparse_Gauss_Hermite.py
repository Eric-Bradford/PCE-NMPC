# Sparse Gauss-Hermite functions
import numpy as np
from scipy.special import factorial 

def GaussHermite(n):

    i  = np.arange(1,n,1)
    a  = np.sqrt(i/2.)
    CM = np.diag(a,1) + np.diag(a,-1)
    
    [L,V]    = np.linalg.eig(CM)
    L        = np.diag(L)
    x        = np.sort(np.diag(L))
    ind      = np.argsort(np.diag(L))
    V        = np.transpose(V[:,ind])
    w        = np.ones((V.shape[0],1))
    w[:,0]   = np.sqrt(np.pi) * V[:,0]**2

    x = np.transpose(x) * np.sqrt(2)
    w = np.transpose(w) / np.sum(w)
        
    return x, w
    
def get_1d_point(index,Point_SI,Manner):

    if Point_SI == 1:

        if index == 0:
            grid_point  = np.array([])
            grid_weight = np.array([])

        else:
            if Manner == 1:
                [grid_point,grid_weight] = GaussHermite(index)

            if Manner == 2:
                [grid_point,grid_weight] = GaussHermite(2*index-1)

            if Manner == 3:
                [grid_point,grid_weight] = GaussHermite(2**index-1)

    return grid_point, grid_weight       

def reasonable(L,D,i):

    ci = np.sum(i)

    if ci <= L+D-1:
        z = 1
    else:
        z = 0

    return z

def generate_nextindex(L,current_index,Oldindex):

    D = np.size(current_index)

    R = np.zeros((D,D))

    for i in range(D):
        R[:,i] = current_index
        R[i,i] = R[i,i] + 1

        if reasonable(L,D,R[:,i]) != 1:
            R[:,i] = 0

    # erase empty column
    R    = np.delete(R,np.where(np.sum(np.abs(R),0)==0),1)
    nr   = np.ma.size(R,1)
    if np.size(Oldindex) == 0:
        nold = 0
    else:
        nold = Oldindex.shape[1]
    for i in range(nold):
        residual = np.abs(R-np.transpose(np.tile(Oldindex[:,i],(nr,1))))
        if residual.size != 0:
            fr = np.where(np.sum(residual,0)==0)
        else:
            fr = []
        R[:,fr] = 0

    R    = np.delete(R,np.where(np.sum(np.abs(R),0)==0),1)
    
    return R

def generate_index(L,D):

    nindex    = generate_nextindex(L,np.ones((D)),np.array([]))
    nnindex   = np.concatenate((np.ones((D,1)),nindex),1)
    tempindex = np.array([])
    while 1:
        for i in range(np.size(nindex,1)):
            tempindex = generate_nextindex(L,nindex[:,i],nnindex)
            nnindex   = np.concatenate((nnindex,tempindex),1)

        if np.size(tempindex) == 0:
            break

        nindex = nnindex

    R = nnindex 

    return R

def generate_point(L,current_index,Point_SI,Manner):

    D = np.size(current_index)

    q = np.sum(current_index) - D

    if q >= L-D and q <= L-1:

        [pt,w] = get_1d_point(current_index[0],Point_SI[0],Manner)

        for i in range(1,D):
            [npt, nw] = get_1d_point(current_index[i],Point_SI[i],Manner)

            num_npt = np.size(nw)

            num_pt = w.shape[1]

            pt     = np.tile(pt,(1,num_npt))
            pt_add = np.repeat(npt,num_pt)
            pt_add = np.transpose(pt_add.flatten())[np.newaxis,:]
            pt     = np.concatenate((pt,pt_add),0)
            
            w      = np.tile(w,(1,num_npt))    
            w_add  = np.repeat(nw,num_pt)  
            w_add  = np.transpose(w_add.flatten())[np.newaxis,:]
            w      = np.concatenate((w,w_add),0)

        if w.shape[0] != 1:
        
            w = np.prod(w,0)
            w = w*(factorial(D-1)/(factorial(L-1-q)*factorial((D-1)-(L-1-q))))*(-1)**(L-1-q)

    else:
        pt = np.array([])
        w  = np.array([])

    return pt, w

def generate_md_points(L,D,Manner):
    # L: accuracy level.
    # D: # of Dimension.
    # Manner: increase manner, L, 2L-1, or something else.
    # coded by bin jia @ 2012.

    Index_set = generate_index(L, D)
    nl        = Index_set.shape[1]
    
    pt       = np.array([])
    w        = np.array([])
    Point_SI = np.ones((D))

    for i in range(nl):
        [tpt,tw] = generate_point(L,Index_set[:,i],Point_SI,Manner)
        ntpt     = np.size(tw)
        npt      = np.size(w)

        if np.size(pt) == 0:
            if np.size(pt) != 0:
                pt = np.concatenate((pt,tpt),1)
            else:
                pt = tpt
                
            if np.size(w) != 0:
                w  = np.concatenate((w,tw),1)
            else: 
                w = tw
                
        else:
            tdelindex = []
            
            for j in range(ntpt):
                residual = np.abs(pt - np.transpose(np.tile(tpt[:,j],(npt,1))))
                
                fi       = np.where(np.sum(residual,0) < 1e-6)[0]
                
                if np.size(fi) > 1:
                    print('Something Wrong')
                if np.size(fi) != 0:
                    w[fi]      = tw[j] + w[fi]
                    tdelindex  = tdelindex+[j]
            
            tpt              = np.delete(tpt,tdelindex,1)
            tw               = np.delete(tw,tdelindex,0)
            pt               = np.concatenate((pt,tpt),1)
            w                = np.concatenate((w,tw),0)

    return pt, w               