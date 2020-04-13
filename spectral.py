import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib
import scipy.sparse.linalg
from math import pi,sin,cos,tan,sqrt,sinh
import matplotlib.pyplot as py
import time as t

def moments(H,GS):
    '''Compute Chebyshev moments'''
    yield (GS.transpose()*GS).todense()[0,0]
    T1,T0 = H*GS,GS
    yield (GS.transpose()*T1).todense()[0,0]
    while True:
        T2 = 2*H*T1-T0
        T0,T1 = T1,T2
        yield (GS.transpose()*T2).todense()[0,0]
def cheb(freq):
    '''Computes Chebyshev polynomials. The first one comes with a factor
    of 0.5 to simplify the code for computing the spectral function''' 
    t0,t1 = sp.ones(len(freq)),freq
    yield t0*0.5
    yield t1
    while True:
        t2 = 2*freq*t1-t0
        yield t2
        t0,t1 = t1,t2

def linearPrediction(moments,first,how_many):
    '''Predicts how_many more chebyshev moments by a linear iterative scheme, from a test_set'''
    first = int(first)
    L = len(moments[first:])    #Train on the set first:end of size L
    p = min(int(L/2),50)        #Use a window of size p: i.e. for each moment first:L, predict that moment using the previous p moments
    R = sp.zeros([p,p],dtype='float64')
    X=sp.zeros(p,dtype='float64')
    print('Window size:{}\nTraining set size:{}\nFirst moment:{}'.format(p,L,first))
    #Set up the relevant matrices which are required
    for i in range(p):
        for j in range(p):
            n=0
            while n<L:
                R[i,j] = R[i,j]+ moments[first+n-j-1]*sp.conj(moments[first+n-i-1])
                n+=1
    for i in range(p):
        n=0
        while n<L:
            X[i] = X[i] + moments[first+n]*sp.conj(moments[first+n-i-1])
            n+=1
    R = -1*sp.matrix(R) + sp.eye(len(R))*(10**-6)
    X = -1*sp.matrix(X).transpose()
    A = -1.0*sp.linalg.inv(R)*X
    M = sp.eye(len(A)-1)
    M = sp.hstack([M,sp.zeros([len(M),1])])
    M = sp.vstack([-1*A.transpose(),M])
    M = sp.matrix(M)
    moving_set = sp.copy(moments[-p:len(moments)])
    moving_set = sp.flip(moving_set,axis=0)
    moving_set = sp.matrix(moving_set).transpose()
    weight = sp.linalg.norm(M*moving_set)
    eVals,V_l,V_r = sp.linalg.eig(M,left=True,right=True)
    V_l = sp.matrix(V_l).getT()
    V_l = sp.matrix(V_l)
    M = V_l*M*V_l.getI()
    diagonals = sp.diag(M).copy()
    divergent = sp.where(np.abs(diagonals)>=1)[0]
    print('Number of divergent eigenvalues is {}'.format(len(divergent)))
    for d in divergent:
        #M[d,d]=0.0+0*1j                #Various protocols to deal with divergent eigenvectors
        M[d,d]=0.99*M[d,d]/abs(M[d,d])
        #M[d,d]=M[d,d]/abs(M[d,d])
    M = V_l.getI()*M*V_l
    new_weight = sp.linalg.norm(M*moving_set)
    print('Discarded weight = %.3f percent'%float((weight-new_weight)*100/weight))
    new_moments = []
    k=0
    while k<how_many:                       #A robust modification would be to turn this into a generator to compute moments
        moving_set = M*moving_set           #on the fly, or to just iterate until the moments decay to zero
        new_moments.append(moving_set[0,0])
        k = k+1
    return sp.real(new_moments)

#Load in matrices
print('Loading matrices...')
operators = sp.load('operators_U=1.5hyb_60.npz')
H_plus = operators['H_plus']
H_minus = operators['H_minus']
a_up = operators['a_up']
c_up = operators['c_up']
GS = operators ['GS']
GS = sp.sparse.csr_matrix(GS)
E = operators['GSE']
E = float(E)

H_plus = sp.reshape(H_plus,(1,))[0]
H_minus = sp.reshape(H_minus,(1,))[0]
a_up = sp.reshape(a_up,(1,))[0]
c_up = sp.reshape(c_up,(1,))[0]
GS = GS.transpose()
Delta = operators['Delta']
'-------------------------------|Rescale|---------------------------------'
a = 40    #Rescaling factor    
b = 0.9*a    #Shift

nMoments = 350  #Ensure this is even as we only predict the even moments                                         
print('Rescaling matrices')
dPlus = sp.shape(H_plus)[0]
dMinus = sp.shape(H_minus)[0]
H_plus_rescaled = (H_plus - sp.sparse.eye(dPlus)*(E+b))/a
H_minus_rescaled = (H_minus - sp.sparse.eye(dMinus)*(E+b))/a  
'Frequency mesh in units of half-bandwidth,rescaled'
fMax = 0.6              #Units of half-bandwidth
frequencies =  sp.linspace(-1.0*fMax,fMax,5000)
frequencies_rescaled = [(frequencies - b)/a,(frequencies + b)/a]
'-------------------------------|Chebyshev - common expansion|---------------------------------'
print('Plotting spectral function')
if b==0:
    greater = moments(H_plus_rescaled,c_up*GS)
    lesser = moments(H_minus_rescaled,a_up*GS)
else:
    greater = moments(H_plus_rescaled,c_up*GS)
    lesser = moments(-1*H_minus_rescaled,a_up*GS)

moments_greater = [next(greater) for k in range(400)]   #Convert to genexp when we don't want to store these
moments_lesser = [next(lesser) for k in range(400)]

nTot =1000
if b==0:
    moments = [moments_greater[n] + moments_lesser[n]*(-1)**n for n in range(len(moments_greater))]
    cheby =cheb(frequencies_rescaled[0])
    w = 2/(pi*sp.sqrt(1-frequencies_rescaled[0]**2))
    A = sum(mu*w*next(cheby) for mu in moments)/a
    moments_LP = linearPrediction(moments[::2],int(0.25*len(moments)),nTot-len(moments))
    moments_LP = [a for b in zip(moments_LP,[0]*len(moments_LP)) for a in b]    #Convert to generator expression
    A_LP = A+sum(mu*w*next(cheby) for mu in moments_LP)/a                       #When we don't want to store these
    sp.savez('A_test',A)
else:
    W = [2/(pi*sp.sqrt(1-freq**2)) for freq in frequencies_rescaled]
    cheby = [cheb(freq) for freq in frequencies_rescaled]
    A = map(sum,((next(ch)*k*w for k in co) for (co,ch,w) in zip([moments_greater,moments_lesser],cheby,W)))
    A = sum(A)/a
    moments_LP = (linearPrediction(m[::2],int(0.25*len(m)),nTot-len(m)) for m in [moments_greater,moments_lesser])
    A_LP = sp.copy(A)
    for coeffs,geny,ker in zip(moments_LP,cheby,W):
        mom = (n for b in zip(coeffs,[0]*len(coeffs)) for n in b)
        A_LP = A_LP + sum(next(geny)*ker*x for x in mom)/a
if b==0:
    colours = sp.full_like(sp.zeros([1,len(moments)]),['r'],dtype='str')
    colours = sp.append(colours,sp.full_like(sp.zeros([1,len(moments_LP)]),['b'],dtype='str'))
    figs,ax = py.subplots(2,1)
    ax[0].scatter(sp.linspace(1,len(moments_LP),len(moments+moments_LP)),moments+moments_LP,c=colours)
    ax[0].plot(moments+moments_LP,'ro')
    ax[1].plot(frequencies/Delta,A*pi*Delta)
    ax[1].plot(frequencies/Delta,A_LP*pi*Delta)
    ax[1].xlabel('Frequency/Δ')
    ax[1].ylabel('AΔπ')
else:
    py.plot(frequencies/Delta,A*pi*Delta)
    py.plot(frequencies/Delta,A_LP*pi*Delta)
    ax[1].xlabel('Frequency/Δ')
    ax[1].ylabel('AΔπ')
py.show()