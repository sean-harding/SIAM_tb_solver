#Contains helper functions to map a many-body bitstring to a hash index and back again
#Author: Sean M. Harding (University of Bristol,2018)

import scipy as sp
def binomial(N):
    binomials={(0,0):0,(0,1):1,(1,1):1}
    for n in range(2,N+1):
        binomials[(0,n)] = 1
        binomials[(n,n)] = 1
        for i in range(1,n):
            binomials[(i,n)] = binomials[(i-1,n-1)] + binomials[(i,n-1)]
    return binomials
'----------------------------------------------------------------------------' 
def getIndex(nSites,phi,binomials):
    '''Hashing function to convert between Fermion bit string and many body basis index, in increasing order
    of binary value'''  
    idx=0;m=0    
    for i in range(0,nSites):
        if phi[i]==1: 
            m=m+1 
            idx = idx + binomials[(m,i+1)]
    return idx
'----------------------------------------------------------------------------' 
def getConfig(n,nSites,index,binomials):
    '''Inverts the map in getIndex. There are n particles'''
    m = n; p=index
    cfig = sp.zeros(nSites)
    for i in range(nSites-1,-1,-1):
        if p >= binomials[(m,i+1)]:
            p = p - binomials[(m,i+1)]
            m = m - 1
            cfig[i] = 1    
        cfig = cfig
    return cfig