#Main thread for performing single-site DMFT on the Hubbard model using a truncated Hilbert space impurity solver
#and the Chebyshev method for calculating spectral functions
#Author: Sean M. Harding, (University of Bristol, 2018)

import scipy as sp                  
import scipy.sparse                 #Have to import sublibraries independently
import scipy.sparse.linalg
import scipy.integrate
import matplotlib.pyplot as py
import time as t
from math import pi,sqrt

import helperFunctions as f
import operators
import solver

#Open files for output. In this version of the code I output to a text file rather than command line
out = open('out.txt','w')
def discretizeFunc(func,numSamples,bandwidth,tpe='lin',logFactor=1.2):
    if tpe=='lin':
        dx = 2*bandwidth/numSamples
        bins = [-bandwidth+dx*n for n in range(0,numSamples+1)]
    elif tpe=='log':
        N=(numSamples+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        bins = bins+bins2
    elif tpe=='linlog':
        n=9
        N=(numSamples-n+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        dx = (bins[-1]-bins2[0])/n
        dx = abs(dx)
        bins3 = [bins[-1]+dx*k for k in range(1,n)]
        bins = bins+bins3+bins2
    samples = [sp.integrate.quad(func,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = [sp.integrate.quad(lambda x:func(x)*x,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = map(lambda x,y:x/y,energies,samples)
    samples = map(sqrt,samples)
    return list(energies),list(samples)

#----------------------------------|PARAMETERS|-------------------------------------
save_entropies = False                #If true, the program also computes entanglement entropies for excited states.
#------|Model parameters|---          #I always compute the entropy for the ground state
nBath = 39                            #In this version of the code, choose nBath=odd so nTotal = even for convenience
N_up = int(sp.ceil((nBath+1)*0.5))    #Number of one spin flavour
N_dn = int(sp.floor((nBath+1)*0.5))   #Number of the other spin flavour         
#-----|Algorithm parameters|-----
cTol = 10**-8                         #Tolerance for determining whether a state contributes to the ground state. In this version, this is the bare amplitude
#------|Other convenient parameters|-----
N_tot = N_up + N_dn                            #Total number of particles
nSites = nBath +1                              #Total number of sites
sites = sp.linspace(0,nSites-1,nSites)         #Site indices
binomials = f.binomial(nSites+1)               #Compute binomial coefficients for bitstring hash functions
ePrev = 10.0                                   #Monitor convergence of ground state. Initialized to random large values

#For now, we only work at half-filling
if nSites%2 !=0:
    raise Exception('The number of total sites should be even. The total number of sites is set to {}'.format(nSites))

#Initialize bath - flat DoS
D = 1
couplingStrength = 0.1

#Initialize bath - SE DoS
#DoS = lambda x: 2*couplingStrength*sqrt(1-x**2)/pi

#Bath - Flat DoS
DoS = lambda x: couplingStrength/sqrt(nBath)

Eb,Vb = discretizeFunc(DoS,nBath,D,tpe='lin',logFactor=1.8)
Delta = pi*sum([x**2 for x in Vb])/2 #Multiply by value of DoS at Ef. Spectral function is then pinned to 1/pi*delta
H_bath = sp.diag(Eb)
H_1p = sp.linalg.block_diag([0],H_bath)
H_1p[0,1:1+len(Eb)] = Vb
H_1p[1:1+len(Eb),0] = Vb

U = 4*pi*Delta                #Hubbard interaction
zvals = [0.2]                 #Rescaling parameter for hybridization
basis_size =  []              #Record basis size and energy for different z
E = []
for z in zvals:
    #Calculate density matrices from which we wish to obtain a Schmidt basis
    H_bath = sp.diag(Eb)
    H_r = sp.linalg.block_diag([0],H_bath)
    H_r[0,1:1+len(Eb)] = [v*z for v in Vb]
    H_r[1:1+len(Eb),0] = [v*z for v in Vb]
    (en,eV) = sp.linalg.eigh(H_r)
    sort = sp.argsort(en)
    eV=eV[:,sort]
    dens_matrix = sp.matmul(eV[:,0:N_up],eV[:,0:N_dn].transpose())
    (GSE,GS,cfigs,dens_up,dens_dn,neighbors,neighbors,H_2c,H_2c,imp) = solver.solve(N_up,N_dn,nSites,U,cTol,H_1p,dens_matrix,'out.txt')
    basis_size.append(len(cfigs))
    E.append(float(GSE))
    #Calculate operators in the N+1 and N-1 particle number basis for spectra
    n_up_plus,n_up_min,n_dn_plus,n_dn_min,c_up,a_up,basis_plus,basis_minus = operators.getImpOperators(cfigs,nSites,imp)
    H_plus = operators.getKinetic(basis_plus,sites,binomials,N_up+1,N_dn,neighbors,H_2c,neighbors,H_2c) +U*(n_up_plus*n_dn_plus) - 0.5*U*(n_up_plus+n_dn_plus)
    H_minus = operators.getKinetic(basis_minus,sites,binomials,N_up-1,N_dn,neighbors,H_2c,neighbors,H_2c)+U*(n_up_min*n_dn_min) - 0.5*U*(n_up_min+n_dn_min)
    sp.savez('operators_U=4hyb_40_log_hr',GS=GS,c_up=c_up,a_up=a_up,H_plus=H_plus,H_minus=H_minus,GSE=GSE,Delta=Delta)
#sp.savez('basis_size_U=2*hyb_60',zvals=zvals,basis_size=basis_size,GSE=E)  #This is for investigating z scaling
print('Done')