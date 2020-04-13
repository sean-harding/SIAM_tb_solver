import scipy as sp
import helperFunctions as f
import schmidtForm as mapping
import operators
import solver
import matplotlib.pyplot as py
import DMFT_functions as DMFT
from math import pi

def configurations(L,noParticles,oAMax,spin,iCfig):
    'List all possible configurations for N particles in S sites with a max occupancy of oAMax'
    Cnow = iCfig                                                              #To hold current configuration
    global configs                                                            #To store generated configurations
    configs = sp.zeros((0,2*L),dtype = 'uint')                                #Initialize
    
    if spin == 'up':
        loopsites(0,L,oAMax,Cnow,noParticles,configs,0)
    elif spin== 'dn':
        loopsites(L,2*L,oAMax,Cnow,noParticles,configs,L)

    #Remember to reshape the output AFTER calling the function in the following manner
    '''
    configs = sp.reshape(configs,[len(configs),2*L])
    for config in configs:
        config = sp.reshape(config,[L,L])
    '''
    return configs                               #Loops over allowed occupancies for a given site
'----------------------------------------------------------------------------'  
def loopsites(site,L,oAmax,Cnow,N,cfigs,start):
    'Loops over all possible occupations for a particular site'
    global configs;configs=cfigs                   
    if site != L-1:                                                             
        for x in range(0,1+int(min(N-sp.sum(Cnow[start:site]),oAmax))):   #Run over all possible occupancies
            Cnow[site] = x
            loopsites(site+1,L,oAmax,Cnow,N,configs,start)                #Move onto next site
    
    elif N-sp.sum(Cnow[start:site])<=oAmax:                               #Set last occupation only if all previous occupations are chosen            
        Cnow[site] = N-sp.sum(Cnow[start:site])
        configs = sp.append([configs],[Cnow])                             #Output
'----------------------------------------------------------------------------'  

U=3.0
nSites=6
N_up = int(nSites/2)
N_dn = N_up
nBath = nSites-1
imp = [0]
binomials = f.binomial(nSites+1)
E = 1.0

#Bath 1
D1=2
x = sp.linspace(-1.2*D1,1.2*D1,6000)
y = 2*sp.sqrt(1-(x**2)/D1**2)/(pi*D1)
E_bath_1,V_bath_1 = DMFT.linearDisc(y,nBath,x)

y1 = sp.copy(y)
#Bath 2
D2=0.5*D1
#x = sp.linspace(-1.2*D2,1.2*D2,6000)
y = 2*sp.sqrt(1-(x**2)/D2**2)/(pi*D2)
E_bath_2,V_bath_2 = DMFT.linearDisc(y,nBath,x)

'C2 SYMMETRIC MODEL'
CFS = 0.0
t=0.4

H_imp = sp.zeros([2,2])
H_imp[0,0] = CFS/2 + e0
H_imp[1,1] = -1*CFS/2 + e0
H_imp[0,1] = t
H_imp[1,0] = t

E_bath = sp.append(E_bath_1,E_bath_2)
H_bath = sp.diag(E_bath)
H_1p = sp.linalg.block_diag(H_imp,H_bath)
H_1p = sp.matrix(H_1p)

#Add bath coupling 1
H_1p[0,2:2+len(E_bath_1)] = V_bath_1/2
H_1p[1,2:2+len(E_bath_1)] = V_bath_1/2
H_1p[2:2+len(E_bath_1),0] = V_bath_1/2
H_1p[2:2+len(E_bath_1),1] = V_bath_1/2

#Add bath coupling 2
H_1p[0,2+len(E_bath_1):len(H_1p)] = -1*V_bath_2/2
H_1p[2+len(E_bath_1):len(H_1p),0] = -1*V_bath_2/2
H_1p[1,2+len(E_bath_1):len(H_1p)] = V_bath_2/2
H_1p[2+len(E_bath_1):len(H_1p),1] = V_bath_2/2

py.matshow(H_1p,cmap='hot')
py.show()

#Build configurations list
initial = sp.zeros(nSites)
configs_1p = configurations(nSites,N_up,1,'up',initial)
N = int(len(configs_1p)/nSites)
MF_solution = sp.reshape(configs_1p,[N,nSites])
MF_hashes = []
particleNumber = []
for cfg in MF_solution:
    MF_hashes.append(f.getIndex(nSites,cfg,binomials))
    particleNumber.append(cfg[imp])
cfgs_CI = []
P = {}
for i in range(len(MF_hashes)):
    P[MF_hashes[i]] = particleNumber[i]
sector = []
for hash1 in MF_hashes:
    for hash2 in MF_hashes:
        cfgs_CI.append([hash1,hash2,hash1*binomials[N_up,nSites+1]+hash2])
        sector.append([P[hash1],P[hash2]])
cfgs_CI = sp.array(cfgs_CI)
cfigs = sp.zeros(len(cfgs_CI),dtype=[('up',object),('dn',object),('idx',object)])
cfigs['up'] = cfgs_CI[:,0]
cfigs['dn'] = cfgs_CI[:,1]
cfigs['idx'] = cfgs_CI[:,2]

HSdim = len(cfigs)

'''
cfigs_CI = cfigs[0:1]
sites = sp.linspace(0,nSites-1,nSites)
HSdim = 0
while HSdim<36:
    HSdim = len(cfigs_CI)
    cfigs_new = solver.expandBasis(cfigs_CI,binomials,N_up,N_dn,sites,nSites,neighbors,neighbors,HSdim)
    cfigs_new = cfigs_new[sp.unique(cfigs_new['idx'],return_index=True)[1]]
    cfigs_CI = sp.append(cfigs_CI,cfigs_new)
    cfigs_CI = cfigs_CI[sp.unique(cfigs_CI['idx'],return_index=True)[1]]
    print('Size of basis = {}'.format(len(cfigs_CI)))
'''

neighbors = mapping.neighborTable(H_1p)
sites = sp.linspace(0,nSites-1,nSites)
H_kin = operators.getKinetic(cfigs,sites,binomials,N_up,N_dn,neighbors,H_1p,neighbors,H_1p)
(num_up_1,num_dn_1) = operators.numberOperators(cfigs,N_up,N_dn,nSites,binomials,imp[0])                   
H_SIAM = H_kin + U*(num_up_1-0.5*sp.sparse.eye(HSdim))*(num_dn_1-0.5*sp.sparse.eye(HSdim)) 

En,state = sp.sparse.linalg.eigsh(H_SIAM,which='SA')
ground_state = state[:,0]
ground_state = sp.matrix(ground_state)
GSE_exact = En[0]
B = f.binomial(nSites+1)
n_up,n_dn = operators.numberOperators(cfigs,N_up,N_dn,nSites,B,0)

dens_up = ground_state*num_up_1*ground_state.transpose()
dens_dn = ground_state*num_dn_1*ground_state.transpose()

print('GSE = {}'.format(GSE_exact))
print('Densities:')
print('Up: {}'.format(dens_up))
print('Dn: {}'.format(dens_dn))
print('Exact calculation complete')
#sp.savez('exact',H_1p=H_1p,GSE_exact=GSE_exact,HSmax=HSdim)
#print('Exact calculation complete')
