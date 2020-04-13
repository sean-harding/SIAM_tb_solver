#The Configuration Interaction impurity solver and associated functions used in DMFT_main.py
#Author: Sean M. Harding (University of Bristol 2018)

import operators
import helperFunctions as f
import scipy as sp
import scipy.sparse 
import time as t
import schmidtForm as mapping
import matplotlib.pyplot as py
from math import pi

def lanczos(H,initVector):
    '''Compute Lanczos vectors for susceptibility calculations. The a and b variables
    are those appearing in the truncated fraction expansion of the susceptibility'''
    i=0
    v_new = initVector
    v_old = 0.0*initVector
    b = sp.sparse.linalg.norm(v_new)
    v_new = v_new/b
    v_old = v_new
    v_new = H*v_new
    a = (v_old.transpose()*v_new).todense()
    yield float(a),float(b)
    while True:
        v_new = v_new - float(a)*v_old
        b = sp.sparse.linalg.norm(v_new)
        v_new = v_new/float(b)
        v_old = v_new
        v_new = H*v_new
        a = (v_old.transpose()*v_new).todense()
        yield float(a),float(b)

def initialCfig(H_schmidt,site_occupancies,N_up):
    '''Given a hamiltonian in Schmidt form, H_schmidt, constructs the mean field ground state
    configurations as dictated by the occupation numbers'''
    nSites = len(H_schmidt)
    template = sp.zeros(len(H_schmidt))
    occupied = sp.where(site_occupancies>1-10**-8)
    partially_occupied = sp.where(site_occupancies>10**-8)[0]
    partially_occupied = sp.setdiff1d(partially_occupied,occupied)   #This gives the set exclusive.
    nPartial = len(partially_occupied)
    #partially_occupied = sp.setdiff1d(partially_occupied,imp)
    template[occupied[0]]=1
    MF_solution = []
    binomials = f.binomial(nSites+1)

    for site in partially_occupied:
        cfg_up = sp.copy(template)
        cfg_up[site] = 1
        if len(MF_solution)==0:
            MF_solution = cfg_up
        else:
            MF_solution = sp.vstack([MF_solution,cfg_up])
    MF_hashes = []
    for cfg in MF_solution:
        MF_hashes.append(f.getIndex(nSites,cfg,binomials))
    cfgs_CI = []
    for hash1 in MF_hashes:
        for hash2 in MF_hashes:
            cfgs_CI.append([hash1,hash2,hash1*binomials[N_up,nSites+1]+hash2])
    cfgs_CI = sp.array(cfgs_CI) #Use 'object' as datatype because the hashes are bigger than C longs
    cfigs = sp.zeros(len(cfgs_CI),dtype=[('up',object),('dn',object),('idx',object)])
    cfigs['up'] = cfgs_CI[:,0]
    cfigs['dn'] = cfgs_CI[:,1]
    cfigs['idx'] = cfgs_CI[:,2]
    return cfigs
'----------------------------------------------------------------------------'
def meanField(H_1p,U,N_up,N_dn,imp, return_dens=False):
    '''Single-impurity mean field tool'''
    a = 0.1         #Mixing parameter
    iMax = 1000      #Maximum number of iterations
    cTol = 10**-8   #Convergence tolerance for density
    densityPrev=0.0
    densityNew=0.0
    ePrev = 0.0
    H_new = sp.matrix(sp.copy(H_1p))
    eCurr = 1.0
    i = 0
    density = []
    field = []
    field.append(0)
    while abs(eCurr-ePrev) > cTol and i<iMax:
        i= i + 1
        ePrev = eCurr
        densityPrev = densityNew
        (en,eV) = sp.linalg.eigh(H_new)
        sort = sp.argsort(en)
        en=en[sort]
        eV=eV[:,sort]
        eCurr = sp.sum(en[0:N_up]) + sp.sum(en[0:N_dn])
        rho_up = sp.matmul(eV[:,0:N_up],eV[:,0:N_up].transpose())
        rho_dn = sp.matmul(eV[:,0:N_dn],eV[:,0:N_dn].transpose())
        #Impurity density
        densityNew = rho_up[imp,imp] + rho_dn[imp,imp]
        field.append(U*rho_up[imp,imp])
        #Construct mean field potential by mixing in new density
        H_new[imp,imp] = H_1p[imp,imp] + a*field[-1] + (1-a)*field[-2]
        density.append(densityNew)
    #print('HF density is {}'.format(rho_up[imp,imp]+rho_dn[imp,imp]))
    figs,ax= py.subplots(2,1)
    ax[0].plot(density,'ro')
    ax[1].plot(field,'bo')
    py.show()
    density = rho_up[imp,imp]
    if return_dens==False:
        print("Density: {}".format(density))
        return rho_up
    else:
        return density
'----------------------------------------------------------------------------'
def chemicalPotential(H_1p,N_up,N_dn,U,imp,nBath,lim1=-5,lim2=5,density=0.5,tol=10**-8):
    'Interval search for the impurity chemical potential which yields a certain impurity density'
    lower = max(lim1,lim2)
    upper = min(lim1,lim2)
    midpoint = lower + 0.5*(upper-lower)
    H_upper = sp.copy(H_1p) 
    H_mid = sp.copy(H_1p) 
    H_lower = sp.copy(H_1p) 

    H_upper[imp,imp] = H_upper[imp,imp] + upper
    H_mid[imp,imp] = H_mid[imp,imp] + midpoint
    H_lower[imp,imp] = H_lower[imp,imp] + lower
    
    dens_upper = meanField(H_upper,U,N_up,N_dn,imp,return_dens=True)
    dens_mid = meanField(H_mid,U,N_up,N_dn,imp,return_dens=True)
    dens_lower = meanField(H_lower,U,N_up,N_dn,imp,return_dens=True)
    while abs(dens_upper-dens_lower)>tol:
        if density<dens_mid:
            mu_lower,mu_upper = chemicalPotential(H_1p,N_up,N_dn,U,imp,lower,midpoint,nBath,density=density,tol=tol)
            return mu_lower,mu_upper
        elif density > dens_mid:
            mu_lower,mu_upper = chemicalPotential(H_1p,N_up,N_dn,U,imp,midpoint,upper,nBath,density=density,tol=tol)
        return mu_lower,mu_upper
    return lower,upper

'----------------------------------------------------------------------------'
def expandBasis(configs,binomials,n_up,n_dn,sites,nSites,neighbors_1,neighbors_2,HSdim):
    '''Gets all configurations reached from state by acting on it with H_1p, the 1-particle part of H_SIAM, in the 2-chain geometry
    As currently implemented, we generate all spin up hops from a reference state and all spin down hops from the same reference
    state. I have to do up and down hops separately as generally there will be an unequal number of up and down spins'''
    N = n_up + n_dn
    newHSdim = HSdim
    U = []
    D = []
    I = []
    d=0
    sites = sp.linspace(0,nSites-1,nSites)
    for d in range(len(configs)):
        #Get bit strings
        c_up = f.getConfig(n_up,nSites,configs[d]['up'],binomials)
        c_dn = f.getConfig(n_dn,nSites,configs[d]['dn'],binomials)      
        #Occupied and unoccupied sites
        up_occ = sites[c_up==1]
        dn_occ = sites[c_dn==1]
        #Perform hopping for up spins to generate new states
        for i in range(len(up_occ)):
            k = int(up_occ[i])
            hopping = neighbors_1[k]
            for j in range(len(hopping)):
                l = int(hopping[j])
                if c_up[l]==0:
                    c_up_n = sp.copy(c_up)
                    c_up_n[k] = 0
                    c_up_n[l] = 1
                    newHSdim = newHSdim+1
                    #Get new indices
                    U.append(f.getIndex(nSites,c_up_n,binomials))
                    D.append(f.getIndex(nSites,c_dn,binomials))
                    I.append(U[-1]*binomials[(n_up,N+1)]+D[-1])
        #Down spins
        for i in range(len(dn_occ)):
            k=int(dn_occ[i])
            hopping = neighbors_2[k]
            for j in range(len(hopping)):
                l = int(hopping[j])
                if c_dn[l] ==0:
                    c_dn_n = sp.copy(c_dn)
                    c_dn_n[k] = 0
                    c_dn_n[l] = 1
                    newHSdim = newHSdim+1
                #New indices
                    U.append(f.getIndex(nSites,c_up,binomials))
                    D.append(f.getIndex(nSites,c_dn_n,binomials))
                    I.append(U[-1]*binomials[(n_dn,N+1)]+D[-1])
    newCfigs = sp.zeros(len(I),dtype = [('up',object),('dn',object), ('idx',object)])
    newCfigs['up'] = U
    newCfigs['dn'] = D
    newCfigs['idx'] = I    
    return newCfigs
'----------------------------------------------------------------------------'
def solve(N_up,N_dn,nSites,U,cTol,H_1p,rho_MF,fle):
    if N_up!=N_dn:
        raise Exception('In this version of the code, the number of up and down spins should be equal')
    sites = sp.linspace(0,nSites-1,nSites)
    ePrev = 100
    rho_prev =10
    '''----------------|SCHMIDT BASIS|-------------------------'''
    U_s = mapping.schmidt(H_1p,rho_MF,1)
    H_2c = U_s.transpose()*H_1p*U_s
    rho_MF = U_s.transpose()*rho_MF*U_s
    imp=0
    site_occupancies=sp.diag(rho_MF)
    neighbors = mapping.neighborTable(H_2c)
    binomials = f.binomial(nSites+1)                                                     
    cfigs = initialCfig(H_2c,site_occupancies,N_up)
    HSdim = len(cfigs)
    '''----------------|TRUNCATED BASIS SOLVER|-------------------------'''
    t0 = t.perf_counter()   
    cfigs = initialCfig(H_2c,site_occupancies,N_up)
    converged =False
    ePrev = 100
    dens_prev = 0
    while converged!=True:
        out = open(fle,'a')
        out.write('\n----------------Enlarging Hilbert space----------------')
        cfigs_new = expandBasis(cfigs,binomials,N_up,N_dn,sites,nSites,neighbors,neighbors,HSdim)  
        HSdim_init =sp.copy(HSdim)
        cfigs_new = cfigs_new[sp.unique(cfigs_new['idx'],return_index=True)[1]]
        cfigs = sp.append(cfigs,cfigs_new)
        cfigs = cfigs[sp.unique(cfigs['idx'],return_index=True)[1]]
        #Compute Anderson Hamiltonian with particle-hole symmetrized interaction
        H_kin = operators.getKinetic(cfigs,sites,binomials,N_up,N_dn,neighbors,H_2c,neighbors,H_2c)                    
        (num_up,num_dn) = operators.numberOperators(cfigs,N_up,N_dn,nSites,binomials,imp)                   
        H_SIAM = H_kin + U*num_up*num_dn - (num_up+num_dn)*U*0.5
        GSE,GS = sp.sparse.linalg.eigsh(H_SIAM,k=1, which='SA')    
        GS=GS.reshape(len(GS))
        chop = sp.where(abs(GS)**2>cTol)[0]
        cfigs = cfigs[chop]                       #Get rid of unwanted configurations.
        GS=sp.matrix(GS)   
        GS = GS[0,chop]                           #Construct H in truncated Hilbert space and evaluate expectation values
        GS = GS/sp.linalg.norm(GS)
        H_SIAM = H_SIAM[:,chop]
        H_SIAM = H_SIAM[chop,:]
        num_up = num_up[:,chop]
        num_up = num_up[chop,:]
        num_dn = num_dn[:,chop]
        num_dn = num_dn[chop,:]
        GSE = GS*H_SIAM*GS.transpose()
        dens_up = GS*num_up*GS.transpose()
        dens_dn = GS*num_dn*GS.transpose()

        dE= abs(ePrev-GSE)
        dRho = abs(dens_up[0,0]+dens_dn[0,0]-dens_prev)        
        delta = len(cfigs)-HSdim
        toc = t.perf_counter()-t0

        HSdim = len(cfigs)
        dens_prev = dRho
        ePrev = GSE

        out.write('\n dE ={} '.format(dE))
        out.write('\nHilbert space dimension (post-truncation) = {} '.format(HSdim))
        out.write('\nGround state energy = {}'.format(float(GSE)))
        out.write('\nElapsed = {}'.format(toc))
        out.write('\nNo. new states = {}'.format(delta))
        out.write('\nTotal density = {}'.format(dens_up[0,0] + dens_dn[0,0]))
        out.write('\n dRho = {}'.format(dRho))
        if delta<=0 or dE<=cTol:    #Sometimes there is oscillation with dE>cTol.        
            converged =True         #Should take care for this although it does not happen often
                                    #Original fix: converged if basis size returns to the same value as 2 iterations ago
    return(GSE,GS,cfigs,dens_up,dens_dn,neighbors,neighbors,H_2c,H_2c,imp)
