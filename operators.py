#Contains functions to construct operators used in many-body quantum problems
#Author: Sean M. Harding (University of Bristol,2018)

import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import helperFunctions as f

def getImpOperators(cfigs,nSites,imp):
    #Only construct up spin c/a operators right now
    n_up = int(nSites/2)
    n_dn = n_up
    binomials = f.binomial(nSites+1)
    HSdim = len(cfigs)
    cfigsPlus = []
    cfigsMinus = []
    occupancies_plus = []
    occupancies_minus = []
    creation_pairs = []
    annihilation_pairs = []
    #Construct the N+1 and N-1 particle sectors by adding an up spin to every site. 
    #Record whethere the impurity site imp is occupied or not with an up spin
    for k in range(len(cfigs)):
        c_up = f.getConfig(n_up,nSites,cfigs[k]['up'],binomials)
        c_dn = f.getConfig(n_dn,nSites,cfigs[k]['dn'],binomials)
        occupied = sp.where(c_up==1)[0]
        unoccupied = sp.where(c_up==0)[0]
        for site in unoccupied:
            c_new = sp.copy(c_up)
            c_new[site] = 1
            newUp = f.getIndex(nSites,c_new,binomials)
            index = newUp*binomials[n_dn,nSites+1] + cfigs[k]['dn']
            cfigsPlus.append([newUp,cfigs[k]['dn'],index])
            N = [newUp,cfigs[k]['dn'],index]
            occupancies_plus.append([c_up[imp],c_dn[imp],c_new[imp]])
            if site==imp:
                fermiString = (-1)**(sp.sum(c_up[0:imp]))
                creation_pairs.append([k,index,fermiString])    #These are states in different sectors linked by a creation operator
        for site in occupied:
            c_new = sp.copy(c_up)
            c_new[site] = 0
            newUp = f.getIndex(nSites,c_new,binomials)
            index = newUp*binomials[n_dn,nSites+1] + cfigs[k]['dn']
            cfigsMinus.append([newUp,cfigs[k]['dn'],index])
            occupancies_minus.append([c_up[imp],c_dn[imp],c_new[imp]])
            if site==imp:
                fermiString = (-1)**(sp.sum(c_up[0:imp]))
                annihilation_pairs.append([k,index,fermiString])
    #Convert dtype to scipy arrays
    cfigsMinus = sp.array(cfigsMinus,dtype=object)
    cfigsPlus = sp.array(cfigsPlus,dtype=object)
    occupancies_minus = sp.array(occupancies_minus)
    occupancies_plus = sp.array(occupancies_plus)
    creation_pairs = sp.array(creation_pairs,dtype=object)
    annihilation_pairs = sp.array(annihilation_pairs,dtype=object)
    uniques_plus = sp.unique(cfigsPlus[:,2],return_index=True)      #Remove duplicates
    uniques_minus = sp.unique(cfigsMinus[:,2],return_index=True)
    cfigsPlus = cfigsPlus[uniques_plus[1]]
    cfigsMinus = cfigsMinus[uniques_minus[1]]
    dimPlus = len(cfigsPlus)
    dimMinus = len(cfigsMinus)
    basis_plus = sp.zeros(dimPlus,dtype=[('up',object),('dn',object),('idx',object)])
    basis_minus = sp.zeros(dimMinus,dtype=[('up',object),('dn',object),('idx',object)])
    basis_plus['up'] = cfigsPlus[:,0]
    basis_plus['dn'] = cfigsPlus[:,1]
    basis_plus['idx'] = cfigsPlus[:,2]
    basis_minus['up'] = cfigsMinus[:,0]
    basis_minus['dn'] = cfigsMinus[:,1]
    basis_minus['idx'] = cfigsMinus[:,2]
    #Get the occupancies of up and down spin in the new particle sectors for num. operators
    occupancies_plus_up = occupancies_plus[uniques_plus[1]][:,2]
    occupancies_plus_dn = occupancies_plus[uniques_plus[1]][:,1]
    occupancies_minus_up = occupancies_minus[uniques_minus[1]][:,2]
    occupancies_minus_dn = occupancies_minus[uniques_minus[1]][:,1]
    indices_plus = sp.linspace(0,dimPlus-1,dimPlus)
    indices_minus = sp.linspace(0,dimMinus-1,dimMinus)
    #Hash to array index dictionaries
    indices_plus = {}
    indices_minus = {}
    for k in range(0,dimPlus):
        indices_plus[basis_plus['idx'][k]] = k  
    for k in range(0,dimMinus):
        indices_minus[basis_minus['idx'][k]] = k  
    for pair in creation_pairs:
        pair[1] = indices_plus[pair[1]]
    for pair in annihilation_pairs:
        pair[1] = indices_minus[pair[1]]
    c_up = sp.sparse.coo_matrix((creation_pairs[:,2],(creation_pairs[:,0],creation_pairs[:,1])),shape=[HSdim,dimPlus],dtype='float64')
    a_up = sp.sparse.coo_matrix((annihilation_pairs[:,2],(annihilation_pairs[:,0],annihilation_pairs[:,1])),shape=[HSdim,dimMinus],dtype='float64')
    n_up_plus = sp.sparse.diags(occupancies_plus_up)
    n_dn_plus = sp.sparse.diags(occupancies_plus_dn)
    n_dn_min = sp.sparse.diags(occupancies_minus_dn)
    n_up_min = sp.sparse.diags(occupancies_minus_up)
    c_up = c_up.tocsr()             #csr format is faster for matrix multiplication
    a_up = a_up.tocsr()
    n_up_plus = n_up_plus.tocsr()
    n_dn_plus = n_dn_plus.tocsr()
    n_dn_min = n_dn_min.tocsr()
    n_up_min = n_up_min.tocsr()
    return n_up_plus,n_up_min,n_dn_plus,n_dn_min,c_up.transpose(),a_up.transpose(),basis_plus,basis_minus

def getKinetic(cfgs,sites,binomials,nUp,nDn,neighbors_1,tij_1,neighbors_2,tij_2):
    'Returns the kinetic part of the Hamiltonian'
    L = len(sites)
    HSdim = len(cfgs)
    I = []
    J = []
    Hij = []
    E = []

    #Lookup dictionary to speedup getting matrix elements
    D = {}
    for i in range(len(cfgs)):
        D[cfgs['idx'][i]] = i

    energies_1 = np.diag(tij_1)
    energies_2 = np.diag(tij_2)
    for i in range(HSdim):
        cfg_up = f.getConfig(nUp,L,cfgs[i]['up'],binomials)
        cfg_dn = f.getConfig(nDn,L,cfgs[i]['dn'],binomials)
        #Get the occupied sites for both flavour of spin
        occ_up = sites[cfg_up==1]
        occ_dn = sites[cfg_dn==1]
        #Get the diagonal matrix element Hii
        energy = np.sum(energies_1[cfg_up==1]) + np.sum(energies_2[cfg_dn==1])
        E.append(energy)
        #Up spins
        for n in range(nUp):
            s = int(occ_up[n])               #Site, s
            nns = neighbors_1[s]
            for j in range(len(nns)):
                k = int(nns[j])              #Neighbor, k
                if cfg_up[k] == 0:      #If k unoccupied, do hop
                    cNew = np.copy(cfg_up)    
                    cNew[s] = 0
                    cNew[k] = 1
                    hsh = f.getIndex(L,cNew,binomials)*binomials[(nDn,L+1)] + cfgs[i]['dn']        #Get FULL index of new state 
                    try:            #If hash is not in the dictionary the code in the try block will throw an exception
                        mIndex = D[hsh]
                        phase = (-1.0)**int( np.sum ( cNew[ np.min((s,k))+1 : np.max((s,k)) ]))  #Compute Fermi phase factor
                        Hij.append(phase*tij_1[s,k])  #Use python lists rather than numpy arrays as these are very very slow when appending
                        I.append(i)
                        J.append(mIndex)  
                    except:
                        pass

        #We treat up and down differently to obtain the correct spectral properties
        #Down spins
        for n in range(nDn):
            s = int(occ_dn[n])
            nns = neighbors_2[s]
            for j in range(len(nns)):
                k = int(nns[j])
                if cfg_dn[k] == 0:
                    cNew = np.copy(cfg_dn)
                    cNew[s] = 0
                    cNew[k] = 1
                    hsh =  cfgs[i]['up']*binomials[(nDn,L+1)] + f.getIndex(L,cNew,binomials)
                    try:
                        mIndex = D[hsh] 
                        phase = (-1.0)**int( np.sum ( cNew[ np.min((s,k))+1 : np.max((s,k)) ]))                                                       
                        Hij.append(phase*tij_2[s,k])
                        I.append(i)
                        J.append(mIndex) 
                    except:
                        pass                                
                        
    Tij = sp.sparse.coo_matrix((Hij,(I,J)),shape=(HSdim,HSdim))
    H = sp.sparse.diags(E,0) + Tij
    return H
    
def numberOperators(cfigs,N_up,N_dn,L,binomials,imp):
    HSdim = len(cfigs)
    oNumbers = np.zeros(HSdim,dtype = [('up','bool'),('dn','bool')])
    
    for k in range(HSdim):
        c_up = f.getConfig(N_up,L,cfigs[k]['up'],binomials)
        c_dn = f.getConfig(N_dn,L,cfigs[k]['dn'],binomials)
        
        if c_up[imp] == 1: 
            oNumbers['up'][k] = True
        if c_dn[imp] == 1:
            oNumbers['dn'][k] = True
    
    ind = np.linspace(0,HSdim-1,HSdim)
    up_ind =  ind[oNumbers['up']]
    dn_ind = ind[oNumbers['dn']]
    
    num_up = sp.sparse.coo_matrix((np.ones(len(up_ind)),(up_ind,up_ind)),shape=(HSdim,HSdim))
    num_dn = sp.sparse.coo_matrix((np.ones(len(dn_ind)),(dn_ind,dn_ind)),shape=(HSdim,HSdim))
    
    num_up = num_up.tocsr()
    num_dn = num_dn.tocsr()

    return num_up,num_dn

def DM(cfigs,GS,nUp,nDn,nSites,imp,what='up'):
    "Computes one-body density matrix"
    if not (what=='up' or what=='dn'):
        raise Exception('"what" should be given as  "up" or "dn"')
    occupancy = []
    B = f.binomial(nSites+1)
    hashList={}
    k=0
    sites=sp.linspace(0,nSites-1,nSites,dtype='uint')
    for cfig in cfigs:
        c_up = f.getConfig(nUp,nSites,cfig[what],B)
        occupancy.append(c_up[sites])
        hashList[cfig['idx']]=k
        k=k+1
    occupancy = sp.array(occupancy)
    rho = sp.zeros([len(sites),len(sites)])

    for site1 in sites:
        for site2 in sites:
            s2 = max(site1,site2)
            s1 = min(site1,site2)
            if site1!=site2:
                occupied_1 = sp.where(occupancy[:,site1]==1)[0]                #Where are the configurations with site 1 occupied
                unoccupied_2 = sp.where(occupancy[:,site2]==0)[0]              #Where are the configurations with site 2 unoccupied
                possiblePairs = sp.intersect1d(occupied_1,unoccupied_2)        #Configurations where both are satisfied. 
                for index in possiblePairs:
                    c_up = f.getConfig(nUp,nSites,cfigs[index][what],B)
                    c_up[site1]=0
                    c_up[site2]=1
                    idxNew = f.getIndex(nSites,c_up,B)*B[nUp,nSites+1] + cfigs[index]['dn']
                    idxNew = int(idxNew)

                    try:
                        s2 = max(site1,site2)
                        s1 = int(min(site1,site2)+1)
                        match = hashList[idxNew]
                        fermiPhase = (-1)**sp.sum(occupancy[index][s1:s2])
                        rho[site1][site2] = rho[site1][site2]+GS[0,index]*GS[0,match]*fermiPhase
                    except:
                        pass
            else:
                n=0
                while n<len(cfigs):
                    rho[site1][site1] = rho[site1][site1]+occupancy[n][site1]*GS[0,n]*GS[0,n]
                    n=n+1
    return rho