#Calculates the entanglement entropy across a given bipartition
#Author: Sean M. Harding, University of Bristol (2019)

import scipy as sp
import scipy.sparse
import scipy.linalg
import helperFunctions as f

'''
STEPS:
1 - Reorganize so that the impurity + partially occupied state are in the middle
2 - for cfig[i] in cfigs:
    Split bitstring at bond n
    Get hash of bistring to the left and to the right
    Create dictionaries to map each sub-hash onto matrix index
    Save triplet, (hashA,hashB,value)
    Make matrix
'''
def coefficient_matrix(state,configs,site,nSites,n_up,n_dn,return_indices=False):
    'This function reshapes the configuration list for a given bipartition. I only have to do this once for a given'
    'basis, so I have defined it as a single function instead of doing it again for each state'
    B = f.binomial(nSites+1)
    size=len(configs)
    C_i = sp.zeros(size,dtype = [('left',object),('right',object),('amplitude','double'),('nUp_left',object),('nDn_left',object),('left_up',object)])
    for j in range(len(configs)):
        bitString = f.getConfig(n_up,nSites,configs['up'][j],B) 
        string1_u = bitString[0:site]                               
        string2_u = bitString[site:len(bitString)]
        cfig_L_up = f.getIndex(len(string1_u),string1_u,B)        
        cfig_R_up = f.getIndex(len(string2_u),string2_u,B)
        nUp_l = sp.sum(string1_u)
        bitString = f.getConfig(n_dn,nSites,configs['dn'][j],B) 
        string1_d = bitString[0:site]                           
        string2_d = bitString[site:len(bitString)]
        cfig_L_dn = f.getIndex(len(string1_u),string1_d,B)          
        cfig_R_dn = f.getIndex(len(string2_d),string2_d,B)
        nDn_l = sp.sum(string1_d)
        C_i[j]['nUp_left'] = nUp_l
        C_i[j]['nDn_left'] = nDn_l
        C_i[j]['left'] = cfig_L_up*B[(nUp_l,nSites+1)] + cfig_L_dn      
        C_i[j]['right'] = cfig_R_up*B[(n_up-nUp_l,nSites+1)] + cfig_R_dn
        C_i[j]['amplitude'] = state[j]
        C_i[j]['left_up'] = cfig_L_up
    C_i = sp.sort(C_i,order='left_up')
    impurity_hashes_up = sp.unique(C_i['left_up'])
    hash_left = {}
    hash_right = {}
    i=0
    for entry in C_i:
        hash_left[(entry['left'],entry['nUp_left'],entry['nDn_left'])] = i
        hash_right[(entry['right'],n_up - entry['nUp_left'],n_dn-entry['nDn_left'])] = i
        i=i+1
    uniques_left = hash_left.keys()
    uniques_right = hash_right.keys()
    j=0
    for cfg in uniques_left:
        hash_left[cfg] = j
        j=j+1
    j=0
    for cfg in uniques_right:
        hash_right[cfg] = j
        j=j+1
    impurity_state = []
    C_ij = sp.zeros((len(uniques_left),len(uniques_right)))
    for entry in C_i:
        index_left = hash_left[(entry['left'],entry['nUp_left'],entry['nDn_left'])]
        index_right =  hash_right[(entry['right'], n_up-entry['nUp_left'],n_dn-entry['nDn_left'])]
        C_ij[index_left][index_right] = entry['amplitude']
        impurity_state.append(entry['left_up'])
    if return_indices == False:
        return C_ij
    else:
        return C_ij,impurity_state


def getEntropy(cfgs,state,nSites,n_up,n_dn,cfigs_2 = 0.0,GS_2 = 0.0,return_schmidt=False):
    '''Note that right now, I can only do this for N_up = N_dn because I do not
       keep explicit track of the particle number for each configuration list. I can
       fairly easily modify to include this however'''
    S = sp.zeros(nSites)
    for site in range(0,nSites):
        C_ij = coefficient_matrix(state,cfgs,site,nSites,n_up,n_dn)
        
        schmidt_coeffs = sp.linalg.svdvals(C_ij)
        if return_schmidt != False and site==return_schmidt:
            sp.save('Schmidt_coeffs_bond={}'.format(return_schmidt),schmidt_coeffs)
        norm = sp.linalg.norm(schmidt_coeffs)
        schmidt_coeffs = schmidt_coeffs/norm
        schmidt_coeffs[schmidt_coeffs<10**-12]=0.0

        S[site] = 0.0
        for i in range(len(schmidt_coeffs)):
            if schmidt_coeffs[i] != 0.0:
                S[site] = S[site] - schmidt_coeffs[i]*sp.log(schmidt_coeffs[i])
    return S
