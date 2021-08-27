import scipy as sp
import helperFunctions as f

'This module takes a ground state from the main module and examines the type of states it contains'

sim_in = sp.load('sim_out.npz')
nSites,cfigs,N_up,imp,ref,amps = int(sim_in['nSites']),sim_in['cfigs'],int(sim_in['N_up']),int(sim_in['imp']),sim_in['ref_state'], sim_in['amps'][0]
binomials = f.binomial(nSites+1)                                                        
ref_up = f.getConfig(N_up,nSites,ref['up'],binomials)   
ref_dn = f.getConfig(N_up,nSites,ref['up'],binomials)
lookup = {hsh:i for i,hsh in enumerate(cfigs['idx'])}

sector = set()
print('Order: up_p,up_h,dn_p,dn_h')
'c1 through c4 store the indices where excitations are located. Sector is the number of excitations in each chain'
st = sp.recarray(len(cfigs),dtype=[('c1',object),('c2',object),('c3',object),('c4',object),('sector','4int')])
for i,c in enumerate(cfigs):
    cup_d = f.getConfig(N_up,nSites,c['up'],binomials)
    cdn_d = f.getConfig(N_up,nSites,c['dn'],binomials)
    swaps_up = sp.nonzero([a!=b for a,b in zip(ref_up,f.getConfig(N_up,nSites,c['up'],binomials))])[0]  #Indices of up spin swaps
    swaps_dn = sp.nonzero([a!=b for a,b in zip(ref_dn,f.getConfig(N_up,nSites,c['dn'],binomials))])[0]  #Indices of down spin swaps
    ind_up_p = swaps_up[swaps_up<imp]   #24,25 are impurity and partially occupied
    ind_up_h = swaps_up[swaps_up>imp+1]
    ind_dn_p = swaps_dn[swaps_dn<imp]
    ind_dn_h = swaps_dn[swaps_dn>imp+1]
    #Particle number sectors for each chain
    sec = (len(ind_up_p), len(ind_up_h), len(ind_dn_p), len(ind_dn_h))
    st[i] = ind_up_p.tolist(),ind_up_h.tolist(),ind_dn_p.tolist(),ind_dn_h.tolist(),sec     #Have made these lists for ease of display
'First thing to look at is the particle number sectors involved'
maxE = 0
for c in st:
    sector.add(tuple(c['sector']))
for s in sector:
    counts = sum(sp.all(c['sector']==s) for c in st)
    if sum(s)>maxE:
        maxE = sum(s)
    print('Sector:{}, counts:{}'.format(s,counts))
print("\nMaximum excitation number: {}".format(maxE))
slices = [[i for i,a in enumerate(st) if sp.all(a['sector']==s)] for s in sector]
sector = {s:i for i,s in enumerate(sector)}

#There should be (nSites-2)/2 empty orbitals, followed by 2 of variable occupancy, followed by (nSites-2)/2 filled orbitals
template = [0]*int((nSites-2)/2)+[0,0]+[1]*int((nSites-2)/2)    #Template for states with 1 particle in a given chain
dense_occs = [[0,0],[1,0],[0,1],[1,1]]
n=0
m=0
for s in slices[sector[(1,0,1,0)]]:
    n+=1
    up_particle = st[s]['c1']
    down_particle = st[s]['c3']
    new_up = sp.copy(template)
    new_dn = sp.copy(template)
    new_up[up_particle]=1
    new_dn[down_particle]=1
    #print(new_up)
    #print(new_dn)
    #U[-1]*binomials.get((n_up,N+1),0)+D[-1])
    idx_a = f.getIndex(nSites,new_dn,binomials) + cfigs[s]['up']*binomials.get((N_up,nSites+1))
    idx_b = cfigs[s]['dn'] + f.getIndex(nSites,new_up,binomials)*binomials.get((N_up,nSites+1))
    try:
        i = lookup[idx_a]
        j = lookup[idx_b]
        m+=1
    except:
        pass
    print("C2 = {},T2 = {}".format(amps[s],amps[s]-0.5*amps[i]*amps[j]))
print("{} out of {} terms found".format(m,n))