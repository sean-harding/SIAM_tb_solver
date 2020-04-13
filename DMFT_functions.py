#Contains functions for performing the DMFT self-consistency step. 
#Author: Sean M. Harding, (University of Bristol, 2018)

import scipy as sp
import scipy.integrate
import math

def getHyb(G_imp,frequencies,chemicalPotential,dispersion,DoS,hyb_initial,energy_axis):
    '''This function calculates the self-energy of the SIAM bath and applies DMFT
    self consistency to obtain a new Hybridization function for a non-Bethe lattice'''

    #Calculate self-energy as a function of frequency
    self_energy = frequencies + chemicalPotential - hyb_initial - 1.0/G_imp

    #We have to do an integral over energy (momentum) for each frequency to get the momentum averaged local G of the lattice
    G_lattice = sp.zeros(len(frequencies))
    for i in range(len(self_energy)):
        integrand = DoS/(frequencies[i] + chemicalPotential - energy_axis - frequencies[i])
        G_lattice[i] = sp.integrate.trapz(integrand,energy_axis)

    #Now apply DMFT self-consistency by using G_lattice to calculate a new hyb.function (the reverse of the first step in this function definition)
    hyb_new = frequencies + chemicalPotential - self_energy - 1.0/G_lattice

    return hyb_new