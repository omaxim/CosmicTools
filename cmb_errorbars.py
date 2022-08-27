from bdb import Breakpoint
import numpy as np
import healpy as hp
from rasterise_map import get_meshgrid, generate_CMB_cutout
from dipole_filtering import polarfilter

def get_errorbars(order, Nhalos,radius,Nbins,cosmo,NSIDE,z_shift,map):
    x,y = get_meshgrid(radius, Nbins)
    arrays = []
    r=[]    
    for j in range(order):
        print(j)
        cutout = generate_CMB_cutout(cosmo,z_shift,Nbins,(x,y),map,NSIDE)

        
        for i in range(Nhalos):
            cutout += generate_CMB_cutout(cosmo,z_shift,Nbins,(x,y),map,NSIDE)


        output = cutout/Nhalos

        plt.pcolormesh(output)
        plt.title("NSIDE = " + str(NSIDE) + " average of " + str(Nhalos))
        plt.show()

        dipole = polarfilter(Nbins,radius,l=1,data=output)
        arrays.append(dipole[1])
        if j==0:
            r.append(dipole[0])
        else:
            continue
    
    testarray = np.array([arrays])
    stdev = np.std(testarray,axis=1)
    
    return stdev.T, np.array(r)[0]


import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

def main():


    cosmo = Planck18
    NSIDE = 2048*2
    map = hp.read_map("NSIDE_"+str(NSIDE)+".fits")

    Nhalos = 10

    Nbins = 100
    z_shift = 0.5
    radius = 5

    stdev1, r1 = get_errorbars(100, 10,radius,Nbins,cosmo,NSIDE,z_shift,map)
    #stdev2, r2 = get_errorbars(100, 100,radius,Nbins,cosmo,NSIDE,z_shift,map)
    #stdev3, r3 = get_errorbars(100, 1000,radius,Nbins,cosmo,NSIDE,z_shift,map)
    #stdev4, r4 = get_errorbars(100, 10000,radius,Nbins,cosmo,NSIDE,z_shift,map)
    np.save("100_1000_stdev.npy",stdev1)
    np.save("100_1000_r.npy",r1)
        
    plt.title("Temperature standard deviation in 100 stacks of N halos")
    #plt.plot(r1,np.log10(stdev1),label="N = 10 halos",linewidth=2)
    #plt.plot(r2,np.log10(stdev2),label="N = 100 halos",linewidth=2)
    #plt.plot(r3,np.log10(stdev3),label="N = 1000 halos",linewidth=2)
    #plt.plot(r4,np.log10(stdev4),label="N = 10000 halos",linewidth=2)
    plt.xlabel("r [cMpc/h]", fontsize=15)
    plt.ylabel(r"$log(T) [K]$", fontsize=15)

    plt.legend()


    plt.xlim((0,radius))

    plt.tight_layout()

    plt.show()

main()