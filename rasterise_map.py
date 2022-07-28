import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter


"""
This code projects out and plots the portion of the CMB
covered by a lens of a given radius and distance from origin.
Smoothing is applied, assuming that the fluctuations are gaussian

Maxim Oweyssi 2022
"""

def random_three_vector():
    """
    Find a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    Code by Andrew from https://andrewbolster.info/2014/04/generating-a-unit-3-vector-in-python-uniform-spherical-projection.html
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return (x,y,z)

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1:        A 3d "source" vector
    :param vec2:        A 3d "destination" vector
    :return mat:        A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    Code by Peter from https://stackoverflow.com/a/59204638
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def get_meshgrid(radius, N):
    """
    Find centres of bins of a N x N 2D histogram as a numpy.meshgrid() tuple
    :param radius:      Proper size of the lens galaxy in Mpc
    :param N:           Number of bins in an N x N histogram
    :return meshgrid:   numpy.meshgrid() 2D tuple of centres of bins of an N x N histogram
    """
    x_edges = np.linspace(-radius,radius,N+1)
    y_edges = np.linspace(-radius,radius,N+1)
    x_centers = np.mean(np.vstack([x_edges[0:-1],x_edges[1:]]), axis=0)
    y_centres = np.mean(np.vstack([y_edges[0:-1],y_edges[1:]]), axis=0)
    x, y = np.meshgrid(x_centers,y_centres)

    return x,y

def generate_CMB_cutout(cosmo,z_shift,N,meshgrid,map,NSIDE,sigma):
    """
    Find and cut out the CMB background around a galaxy of a given proper size
    :param cosmo:       astropy.cosmology object
    :param z_shift:     Redsift of the lens galaxy
    :param N:           Number of bins in an N x N histogram
    :param meshgrid:    numpy.meshgrid() 2D tuple of centres of bins of an N x N histogram
    :param map:         HealPix map of the CMB
    :param NSIDE:       HealPix resolution of the sky map
    :param sigma:       Gaussian smoothing parameter
    :return cutout:     Gausian smoothed CMB cutout of the correct apparent size
    """
    #Create a vector for each bin around the equator, project into 1D
    equator = [0,0,1]
    distance = cosmo.comoving_distance(z_shift)
    x , y = meshgrid
    z = np.zeros((N,N)) + distance.value
    vectors = np.array([x.ravel(),y.ravel(),z.ravel()])

    #Rotate to align with a random direction
    randvec = random_three_vector()
    rotmat = rotation_matrix_from_vectors(equator,randvec)
    rotated_vectors = hp.rotator.rotateVector(rotmat,vectors)

    #Find the value at each pixel, reshape back to 2D and apply gaussian smoothing
    pixels = hp.vec2pix(NSIDE,rotated_vectors[0],rotated_vectors[1],rotated_vectors[2])
    values = map[pixels]
    smoothed_cutout = gaussian_filter(values.reshape((N,N)),sigma=sigma)

    return smoothed_cutout


import matplotlib.pyplot as plt
from astropy.cosmology import Planck18

def main():
    cosmo = Planck18
    map = hp.read_map("COM_CMB_IQU-commander-field-Int_2048_R2.01_year-2.fits")
    NSIDE = 2048
    Nbins = 200
    z_shift = 1
    radius = 5
    sigma = 11

    x,y = get_meshgrid(radius, Nbins)
    cutout = generate_CMB_cutout(cosmo,z_shift,Nbins,(x,y),map,NSIDE,sigma)

    plt.pcolormesh(x,y,cutout)
    plt.colorbar()
    plt.show()

main()