import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import geometric_transform

def topolar(img, order=2):

    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order. 
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)
        
        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]
        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]

        return i,j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)


def binedges(Nbins,range):
    edges = np.linspace(range[0],range[1],Nbins+1)
    return edges, edges

def bincentres(xedges,yedges):
    x_centers = np.mean(np.vstack([xedges[0:-1],yedges[1:]]), axis=0)
    y_centres = np.mean(np.vstack([xedges[0:-1],yedges[1:]]), axis=0)
    return x_centers, y_centres

def legendre(mu,l):
    """
    Legendre polynomial of order zero, one and two.
    :param mu:        N x N matrix representing value of cos(x) at each point
    """
    if l==0:
        polynomial = 1
    elif l==1:
        polynomial = mu
    elif l==2:
        polynomial = 0.5*(3*mu**2-1)
    else:
        print("WRONG l VALUE")
    return polynomial

def xi(mu,l,data):
    integrand = (0.5*(2*l+1)*legendre(mu,l)*data)

    return integrand

def polarfilter(Nbins,radius,l,data):

    """get input grid"""
    x_edges, y_edges = binedges(Nbins,[-radius,radius])
    x,y = bincentres(x_edges,y_edges)
    xs,ys = np.meshgrid(x,y)
    dx = 2*radius/(Nbins)

    """calculate variables"""
    s = np.sqrt(xs**2+ys**2)
    cos = xs/s
    sin = ys/s
    
    integrand = -np.nan_to_num(xi(cos,l,data))*sin

    pol, (rads,angs) = topolar(integrand)
    pol = (pol - np.flip(pol,axis = 1))
    pol[:,int(len(angs)/2):] = 0

    dang = np.pi/len(angs)
    integral = np.sum(pol,axis=1)*dang
    
    return rads*dx,integral