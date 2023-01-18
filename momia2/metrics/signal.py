import numpy as np
from skimage import filters
import scipy.ndimage as ndi
from ..utils.linalg import *



def orthogonal_mesh(midline,contour,width_list,
                    length=None,
                    median_width=None,scale=2):
    """
    
    """
    if type(length) not in [float,int]:
        length = measure_length(midline)
    if type(median_width) not in [float,int]:
        median_width = np.median(width_list)
    
    n_length = int(len(midline)*scale)
    n_width = int(len(midline)*median_width*scale/length)
    
    edge_points = []
    subpixel_midline = spline_approximation(midline,n=n_length,
                                            smooth_factor=1,closed=False)
    unit_perp=unit_perpendicular_vector(subpixel_midline)
    
    for i in range(1,len(subpixel_midline)-1):
        # make core mesh
        p= line_contour_intersection(subpixel_midline[i],
                                    subpixel_midline[i]+unit_perp[i],contour)
        flip = point_line_orientation(subpixel_midline[i],subpixel_midline[i+1],p[0])
        if flip==-1:
            p=np.flip(p,axis=0)
        edge_points.append(p.flatten())
    
    edge_points=np.array(edge_points)
    
    x,y = edge_points[:,0],edge_points[:,1]
    dxy = (edge_points[:,2:]-edge_points[:,:2])/n_width
    mat = np.tile(np.arange(n_width+1),(len(edge_points),1))
    mat_x = x[:,np.newaxis]+mat*dxy[:,0][:,np.newaxis]
    mat_y = y[:,np.newaxis]+mat*dxy[:,1][:,np.newaxis]
    
    return np.array([mat_x,mat_y])