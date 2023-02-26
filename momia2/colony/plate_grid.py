from skimage import transform, filters, feature, measure, registration, io, color, morphology
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import ndimage as ndi
import glob
import pandas as pd
import pickle as pk


def find_houghline_intersec(theta1, dist1, theta2, dist2,):
    if theta1 == theta2:
        return None
    
    elif theta1 == 0:
        x = dist1
        y = (dist2/np.sin(theta2))-dist1/np.tan(theta2)
        return (x,y)    
    elif theta2 == 0:
        x = dist2
        y = (dist1/np.sin(theta1))-dist2/np.tan(theta1)
        return (x,y)    
    else:
        a1 = -1/np.tan(theta1)
        a2 = -1/np.tan(theta2)
        b1 = dist1/np.sin(theta1)
        b2 = dist2/np.sin(theta2)
        x = (b2-b1)/(a1-a2)
        y = a1*x+b1
        return (x,y)

class GridProjection:
    
    def __init__(self, file):
        self.image = color.rgb2gray(io.imread(file))
        self.hough_peak_list = []
        self.key_points = []
        self.mask = np.zeros(self.image.shape)
        self.horizontal_sorted = None
        self.vertical_sorted = None
        self.selected_horizontal_hough_id = []
        self.selected_vertical_hough_id = []
        
    def mask_frangi(self, sigmas = (2.5), threshold = 0.000001):
        self.mask = filters.frangi(min_max(self.image),sigmas=sigmas,black_ridges=False)>threshold
        
    def find_hough_lines(self,min_distance=30,threshold=0.35):
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h,t,d = hough_line(self. mask,theta = tested_angles)
        for _, ang, dist in zip(*hough_line_peaks(h,t,d,
                                                  min_distance=min_distance,
                                                  threshold=threshold*np.max(h))):
            self.hough_peak_list.append([ang,dist])   
        self.hough_peak_list = np.array(self.hough_peak_list)
        
    def sort_hough_lines(self,vertial_min_rotation=0.1, horizontal_max_rotation=1.52):
        angles = self.hough_peak_list[:,0]
        vertical = self.hough_peak_list[np.abs(angles)<vertial_min_rotation]
        horizontal = self.hough_peak_list[np.abs(angles)>horizontal_max_rotation]
        xh,yh=np.array([np.cos(horizontal[:,0]), np.sin(horizontal[:,0])])*horizontal[:,1][np.newaxis,:]
        xv,yv=np.array([np.cos(vertical[:,0]), np.sin(vertical[:,0])])*vertical[:,1][np.newaxis,:]
        self.horizontal_sorted = horizontal[np.argsort(yh)]
        self.vertical_sorted = vertical[np.argsort(xv)]     
        
    def plot_hough_lines(self,vertical='all',horizontal='all'):
        fig=plt.figure(figsize=(8,8))
        plt.imshow(self.mask)
        self.selected_horizontal_hough_id = filter_hough_id(horizontal, self.horizontal_sorted)
        self.selected_vertical_hough_id = filter_hough_id(vertical, self.vertical_sorted)
        for (angle, dist) in self.horizontal_sorted[self.selected_horizontal_hough_id]:
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            plt.axline((x0, y0), slope=np.tan(angle + np.pi/2),color='salmon')
        for (angle, dist) in self.vertical_sorted[self.selected_vertical_hough_id]:
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            plt.axline((x0, y0), slope=np.tan(angle + np.pi/2),color='salmon')
            
    def projection(self, 
                   reference_key_points, 
                   output_name,
                   output_shape=(2000,2000),
                   plot=True,
                   save=False):
        selected_h = self.horizontal_sorted[self.selected_horizontal_hough_id,:]
        selected_v = self.vertical_sorted[self.selected_vertical_hough_id,:]
        target_key_points = np.array([find_houghline_intersec(h[0],h[1],v[0],v[1]) for h in selected_h for v in selected_v]).astype(float)
        target_aspect = target_key_points[-1]-target_key_points[0]
        ref_aspect = reference_key_points[-1]-reference_key_points[0]
        
        rescaled_ref = (reference_key_points-reference_key_points[0][np.newaxis,:])*((target_aspect/ref_aspect)[np.newaxis,:])+reference_key_points[0][np.newaxis,:]
        tform = transform.PiecewiseAffineTransform()
        tform.estimate(rescaled_ref,target_key_points)
        out = transform.warp(self.image, tform, output_shape=np.array(output_shape)+500)
        x,y = np.where(out>0)
        cropped = out[x.min():x.max(),y.min():y.max()]
        resized=transform.resize(cropped,(2000,2000),anti_aliasing=True)
        transformed = (resized*255).astype(np.uint8)
        if save:
            tifffile.imsave(output_name,transformed,imagej=True)
        if plot:
            tifffile.imshow(transformed, cmap='gist_gray')
    
    def _adjust_hough_line(self,hough_id,dd=0,da=0,vertical = False):
        if vertical:
            self.vertical_sorted[hough_id] += np.array([da,dd])
        else:
            self.horizontal_sorted[hough_id] += np.array([da,dd])
            
    def _add_hough_line(self,d,a):
        self.hough_peak_list = np.vstack([self.hough_peak_list,np.array([[a,d]])])
        
def min_max(data):
    return (data-data.min())/(data.max()-data.min())

def filter_hough_id(param, hough_peaks):
    if isinstance(param,str):
        if param=='all':
            return np.arange(len(hough_peaks))
        else:
            raise ValueError('Illegal value!')
    elif isinstance(param,list) or isinstance(param,np.array):
        return np.array(param)