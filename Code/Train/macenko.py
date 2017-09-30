'''
This file contains my implementation of the method to determine 
Stain Vector using SVD as outlined in the following paper:-

Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 
Biomedical Imaging: From Nano to Macro, 2009.
ISBI'09. IEEE International Symposium on. IEEE, 2009.
'''

import numpy as np
import math 
import scipy.sparse.linalg
from sklearn.preprocessing import normalize

def rgb2od(img):
    img[np.where(img == 0)] = np.min(img[np.where(img > 0)])
    return -np.log(img)

def od2rgb(img):
    return np.exp(-img)

def rodrigues_rot_matrix(v,k,theta):
    '''
    	This function implements rotation of vectors in v,
    	along the directions specified in k, by an angle theta
    	
        v,k are arrays like [[1,2,3]]
        theta is an array like [[1,2,3,....]]
    '''
        
    v = v.astype('float32')
    k = k.astype('float32')
    theta = theta.astype('float32')    
            
    k = k / math.sqrt(k[0,0]*k[0,0] + k[0,1]*k[0,1] + k[0,2]*k[0,2]) #normalize the normal vector
    
    crosskv = np.zeros(v[0,:].shape) # initialize cross of k and v with the correct dimension 
    crosskv[0] = k[0,1]*v[0,2] - k[0,2]*v[0,1]
    crosskv[1] = k[0,2]*v[0,0] - k[0,0]*v[0,2]
    crosskv[2] = k[0,0]*v[0,1] - k[0,1]*v[0,0] 
    theta_t = np.transpose(theta)
    
    return np.transpose(np.cos(theta_t)*np.transpose(v[0,:])+ \
                        np.sin(theta_t)*np.transpose(crosskv) + \
                        np.dot((1-np.cos(theta_t)),k*np.dot(k,v[0,:])))

def GetOrthogonalBasis(image_rgb):
    w,h,d =  image_rgb.shape
    
    img_od = np.transpose(np.reshape(rgb2od(image_rgb),[w*h,3]))
        
    # remove pixels below threshold
    img_od_norm = np.sqrt(np.sum(img_od**2,0))
    good = img_od_norm > 0.15
    img_od_good = img_od[:,good]
        
    phi_ortho = scipy.linalg.svd(img_od_good,full_matrices=False)[0]
    # reverse any axis having more than one negative value
    mask = phi_ortho<0
    cols = np.where(np.sum(mask,0)>1,1,0)
    phi_ortho = np.transpose([-phi_ortho[:,j] if cols[j] else phi_ortho[:,j] for j in range(len(cols))])
    a_ortho = np.dot(scipy.linalg.pinv(phi_ortho),img_od)
    
    return phi_ortho, a_ortho

def GetWedgeMacenko(image_rgb, squeeze_percentile):
    phi_ortho, a_ortho = GetOrthogonalBasis(image_rgb)
    
    image_od = np.dot(phi_ortho,a_ortho)
    phi_wedge = phi_ortho
    phi_plane = phi_ortho[:,[0,1]]
    a_plane = a_ortho[[0,1],:]    
    
    # normalize a_plane
    a_plane_normalized = normalize(a_plane,axis=0,norm='l2')
     
    #find robust extreme angles
    min_a = np.percentile(np.arctan(a_plane_normalized[1,:]/a_plane_normalized[0,:]),squeeze_percentile)
    max_a = np.percentile(np.arctan(a_plane_normalized[1,:]/a_plane_normalized[0,:]),100-squeeze_percentile)
    
    
    phi_1 = rodrigues_rot_matrix(np.array([phi_plane[:,0]]), \
                                 np.array([np.cross(phi_plane[:,0],phi_plane[:,1])]),np.array([min_a]))
    phi_2 = rodrigues_rot_matrix(np.array([phi_plane[:,0]]), \
                                 np.array([np.cross(phi_plane[:,0],phi_plane[:,1])]),np.array([max_a]))
    
    phi_wedge[:,[0,1]] = np.transpose([phi_1,phi_2])
    a_wedge = np.dot(scipy.linalg.pinv(phi_ortho), image_od)
    
    return phi_wedge,a_wedge
