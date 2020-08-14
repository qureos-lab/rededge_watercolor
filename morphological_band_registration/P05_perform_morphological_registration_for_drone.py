# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:28:05 2020

@author: wonk1
"""

import glob, os
import micasense.capture as capture

import numpy as np
import math
#from pathlib import Path  
import library_morph as morph
import cv2


##  01 Process Rededge Data ======================================================
if __name__ == '__main__':
    dir_tif = ".\\data\\"
    dir_cat = ".\\results\\catalog\\"
    dir_res = ".\\results\\morph\\"
    
    #=========================================================================
    # Control Parameters (HARD CODE HERE)
    #=========================================================================
    # blur factor of the original radiance data (wl_blur=1 means using original resolution)
    ws_blur = 1
    # half-window size for the morphological process
    # this number must be an odd number (1, 3, 5, etc.)
    # the full window size will be then (ws_morph*2+1)-by-(ws_morph*2+1)
    ws_morph = 15
    id_pan = 0  # image ID for panel image
    id_sky = 7   # image ID for sky image

    id_water1 = 130   # starting ID for water image 
    id_water2 = 132   # ending ID for water image
    id_step = 1
    
    display = 0  # make this 1 if you want to produce byproducts during the morphological registration
                   # (namely, the slope and y-intercept images)
                   
    #=========================================================================
    # min/max range for byproduct images (slope and y-intercept images)
    #=========================================================================
    # min/max values for the Slope image (for 5 bands)
    vmma = [ [0.5, 1.5], [0.7, 1.7], [0.7, 1.7], [0.7, 1.7], [0.7, 1.7]]
    # min/max values for the y-intercept image (for 5 bands)
    vmmb = [ [-12, 5.], [-15, 0.], [-20., 0.], [-20., 0.], [-20., 0.]]
    # To produce by product images, set disply=1 in the morphological registration function
    
    # Reflectance of RedEdge Gray Panel 
    ref_pan = np.array([52.5, 52.6, 52.5, 52.5, 52.3])*0.01
    
    
    
    
    
    #=========================================================================
    # Process reflectance panel
    #=========================================================================
    imageNames = glob.glob(os.path.join(dir_tif,'**/IMG_{:04d}_*.tif'.format(id_pan)), recursive=True)
    cap_pan = capture.Capture.from_filelist(imageNames)
    rad_pan = cap_pan.panel_radiance()
    irr_pan = rad_pan/ref_pan*math.pi    
    tmp = irr_pan[3]
    irr_pan[3] = irr_pan[4]
    irr_pan[4] = tmp
    with open(dir_res + "panel_irr.txt", "w") as text_file:
        text_file.write("{:f}, {:f}, {:f}, {:f}, {:f}".format(irr_pan[0], irr_pan[1],irr_pan[2],irr_pan[3],irr_pan[4])) 
        
    #=========================================================================
    # Process sky image
    #=========================================================================
    imageNames = glob.glob(os.path.join(dir_tif,'**/IMG_{:04d}_*.tif'.format(id_sky)), recursive=True)
    cap_sky = capture.Capture.from_filelist(imageNames)
    cap_sky.create_aligned_capture(img_type = 'radiance')
    im_i = cap_sky._Capture__aligned_capture
    im_aligned = morph.swap_4_and_5(im_i)
    img_ref_sky = im_aligned/irr_pan
    tar_type = 's'
    scale_factor = 2.
    morph.write_img_to_tiff(img_ref_sky, dir_res+'SKY' +'.'+tar_type + '.ref', scale_factor)
    
    #=========================================================================
    # Process (serial) water images
    #=========================================================================
 
    for id_i in range(id_water1, id_water2, id_step):
        imageNames = glob.glob(os.path.join(dir_tif,'**/IMG_{:04d}_*.tif'.format(id_i)), recursive=True)
        if len(imageNames) == 0:
            continue
        print(imageNames[0])
        cap_wat = capture.Capture.from_filelist(imageNames)
        str_timestamp, str_info, fname_tif, fname_i, irr_dls = morph.get_capture_save_filename(cap_wat)


        cap_wat.create_aligned_capture(img_type = 'radiance')
        cap_wat.save_capture_as_rgb(dir_cat + fname_i+'_RGB.png')
        
        
        im_i = cap_wat._Capture__aligned_capture
        im_aligned = morph.swap_4_and_5(im_i)
        
        img_ref_wat = im_aligned/irr_pan

        tar_type = 'w'
        scale_factor = 2.
        morph.write_img_to_tiff(img_ref_wat, dir_res+fname_tif+'.'+tar_type + '.ref', scale_factor)
        
        # Perform blur
        img_ref_wat_blur = img_ref_wat.copy()
        if ws_blur != 1:
            for i in range(5):
                img_band_blur = cv2.blur(img_ref_wat[:,:,i],(ws_blur,ws_blur))
                img_ref_wat_blur[:,:,i]= img_band_blur
        
        # Find non-water pixel (choose either of the two lines below)
        idx_non = morph.find_non_water(img_ref_wat_blur)
        idx_non = [[0], [0]]  # for non-water scene    
     
        # Perform morphological registration
        tar_type = 'w'
        scale_factor = 10.
        img_ref_wat_cor = morph.perform_morphological_registration(img_ref_wat_blur, ws_morph, idx_non, vmma, vmmb, dir_res, dir_res, display=display)
        morph.write_img_to_tiff(img_ref_wat_cor, dir_res+fname_tif+'.'+tar_type + '.ref.blur{:0d}'.format(ws_blur)+ '_morphed', scale_factor)


        with open(dir_res + "panel_irr.txt", "w") as text_file:
            text_file.write("{:f}, {:f}, {:f}, {:f}, {:f}".format(irr_pan[0], irr_pan[1],irr_pan[2],irr_pan[3],irr_pan[4])) 
        