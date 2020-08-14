# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:44:59 2020

@author: wonk1
"""
import math
import numpy as np
import micasense.imageutils as imageutils
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
import time
from sklearn import datasets, linear_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path  


def imshow_scaled(img_rgb, minmaxvals):
#    minmaxvals = np.zeros(3, 2)
    img_scaled = img_rgb.copy()
    for i in range(3):
#        max_i = np.max(img_rgb[:,:,i])
#        min_i = np.min(img_rgb[:,:,i])
#        range_i = (max_i - min_i)*0.5
#        minmaxvals[i]=max_i
        min_i = minmaxvals[i, 0]
        max_i = minmaxvals[i, 1]
        range_i = max_i - min_i
        img_scaled[:,:,i] = (img_rgb[:,:,i] - min_i)/range_i
    
    ax = plt.imshow(img_scaled)
    return ax, img_scaled



def read_tif(filename, scale_factor, nband):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    n_rows, n_cols = arr.shape
    data = np.zeros([n_rows, n_cols, 5])
    for i in range(0,nband):
        band = ds.GetRasterBand(i+1)
        arr = band.ReadAsArray()
        data[:,:,i] = arr
    data = data/scale_factor/65535.
    return data

def write_img_to_tiff(im_aligned, filename, scale_factor):
    rows, cols, bands = im_aligned.shape
    driver = gdal.GetDriverByName('GTiff')
#    filename = "IMG_" + set_i + "_rad_"+var_ids[0] #blue,green,red,nir,redEdge
    outRaster = driver.Create(filename+".tiff", cols, rows, im_aligned.shape[2], gdal.GDT_UInt16)
   
    for i in range(0,5):
        print(i)
        outband = outRaster.GetRasterBand(i+1)
        outdata = im_aligned[:,:,i]*scale_factor*65535.
        outdata[outdata<0] = 0.
        outdata[outdata>65535] = 65535.
        outband.WriteArray(outdata)
        outband.FlushCache()
    outRaster = None
    
def create_catalog_image(im_aligned, dir_catalog, str_timestamp, str_info, fname_i):
    im_rgb = im_aligned[:,:, [2, 1, 0]]
    fig = plt.figure()
    minmaxvals = np.array([ [500., 3000.], [100., 5000.], [50., 4500.]])/5. /65535.
    ax = imshow_scaled(im_rgb, minmaxvals)
    
    plt.title(str_timestamp, fontsize=13)
    plt.text(-75, -85, str_info ,fontsize=13)
    plt.savefig(dir_catalog + fname_i + '.png')
    plt.close()
    
    
def align_capture(capture, warp_mode, img_type):
    ## Alignment settings
    match_index = 1 # Index of the band 
    max_alignment_iterations = 50
#    warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
#    warp_mode = cv2.MOTION_TRANSLATION
    pyramid_levels = 0 # for images with RigRelatives, setting this to 0 or 1 may improve alignment
    epsilon_threshold=1e-10
    print("Alinging images. Depending on settings this can take from a few seconds to many minutes")
    # Can potentially increase max_iterations for better results, but longer runtimes
    warp_matrices, alignment_pairs = imageutils.align_capture(capture,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
#                                                              multithreaded=False, 
#                                                              debug = True,
                                                              epsilon_threshold = epsilon_threshold, 
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)
    
    print("Finished Aligning, warp matrices={}".format(warp_matrices))
    
    
    cropped_dimensions, edges = imageutils.find_crop_bounds(capture, warp_matrices, warp_mode=warp_mode)
    im_aligned = imageutils.aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type=img_type)
    
    return im_aligned, warp_matrices


def swap_4_and_5(im_test):
    im_4 = im_test[:,:,3].copy()
    im_5 = im_test[:,:,4].copy()
    im_new = im_test.copy()
    im_new[:,:,3] = im_5
    im_new[:,:,4] = im_4
    return im_new   

def read_tif(filename, scale_factor, nband):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    n_rows, n_cols = arr.shape
    data = np.zeros([n_rows, n_cols, 5])
    for i in range(0,nband):
        band = ds.GetRasterBand(i+1)
        arr = band.ReadAsArray()
        data[:,:,i] = arr
    data = data/scale_factor/65535.
    return data

def get_capture_save_filename(cap_i):
     # image names
    
    tmp = cap_i.images[0].path
    tmp = Path(tmp).name
    id_i = int(tmp.split("_")[1])
    
    ymdhns = cap_i.utc_time()
    
    yy_i = ymdhns.year
    oo_i = ymdhns.month
    hh_i = ymdhns.hour + 9
    dd_i = ymdhns.day
    mm_i = ymdhns.minute
    ss_i = ymdhns.second
    
    lla = cap_i.location()
    lat = lla[0]
    lon = lla[1]
    alt = lla[2]
    
    lat_d = int(np.floor(lat))
    lat_m = (lat - lat_d)*60.
    lon_d = int(np.floor(lon))
    lon_m = (lon - lon_d)*60.
    
    
    
    irr_dls = cap_i.dls_irradiance()
    tmp = irr_dls[3]
    if len(irr_dls) > 4:
        irr_dls[3] = irr_dls[4]
        irr_dls[4] = tmp
    kpw = cap_i.dls_pose()
    ww = kpw[2]
    pp = kpw[1]
    kk = kpw[0]
    if hh_i > 23:
        hh_i -= 24
        dd_i += 1
    dg = u'\N{DEGREE SIGN}'
    str_timestamp = "{:4d}/{:02d}/{:02d}  {:02d}:{:02d}:{:02d} [{:04d}]".format(yy_i, oo_i, dd_i, hh_i, mm_i, ss_i, int(id_i))   
    str_info = " Alt={:4.1f} m,  Roll={:3.1f}{:s},  Pitch={:3.1f}{:s},  Yaw={:3.1f}{:s}".format(alt, math.degrees(ww), dg, math.degrees(pp), dg, math.degrees(kk), dg)
    fname_i = "{:4d}{:02d}{:02d}_{:02d}{:02d}{:02d}_{:02d}_{:4.2f}_{:02d}_{:4.2f}_[{:04d}]".format(yy_i, oo_i, dd_i, hh_i, mm_i, ss_i, lat_d, lat_m, lon_d, lon_m, int(id_i)) 
    fname_tif = "{:4d}{:02d}{:02d}_{:02d}{:02d}_{:02d}_{:4.2f}_{:02d}_{:4.2f}_{:04d}".format(yy_i, oo_i, dd_i, hh_i, mm_i, lat_d, lat_m, lon_d, lon_m, int(id_i)) 
    
    return str_timestamp, str_info, fname_tif, fname_i, irr_dls
    
def find_non_water(img_ref_wat):
    idx1 = np.where( img_ref_wat[:,:,0] > 0.07)
    idx2 = np.where( img_ref_wat[:,:,1] > 0.06)
    idx3 = np.where( img_ref_wat[:,:,2] > 0.035)
    idx4 = np.where( img_ref_wat[:,:,3] > 0.035)
    idx5 = np.where( img_ref_wat[:,:,4] > 0.035)
    
    # for removing white cap
    idx1 = np.where( img_ref_wat[:,:,0] > 0.0534)
    idx2 = np.where( img_ref_wat[:,:,1] > 0.0458)
    idx3 = np.where( img_ref_wat[:,:,2] > 0.0115)
    idx4 = np.where( img_ref_wat[:,:,3] > 0.0076)
    idx5 = np.where( img_ref_wat[:,:,4] > 0.0076)
    
    
    dims = img_ref_wat.shape
    
    idxs = [idx1, idx2, idx3, idx4, idx5]
    idxas = []
    for i in range(5):
        idxia = idxs[i][1] + idxs[i][0]*dims[1]
        idxas.append(idxia)
    
    idx_tmp = np.intersect1d(idxas[2], idxas[3])
    idx_nir = np.intersect1d(idx_tmp, idxas[4])
    idx_com = np.union1d(idxas[1], idx_nir)
    idx_non = np.union1d(idxas[0], idx_com)
    
    idx_oth = np.where(img_ref_wat[:,:,1] < img_ref_wat[:,:,4]*1.5 )
    idx_otha = idx_oth[1] + idx_oth[0]*dims[1]
    idx_non = np.union1d(idx_otha, idx_non)
    
    idx_col = np.mod(idx_non, dims[1])
    idx_row = ((idx_non-idx_col)/dims[1]).astype(np.int64)
    
    idx = [idx_row, idx_col]
    return idx

def perform_morphological_registration(img_ref_wat_blur, ws_morph, idx_non, vmma, vmmb, dir_results, dir_archive, display=1):
    n_rows, n_cols, n_bands = img_ref_wat_blur.shape
    
    # set a morphological window

    ws_half = int(np.floor(ws_morph/2))
    rr_min = ws_half
    cc_min = ws_half
    rr_max = n_rows-ws_half-1
    cc_max = n_cols-ws_half-1
    conv_step = int(ws_half)
    
    # set Gaussian weights
    xx, yy = np.meshgrid(np.linspace(-ws_half,ws_half,ws_morph), np.linspace(-ws_half,ws_half,ws_morph))
    dd = np.sqrt(xx*xx+yy*yy)
    sigma, mu = int(np.floor(ws_half/2.)), 0.0
    gg = np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 ) ) )
    
    img_a_box = np.zeros((n_rows, n_cols, 5))     # box for slope (a)
    img_b_box = np.zeros((n_rows, n_cols, 5))     # box for y-intercept (b)
    time_box = np.zeros(5)                          # box for elapsed time
        
    
    
    for i in [0, 2, 3, 4]:
        t0= time.clock()
        img_i = img_ref_wat_blur[:,:,i]
        img_ref = img_ref_wat_blur[:,:,1]
        img_i[idx_non[0][:], idx_non[1][:]] = np.nan
        img_ref[idx_non[0][:], idx_non[1][:]] = np.nan    
        
        img_aa = np.zeros((n_rows, n_cols)) 
        img_bb = np.zeros((n_rows, n_cols)) 
        img_weight = np.zeros((n_rows, n_cols))
        
        count = 0
        for rr in range(rr_min, rr_max, conv_step):
            print("{:3d}% done".format(int(np.floor(rr/(rr_max-rr_min)*100.))))
            for cc in range(cc_min, cc_max, conv_step):
                
                row_min = int(rr-ws_half)
                row_max = int(rr+ws_half+1)
                col_min = int(cc-ws_half)
                col_max = int(cc+ws_half+1)
                rc_subset = [ [row_min, row_max], [col_min, col_max]]
                img_sub_i = img_i[ rc_subset[0][0]:rc_subset[0][1], rc_subset[1][0]:rc_subset[1][1]]
                img_sub_ref = img_ref[ rc_subset[0][0]:rc_subset[0][1], rc_subset[1][0]:rc_subset[1][1]]
                v_i = np.sort(img_sub_i.ravel())
                v_ref = np.sort(img_sub_ref.ravel())
                
                # perform regression
                regr = linear_model.LinearRegression()
                ind_valid_ref = ~np.isnan(v_ref)
                ind_valid_i = ~np.isnan(v_i)
                ind_valid = ind_valid_ref & ind_valid_i
                
                if len(v_ref[ind_valid]) < ws_morph*ws_morph/2.:
                    continue
                if len(v_i[ind_valid]) < ws_morph*ws_morph/2.:
                    continue
                
                Xtrain = v_ref[ind_valid, np.newaxis]
                Ytrain = v_i[ind_valid]
                regr.fit(Xtrain, Ytrain)
                aa = regr.coef_
                aa = aa[0]
                bb = regr.intercept_
                
                img_sub_aa = np.zeros((ws_morph, ws_morph)) + aa
                img_sub_bb = np.zeros((ws_morph, ws_morph)) + bb
                
                img_aa[rc_subset[0][0]:rc_subset[0][1], rc_subset[1][0]:rc_subset[1][1]] += img_sub_aa*gg
                img_bb[rc_subset[0][0]:rc_subset[0][1], rc_subset[1][0]:rc_subset[1][1]] += img_sub_bb*gg
                
                img_weight[rc_subset[0][0]:rc_subset[0][1], rc_subset[1][0]:rc_subset[1][1]] += gg  
                count += 1                 
                       
    
        img_aa[idx_non[0][:], idx_non[1][:]] = np.nan
        img_bb[idx_non[0][:], idx_non[1][:]] = np.nan
        
        tf_zero = img_weight == 0
        img_weight[tf_zero] = np.nan
        img_a_box[:,:,i] = img_aa/img_weight
        img_b_box[:,:,i] = img_bb/img_weight
        
        t1 = time.clock() - t0
        time_box[i] = t1
        
        if display:
            fig, ax = plt.subplots(1, figsize=(13,8))
            im = ax.imshow(img_aa/img_weight, vmin=vmma[i][0], vmax=vmma[i][1], cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05) 
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label('$a$ (slope)', fontsize=16)
            plt.savefig(dir_results + "Fig41_a_image_Band{:d}_ws25.png".format(i+1))
            plt.close()
                    
            
            fig, ax = plt.subplots(1, figsize=(13,8))
            im = ax.imshow(img_bb/img_weight*1e3, vmin=vmmb[i][0], vmax=vmmb[i][1], cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05) 
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label('$b$ (y-intercept x 10$^3$)', fontsize=16)
            plt.savefig(dir_results + "Fig42_b_image_Band{:d}_ws25.png".format(i+1))
            plt.close()
            
    #with open(dir_results + "img_a_box_and_img_b_box.pkl", 'wb') as f:
    #    pickle.dump([img_a_box, img_b_box, time_box], f)
    
    #np.savetxt(dir_results+"Tab31_elapsed_time_in_second_for_5_bands.csv", time_box, delimiter=",")
    
    #====================================================================
    # Re-construct Water Reflectance images
    #====================================================================
#    img_ref_wat_blur
    img_ref = img_ref_wat_blur[:,:,1]
    img_ref_wat_cor = img_ref_wat_blur.copy()
    for i in [0, 2, 3, 4]:
        img_a = img_a_box[:,:,i]
        img_b = img_b_box[:,:,i]
        img_i = img_ref*img_a + img_b
        img_ref_wat_cor[:,:,i] = img_i

        if display:        
            img_rgb = img_ref_wat_cor[:,:,[2, 1, 0]]
            minmaxvals = np.array([[0., 0.01], [0., 0.01], [0., 0.01]])
            minmaxvals = np.array([ [500., 3000.], [100., 5000.], [50., 4500.]])/5. /65535.*2.5
            minmaxvals = np.array([ [700., 3500.], [120., 6000.], [80., 4500.]])/5. /65535.*2.5
            minmaxvals = np.array([ [700., 4000.], [120., 5000.], [80., 4000.]])/5. /65535.*2.5
            plt.figure(figsize=(12, 10))
            imshow_scaled(img_rgb, minmaxvals)
            plt.savefig(dir_results + "Fig43_ref_wat.png")
            plt.close()
    
    
    return img_ref_wat_cor
    
    
    
    
    
    
    