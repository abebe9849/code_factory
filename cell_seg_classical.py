"""
古典的手法、アノテーション不要なのが良いが精度不安
watershedの何処から水を流すか、どの領域目安で流すか　を考えると脂肪壁のにじみを対処できた

"""

import numpy as np 
import pandas as pd 
import glob
import openslide
import matplotlib.pyplot as plt
import cv2,math,sys,time,os
from tqdm import tqdm
import itertools
import warnings


def water_shed(gray_img,color_img,p=0.1):
        ### ~~~~~~~~~~
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sure_bg = cv2.dilate(gray_img, kernel, iterations=2)## はいぱら
    dist = cv2.distanceTransform(gray_img, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, p * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    #fig = plt.figure(figsize=(20,20))
    #plt.imshow(dist,cmap="gray")
    #plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(markers)
    plt.show()
    """
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_img, markers)
    return markers

def large_tile(df_tmp,idx):
    img =cv2.imread(df_tmp["file_png"].values[idx])
    img_l_path = df_tmp["file_ndpi"].values[idx]
    img_y = df_tmp["np_y"].values[idx]
    img_x = df_tmp["np_x"].values[idx]
    slide = openslide.OpenSlide(img_l_path)
    pil_img = slide.read_region((img_x*8,img_y*8),0,(128*8,128*8))
    np_img = np.asarray(pil_img)[:,:,:3]
    return np_img

def img_and(mask,canny):
    """
    共にuint8のndarray0~255
    img1:mask
    img2:canny
    
    """
    canny = (255-canny)/255
    mask = mask/255
    
    new_img = np.logical_and(canny,mask)
    return new_img
    
def sharpen_image(img,k):
    #kernel = np.array([[-k / 9, -k / 9, -k / 9],[-k / 9, 1 + 8 * k / 9, k / 9],[-k / 9, -k / 9, -k / 9]], np.float32)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32) 
    new_img = cv2.filter2D(img, -1, kernel).astype("uint8")
    #new_img2 = cv2.filter2D(new_img, -1, kernel).astype("uint8")
    return new_img

def each_cell(markers,thresh=0.03,plot=True):
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            margin_num+=1
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    #print(back_area)
    return emp,areas,back_area,margin_num

def margin_to_black(markers,color_img,gray_img,distant=None,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False):
    u, counts = np.unique(markers.ravel(), return_counts=True)

    """
    for cls,s in zip(u,counts):
        if s >1024*1024*thresh_margin:
            gray_img[np.where(markers == cls)]=0"""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sure_bg = cv2.dilate(gray_img, kernel, iterations=2)## はいぱら
    if distant is None:
        
        dist = cv2.distanceTransform(gray_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max(), 255, cv2.THRESH_BINARY)
    else:
        """
        for cls,s in zip(u,counts):
            if s >1024*1024*thresh_margin:
                distant[np.where(markers == cls)]=0"""
        dist = cv2.distanceTransform(distant, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, p * dist.max()*0.05, 255, cv2.THRESH_BINARY)

    
    sure_fg = sure_fg.astype(np.uint8)
    #fig = plt.figure(figsize=(20,20))
    #print("distant!!!!")
    #plt.imshow(dist,cmap="gray")
    #plt.show()
    #plt.imshow(gray_img,cmap="gray")
    #plt.show()
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    """
    fig = plt.figure(figsize=(20,20))
    plt.imshow(markers)
    plt.show()
    """
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(color_img, markers)
    
    u, counts = np.unique(markers.ravel(), return_counts=True)
    areas = []
    emp = np.zeros_like(markers)
    margin_num = 0
    for cls,s in zip(u,counts):
        if cls<0:
            """
            tmp_img = (markers == cls)
            plt.imshow(tmp_img)
            plt.show()
            """
            continue#枠(-1)
        if s<500:continue
        if s >1024*1024*thresh:
            margin_num+=1
            if plot:
                tmp_img = (markers == cls)
                plt.title(f"class{cls} area{s}")
                plt.imshow(tmp_img)
                plt.show()
            
            continue#大きい余白を除く
        tmp_img = (markers == cls)
        emp[np.where(markers == cls)]=cls
        if cls%200==0 and plot:
            plt.title(f"class{cls} area{s}")
            plt.imshow(tmp_img)
            plt.show()
        areas.append(s)
        #if cls>20:break
    back_area = counts[2]
    #print(back_area)
    return emp,areas,back_area,margin_num

def analyze_data(areas,backs,figname):
    nums = [len(area) for area in areas]
    plt.figure()
    plt.hist(np.array(nums),bins=100)
    if figname:
        plt.savefig(f"{figname}_num.png")
    else:
        plt.show()
    areas_f = list(itertools.chain.from_iterable(areas))
    plt.figure()
    plt.title(f"mean{np.mean(np.array(areas_f))},medain{np.median(np.array(areas_f))}")
    plt.hist(np.array(areas_f),bins=100)
    if figname: 
        plt.savefig(f"{figname}_areas.png")
    else:
        plt.show()
    plt.figure()
    plt.title(f"mean{np.mean(np.array(backs))},medain{np.median(np.array(backs))}")
    plt.hist(np.array(backs),bins=100)
    if figname:
        plt.savefig(f"{figname}_backs.png")
    else:
        plt.show()
    

def calc_analyze(df_tmp,label,num):
    areas_ = []
    backgrounds_ = []
    for idx in tqdm(range(num)):
        if label==0:
            img = large_tile(predcit,idx)
        else:
            img = large_tile(predcit,-(idx+1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        retval, mask = cv2.threshold(gray, thresh=gray.mean(0).mean(0), maxval=255, type=cv2.THRESH_BINARY)
        markers = water_shed(mask,img)
        new_markers,areas,back_area,margin_num = each_cell(markers,plot=False)#こちらのほうがきれいにとれる
        if len(areas)<20:##len(areas)<20のものはなにか問題ある(余白が半分以上、他の組織、ぼやけてる)
            gray_3 = cv2.Canny(gray,0,30)
            kernel_size = (5,5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            dilation = cv2.dilate(gray_3,kernel,iterations = 1)
            dilation = 255-dilation
            gray_3 = img_and(mask,gray_3)
            gray_3 = (gray_3*255).astype("uint8")
            gray_color = cv2.cvtColor(gray_3,cv2.COLOR_GRAY2RGB)
            new_markers_2,areas_new,back_area_new,margin_num_new=margin_to_black(markers,img,mask,distant=dilation,p=0.1,thresh = 0.03,thresh_margin=0.05,plot=False)
            
        """
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1, 4, 1)
        plt.imshow(img)
        ax2 = fig.add_subplot(1, 4, 2)
        plt.imshow(markers)
        ax3 = fig.add_subplot(1, 4, 3)
        plt.imshow(new_markers)
        """
        backgrounds_.append(back_area)
        if len(areas)<20 :
            #ax4 = fig.add_subplot(1, 4, 4)
            #plt.title(f"old {len(areas)}:{margin_num}^new{len(areas_new)}:{margin_num_new}")
            #plt.imshow(new_markers_2)
            areas_.append(areas_new)
        else:
            areas_.append(areas)
        #plt.show()
        
    return areas_,backgrounds_


"""
#predcitはDataFrame
areas,backgrounds = calc_analyze(predcit,1,4000)
analyze_data(areas,backgrounds,figname="analyze_fat/pred_1")
"""


