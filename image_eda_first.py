###重複画像の発見と全データでの統計値出す


from tqdm import tqdm
import numpy as np
import pandas as pd 
import os,cv2,glob
import cv2,sys
import imagehash
from PIL import Image

funcs = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash,
    #lambda x: imagehash.whash(x, mode='db4'),
]



def get_stat(paths):
    x_tot = []
    x_tot_2 = []
    for path in tqdm(paths):
        img = cv2.imread(path)
        x_tot.append((img/255.0).reshape(-1,3).mean(0))
        x_tot_2.append(((img/255.0)**2).reshape(-1,3).mean(0)) 

    img_avr =  np.array(x_tot).mean(0)
    img_std =  np.sqrt(np.array(x_tot_2).mean(0) - img_avr**2)
    print("total_num:",len(paths),'mean:',img_avr, ', std:', np.sqrt(img_std))


def get_hash(paths,threshold = 0.9):
    """
    paths:[~~.png,===.png]というlist
    """
    import torch
    hashes = []
    for path in tqdm(paths):
        image = cv2.imread(path)
        image = Image.fromarray(image)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
    hashes = torch.Tensor(np.array(hashes).astype(int)).cuda()
    sims = np.array([(hashes[i] == hashes).sum(dim=1).cpu().numpy()/256 for i in range(hashes.shape[0])])

    duplicates = np.where(sims > threshold)

    cnt = 0
    for i,j in zip(*duplicates):
        if i == j:continue
        cnt+=1
    print("DUPLICATE",cnt)
    return 


