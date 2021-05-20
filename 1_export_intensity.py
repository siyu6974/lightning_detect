from glob import glob
import numpy as np
import cv2

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt


files = sorted(glob('frames/*.jpg'))

def imgFile2intensity(f):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    s = np.sum(img)
    return f, s

import multiprocessing as mp
if __name__ == '__main__':

    with mp.Pool(8) as pool:
        r = pool.map(imgFile2intensity, files)

        np.save('r.npy', r)



