import cv2
import pydicom
import sys
from skimage import exposure
import numpy as np
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder')
    parser.add_argument('--dicom', type=bool)
    parser.add_argument('--pool-sz', type=int)
    parser.add_argument('--output-folder')
    return parser.parse_args()

def resize_dicom(dicom_file, out='./input/train/i512'):
    dicom = pydicom.read_file(dicom_file)
    img = dicom.pixel_array
    img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
    
    # Applying CLAHE
    img = exposure.equalize_adapthist(img, clip_limit=0.0)
    file_name = dicom_file.split('/')[-1].split('.')[0]
    
    output_file = '%s%s%s' % (out, file_name, '.npy')
    np.save(output_file, img)

def resize_img(file, out='./input/train/i512'):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
    
    # Applying CLAHE
    img = exposure.equalize_adapthist(img, clip_limit=0.0)
    
    file_name = file.split('/')[-1].split('.')[0]
    output_file = '%s%s%s' % (out, file_name, '.npy')
    np.save(output_file, img)


def main():
    args = get_args()
    
    pool   = ThreadPool(args.pool_sz)
    images = glob.glob(args.input_folder)

    if args.dicom == True:
        pool.map(resize_dicom, images)
    else:
        pool.map(resize_img, images)

if __name__ == "__main__":
    print(sys.argv)
    main()
