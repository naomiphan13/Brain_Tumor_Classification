### 1. Download Brain Tumor MRI dataset from Kaggle ###
# Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Training
import os
import kagglehub # pip install kagglehub in Terminal (pip install kagglehub)
import shutil
import scipy.io

CURRENT_DIR = os.getcwd()
DATA_FOLDER = "data"
try:
    os.mkdir(DATA_FOLDER)
    print(f"Directory '{DATA_FOLDER}' created successfully.")
except FileExistsError:
    print(f"Directory '{DATA_FOLDER}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")
		
DATA_DIR = f"{CURRENT_DIR}/{DATA_FOLDER}"

IMG_SIZE = 128

def download_data():
    # Download latest version of the dataset
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    print("Path to dataset files:", path)
    return path

# Move the dataset to the data folder
# # We give it a specific name (e.g., 'BrainTumorMRI') to keep things organized
def move_downloaded_data(path):
    dataset_name = "BrainTumorMRI"
    destination_path = os.path.join(DATA_DIR, dataset_name)

    # 2. Check if the destination already exists to prevent overwriting/errors
    if os.path.exists(destination_path):
        print(f"The folder '{destination_path}' already exists. Skipping move.")
    else:
        # 3. Move the files from the Kaggle cache (path) to your local data folder
        shutil.move(path, destination_path)
        print(f"Success! Dataset moved to: {destination_path}")

### 2. Preprocessing data ###
# Using the pre-processing procedure provided by the author of the dataset
# Source: https://github.com/masoudnick/Brain-Tumor-MRI-Classification/blob/main/Preprocessing.py

import numpy as np 
from tqdm import tqdm
import cv2 # pip install via terminal (pip install opencv-python)
import imutils # pip install via terminal (pip install imutils)

def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img

def convert_to_grayscale(path):
    # 1. Load image in Grayscale (0 for grayscale mode)
    image = cv2.imread(path, 0)

    # 2. Convert to float and divide by 255
    # This squishes the range from [0, 255] to [0.0, 1.0]
    norm_image = image.astype('float32') / 255.0

    return norm_image
	
if __name__ == "__main__":
    path = download_data()
    move_downloaded_data(path)
	
    # Reshize Image and Convert to Grayscale floats
    if os.path.exists("data/cleaned"):
        print("Images have been resized. Skipping...")
    else:
        training = "data/BrainTumorMRI/Training"
        testing = "data/BrainTumorMRI/Testing"
        
        training_dir = os.listdir(training)
        testing_dir = os.listdir(testing)
        
        train_data = []
        for dir in training_dir:
            save_path = 'data/cleaned/Training/'+ dir
            path = os.path.join(training,dir)
            image_dir = os.listdir(path)
            for img in image_dir:
                image = cv2.imread(os.path.join(path,img))
                new_img = crop_img(image)
                new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                image_path = save_path+'/'+img
                cv2.imwrite(image_path, new_img)
                image_norm = convert_to_grayscale(image_path) # Convert to grayscale floats
                col_vector = image_norm.reshape(-1, 1) # Convert to Column Vector
                train_data.append(col_vector)
        Xtr = np.hstack(train_data) # Convert to data matrix
        
        test_data = []
        for dir in testing_dir:
            save_path = 'data/cleaned/Testing/'+ dir
            path = os.path.join(testing,dir)
            image_dir = os.listdir(path)
            for img in image_dir:
                image = cv2.imread(os.path.join(path,img))
                new_img = crop_img(image)
                new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                image_path = save_path+'/'+img
                cv2.imwrite(image_path, new_img)
                image_norm = convert_to_grayscale(image_path)
                col_vector = image_norm.reshape(-1, 1)
                test_data.append(col_vector)
        Xte = np.hstack(test_data)

    # Save to matlab data (.mat)

    scipy.io.savemat('data/Xtr_tumor.mat', {"Xtr": Xtr})
    scipy.io.savemat('data/Xte_tumor.mat', {"Xte": Xte})

    print("Data saved successfully without labels.")


