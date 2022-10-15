import os
import cv2
import numpy as np
from PIL import Image

output_path = "testing/testing_files/test_images"
image_path = "model_training/Photos/InSitu/is001.tif"
def generate_images_diff_dimension():
    """
    function to generate an image with different dimensions
    """
    dimensions = [(1000,1000),(660,780),(470,625),(450,625),(400,515),(322,400),(50,63),(7,11),(4,4),(1,1)]
    try:
        os.mkdir(output_path)    
    except FileExistsError:
        pass
    finally:
        for dim in dimensions:
            read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))                
            img = read(image_path)                      
            img = cv2.resize(img, dim) 
            filepath = f"{output_path}/{dim[0]}_{dim[1]}.png"
            data = Image.fromarray(img)
            data.save(filepath)

def generate_images_diff_type():
    """
    Function to generate a given image in different extension
    """
    filetype = ["png", "jpeg", "jpg", "tif", "pdf"]
    try:
        os.mkdir(output_path)    
    except FileExistsError:
        pass
    finally:
        for ftype in filetype:
            read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))                
            img = read(image_path)                      
            img = cv2.resize(img, (224,224)) 
            filepath = f"{output_path}/{ftype}_image.{ftype}"
            data = Image.fromarray(img)
            data.save(filepath)       

def generate_image_grayscale():
    """
    Function to generate a given image in grayscale
    """
    try:
        os.mkdir(output_path)   
    except FileExistsError:
        pass
    finally:
        read = lambda imname: np.asarray(Image.open(imname).convert("L"))                
        img = read(image_path)                      
        filepath = f"{output_path}/grayscale_image_insitu.png"
        data = Image.fromarray(img)
        data.save(filepath)

def generate_image_CMYK():
    """
    Function to generate a given image in CMYK
    """
    try:
        os.mkdir(output_path)   
    except FileExistsError:
        pass
    finally:
        image = cv2.imread(image_path)
        img = image.astype(np.float64)/255.
        K = 1 - np.max(img, axis=2)
        C = (1-img[...,2] - K)/(1-K)
        M = (1-img[...,1] - K)/(1-K)
        Y = (1-img[...,0] - K)/(1-K)
        CMYK_image= (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
        os.chdir(output_path)
        cv2.imwrite('CMYK_image_insitu.png', CMYK_image)

if __name__ == "__main__":
    generate_images_diff_dimension()
    generate_images_diff_type()
    generate_image_grayscale()
    generate_image_CMYK()
    
