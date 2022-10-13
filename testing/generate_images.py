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

if __name__ == "__main__":
    generate_images_diff_dimension()
    generate_images_diff_type()
