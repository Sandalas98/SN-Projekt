import numpy as np
from PIL import Image, ImageOps
import os
import threading
import time

train_labels = np.genfromtxt('./data/boneage-training-dataset.csv', delimiter=',')
test_labels = np.genfromtxt('./data/boneage-test-dataset.csv', delimiter=',')

path = './data/test_fol'

images_count = 0
images_processed = 0

def ShowStatus():
    while True:
        print(f'{images_processed}/{images_count}', end="")
        print("\r", end="") 
        time.sleep(100)


th = threading.Thread(target=ShowStatus)

for img in os.listdir(path):
    images_count += 1

th.start()

for img in os.listdir(path):
    image = Image.open(path+'/'+img)
    image = ImageOps.grayscale(image)
    image = image.resize((512,512))
    image = np.array(image)
    max_higness = np.max(image)
    mean_higness = np.mean(image)

    if mean_higness < .15 * 255:
        continue
    
    
    
    for i in range(1,image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            if image[i, j] >= 0.8*max_higness:
                image[i, j] = mean_higness
                image[i-1, j] = mean_higness
                image[i+1, j] = mean_higness
                image[i, j-1] = mean_higness
                image[i, j+1] = mean_higness
    
    
    image = Image.fromarray(image)
    image.save(path+'/post_'+img)

    images_processed += 1
    
th.join()
