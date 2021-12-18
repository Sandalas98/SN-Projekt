import numpy as np
from PIL import Image, ImageOps
import os
import threading
import time

def preprocessing_x_ray(path = './data/test_fol'):
    train_labels = np.genfromtxt('./data/boneage-training-dataset.csv', delimiter=',')
    test_labels = np.genfromtxt('./data/boneage-test-dataset.csv', delimiter=',')

    images_amount = 0
    images_count = 0
    images_processed = False

    def ShowStatus():
        while not images_processed:
            print(f'Zdjęcie nr {images_count}', end="")
            print("\r", end="") 


    # Wątek do wyświetlania postępu przetwarzania zdjęć
    th = threading.Thread(target=ShowStatus)

    for img in os.listdir(path):
        images_amount += 1

    th.start()

    # Dla każdego zdjęcia zrób:
    # 1. Zamiana na skalę szarości
    # 2. Zmnień rozmiar na 512x512
    # 3. Wyznacz najjaśniejszy piksel
    # 3.1 Wszystkie piksele które mają więcej jasności niż 80% najjaśniejszego piksela zastąp wartością średnią
    #     barwy wszystkich pikseli
    # 3.2 W tym celu użyj filtra do "wyczyszczenia" literek
    # Zapisz zdjęcie


    # usunąć ,,jasne" zdjęcia
    # przejście po wszystkich 

    for img in os.listdir(path):
        print(path+'/'+img)
        if img == 'post':
            continue
        image = Image.open(path+'/'+img)
        image = ImageOps.grayscale(image)
        image = image.resize((512,512))
        image = np.array(image)
        #image = image.reshape((512, 512, 1))
        
        max_higness = np.max(image)
        mean_higness = np.mean(image)

        if mean_higness > 100:
            continue

        print(f'Przetworzono: {img}')
        
        '''
        Funkcja czyszczenia:

            #
          # # #
        # # # # #
          # # #
            #
        
        '''

        for i in range(2,image.shape[0]-2):
            for j in range(2, image.shape[1]-2):
                if image[i, j] >= 0.8*max_higness:
                    image[i, j] = mean_higness
                    
                    image[i-2, j] = mean_higness
                    image[i-1, j] = mean_higness
                    
                    image[i+1, j] = mean_higness
                    image[i+2, j] = mean_higness

                    image[i, j-1] = mean_higness
                    image[i, j-2] = mean_higness

                    image[i, j+1] = mean_higness
                    image[i, j+2] = mean_higness

                    image[i-1, j-1] = mean_higness
                    image[i-1, j+1] = mean_higness
                    image[i+1, j-1] = mean_higness
                    image[i+1, j+1] = mean_higness

        image = Image.fromarray(image)
        
        image.save(path+'/post/'+img)

        images_count += 1

    images_processed = True
        
    th.join()