import os
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

class dataProcess:
    def __init__(self, data_path="./MVC_image_pairs_resize_new/shirts_1", 
                 mask_path="./MVC_image_pairs_resize_new/fc8_mask_5_modified", 
                 label_path="./MVC_image_pairs_resize_new/shirts_5", 
                 img_type="jpg"):
        
        self.data_path = data_path
        self.mask_path = mask_path
        self.label_path = label_path
        self.img_type = img_type
        self.resize_size = 256

    def create_train_data(self):
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(os.path.join(self.data_path, f"*.{self.img_type}"))
        print(f'Total training images found: {len(imgs)}')
        
        imgdatas = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)
        imgmasks = np.ndarray((len(imgs), self.resize_size, self.resize_size, 1), dtype=np.float32)
        imglabels = np.ndarray((len(imgs), self.resize_size, self.resize_size, 3), dtype=np.float32)

        for i, imgname in enumerate(imgs):
            midname = os.path.basename(imgname)
            maskname = f"{midname.split('_')[0]}_00_5.jpg"

            img = load_img(imgname, color_mode='rgb')
            img_mask = load_img(os.path.join(self.mask_path, maskname), color_mode='grayscale')
            label = load_img(os.path.join(self.label_path, midname), color_mode='rgb')

            img = img.resize((self.resize_size, self.resize_size), Image.BILINEAR)
            img_mask = img_mask.resize((self.resize_size, self.resize_size), Image.BILINEAR)
            label = label.resize((self.resize_size, self.resize_size), Image.BILINEAR)

            imgdatas[i] = img_to_array(img)
            imgmasks[i] = img_to_array(img_mask)
            imglabels[i] = img_to_array(label)

            if i % 100 == 0:
                print(f'Done: {i}/{len(imgs)} images')

        return imgdatas, imgmasks, imglabels
