import os
import numpy as np
import glob
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define paths
test_data_path = './Conditioning/testing_results_with_conditioning_data'
test_img_folder = 'target_image'
test_mask_folder = 'combined_mask'
model_name = './model_perceptualloss_256_conv5_mse_0.9999_0.0001/unet-19-32.47-0.80.hdf5'
result_dir = './Conditioning/testing_results_with_conditioning_data/warping_results'


def create_test_data():
    i = 0
    resize_size = 256
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    img_folder = os.path.join(test_data_path, test_img_folder)
    mask_folder = os.path.join(test_data_path, test_mask_folder)
    imgs_names = glob.glob(img_folder + "/*.jpg")

    imgdatas = np.ndarray((len(imgs_names), resize_size, resize_size, 3), dtype=np.uint8)
    imgmasks = np.ndarray((len(imgs_names), resize_size, resize_size, 1), dtype=np.uint8)

    for imgname in imgs_names:
        midname = imgname[imgname.rindex("/") + 1:]
        maskname = midname.split("_")[0] + "_00_5.jpg"

        # Load and resize image
        img = load_img(img_folder + "/" + midname, grayscale=False)
        img = img.resize((resize_size, resize_size), Image.BILINEAR)
        img = img_to_array(img)

        # Load and resize mask
        img_mask = load_img(mask_folder + "/" + midname, grayscale=True)
        img_mask = img_mask.resize((resize_size, resize_size), Image.BILINEAR)
        img_mask = img_to_array(img_mask)

        imgdatas[i] = img
        imgmasks[i] = img_mask
        i += 1

    print('loading done')
    return imgdatas, imgmasks, imgs_names


def test(imgs_test, imgs_test_mask, model):
    print('predict test data')
    imgs_test_result = model.predict([imgs_test, imgs_test_mask], batch_size=1, verbose=1)
    return imgs_test_result


def get_model():
    resize_size = 256

    def custom_loss_one_layer(y_true, y_pred): 
        weight_p = 0.9999
        weight_m = 0.0001
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(resize_size, resize_size, 3)) 
        loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output) 
        loss_model.trainable = False

        loss_p = K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
        loss = weight_p * loss_p + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)

        return loss

    model = load_model(model_name, custom_objects={'custom_loss_one_layer': custom_loss_one_layer})
    return model


def save_img(imgs_names, imgs_test, imgs_test_mask, imgs_test_result):
    for i in range(imgs_test.shape[0]):
        img = imgs_test[i]
        img = np.clip(img, 0, 255).astype('uint8')  # Convert the array to uint8 and clip values
        save_dir = os.path.join(result_dir, 'test_input')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_pil = Image.fromarray(img)  # Convert to PIL Image
        img_pil.save(os.path.join(save_dir, imgs_names[i].split('/')[-1]))  # Save the image

    for i in range(imgs_test_mask.shape[0]):
        img = imgs_test_mask[i]
        img = np.clip(img, 0, 255).astype('uint8')  # Convert the mask to uint8 and clip values
        img = np.squeeze(img)  # Remove any singleton dimensions
        save_dir = os.path.join(result_dir, 'test_mask')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_pil = Image.fromarray(img)  # Convert to PIL Image
        img_pil.save(os.path.join(save_dir, imgs_names[i].split('/')[-1]))  # Save the mask image

    for i in range(imgs_test_result.shape[0]):
        img = imgs_test_result[i]
        img = np.clip(img, 0, 255).astype('uint8')  # Convert the result to uint8 and clip values
        save_dir = os.path.join(result_dir, 'test_result')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_pil = Image.fromarray(img)  # Convert to PIL Image
        img_pil.save(os.path.join(save_dir, imgs_names[i].split('/')[-1]))  # Save the result image


if __name__ == '__main__':
    imgs_test, imgs_test_mask, imgs_names = create_test_data()
    model = get_model()
    imgs_test_result = test(imgs_test, imgs_test_mask, model)
    save_img(imgs_names, imgs_test, imgs_test_mask, imgs_test_result)
