import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from data import *  # Ensure you have your data processing functions defined here

class MyUnet:

    def __init__(self):
        self.resize_size = 256
        # Initialize VGG16 once during the class instantiation
        self.vgg = VGG16(include_top=False, weights='imagenet', input_shape=(self.resize_size, self.resize_size, 3))
        self.vgg.trainable = False  # Freeze the layers of VGG16 to prevent training
        # Create the loss model only once
        self.loss_model = Model(inputs=self.vgg.input, outputs=self.vgg.get_layer('block5_conv3').output)

    def load_data(self):
        mydata = dataProcess()
        imgs_train, imgs_train_mask, imgs_train_label = mydata.create_train_data()
        return imgs_train, imgs_train_mask, imgs_train_label

    def get_unet(self):

        def custom_loss_one_layer(y_true, y_pred): 
            weight_p = 0.9999
            weight_m = 0.0001
            loss_p = K.mean(K.square(self.loss_model(y_true) - self.loss_model(y_pred)))
            loss = weight_p * loss_p + weight_m * K.mean(K.square(y_pred - y_true), axis=-1)
            return loss

        input_shirt = Input((self.resize_size, self.resize_size, 3))
        input_mask = Input((self.resize_size, self.resize_size, 1))
        inputs = concatenate([input_shirt, input_mask])

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("conv1 shape:",conv1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("conv2 shape:",conv2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("conv3 shape:",conv3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print ("conv4 shape:",conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print ("conv4 shape:",conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(3, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

        model = Model(inputs=[input_shirt, input_mask], outputs=conv9)

        model.compile(optimizer=Adam(learning_rate=1e-4), loss=custom_loss_one_layer, metrics=[custom_loss_one_layer, 'accuracy'])

        return model

    def train(self):
        print("Loading data...")
        imgs_train, imgs_train_mask, imgs_train_label = self.load_data()
        print("Loading data done.")

        # Check if a new training session or load previous model
        if new_train:
            model = self.get_unet()
            print("Created a new U-Net model.")
        else:
            list_of_models = glob.glob(pervious_model_dir + '/*.hdf5')
            if list_of_models:
                latest_model = max(list_of_models, key=os.path.getctime)
                model = tf.keras.models.load_model(latest_model, custom_objects={'custom_loss_one_layer': custom_loss_one_layer})
                print("Loaded model from: " + latest_model)
            else:
                model = self.get_unet()
                print("Created a new U-Net model.")

        # Define callbacks
        filepath = os.path.join(model_dir, "unet-{epoch:02d}-{custom_loss_one_layer:.2f}-{val_accuracy:.2f}.hdf5")
        model_checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
        tb_cb = TensorBoard(log_dir="./keras_log_perceptual", histogram_freq=1)
        callbacks_list = [model_checkpoint, tb_cb]

        print('Fitting model...')
        history = model.fit([imgs_train, imgs_train_mask], imgs_train_label, batch_size=4, epochs=30, 
                            verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)
        print('Fitting model done.')

if __name__ == '__main__':
    new_train = True
    pervious_model_dir = "./model_perceptualloss_256_conv4"
    model_dir = "./model_perceptualloss_256_conv5_conv5_mse"
    result_dir = "./results_perceptualloss_256_conv5_conv5_mse"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    my_unet = MyUnet()
    my_unet.train()

