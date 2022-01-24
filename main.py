import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 
from PIL import Image
from sklearn.metrics import confusion_matrix
# import cv2 #pip install opencv-python==4.0.0.21
import glob
import pickle
import json

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

from skimage.io import imread, imshow
from skimage.transform import resize

import rasterio as rio
from rasterio.plot import show

import build_CloudXNet 
import utils
from metrics import *
from generator import DataGenerator













def build_model(checkpoint_path, lr=1e-4, size=256, resume=False):

    model, pre_process = build_CloudXNet.model_arch(input_rows=size, input_cols=size, num_of_channels=3, num_of_classes=1)
    model.compile(optimizer = Adam(lr = lr), loss = jacc_loss, metrics = [dice_score,'accuracy'])

    if resume:
        print('Resuming from weights')
        model.load_weights(checkpoint_path)

    return model, pre_process


def train_model(model, train_im, train_an, val_im, val_an, pre_process, log_dir, batch_size=4, epochs=50, size=256, augmentation=False):

    train_gen = DataGenerator(train_im, train_an, pre_process, batch_size=batch_size, width=size,
        height=size, augmentation=augmentation)
    val_gen = DataGenerator(val_im, val_an, pre_process, batch_size=batch_size, width=size,
        height=size, augmentation=augmentation)

    utils.ensure_directory_existance(log_dir)
    log_name = "tensorboard"
    tensorboard = TensorBoard(log_dir=os.path.join(log_dir, log_name))
    checkpoint_path = os.path.join(log_dir, "weights.{epoch:02d}.hdf5")
    checkpointer = ModelCheckpoint(checkpoint_path, monitor= "val_loss", save_best_only = True, save_weights_only=True)

    results = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, callbacks=[tensorboard, checkpointer])

    # Write metrics
    json_path = os.path.join(log_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(str(results.history), f)
        
    return model


def create_tiles_for_prediction(tif_path, resize_factor=100, size=256, nr_channels=3):
    # read downsized image
    img = rio.open(tif_path)
    downsized_img = img.read(out_shape = (img.count, img.height//resize_factor, img.width//resize_factor))

    # pad image
    height = downsized_img.shape[1]
    width = downsized_img.shape[2]
    nr_tiles_h = (height//size + 1)
    nr_tiles_w = (width//size + 1)
    pad_height = nr_tiles_h*size - height
    pad_width = nr_tiles_w*size - width

    red = np.pad(downsized_img[0], pad_width = ((0,pad_height), (0,pad_width)))
    green = np.pad(downsized_img[1], pad_width = ((0,pad_height), (0,pad_width)))
    blue = np.pad(downsized_img[2], pad_width = ((0,pad_height), (0,pad_width)))

    full_img = np.stack((red, green, blue), axis=2)

    # loop over image and create size*size tiles with padding
    k=0
    tile_name = []
    all_tiles = np.zeros((nr_tiles_h*nr_tiles_w, size , size, nr_channels), dtype=np.float32)
    for i in range(nr_tiles_h):
        for j in range(nr_tiles_w): #might need to swap nr_tiles_h/w
            all_tiles[k,:,:,:] = full_img[i*size:(i+1)*size,j*size:(j+1)*size,:]/255
            tile_name.append(str(i*size) + '_' + str(j*size))
            k +=1
    
    return downsized_img, all_tiles, tile_name


def load_test_images(test_im, test_an, size=256):
    X_test = np.zeros((len(test_im), size, size, 3), dtype=np.float32)
    Y_test = np.zeros((len(test_im), size, size), dtype=np.float32)

    for n in range(len(test_im)):
        img = Image.open(test_im[n])
        np_img = (np.array(img)/255).astype(np.float32)
        X_test[n] = np_img
        
        anno = Image.open(test_an[n])
        np_an = (np.array(anno)/255).astype(int)
        Y_test[n] = np_an
        

    return X_test, Y_test
    

def predict(X_test, log_dir, filename, Y_test=None, tile_name=None):

    preds_test = model.predict(X_test, batch_size=2, verbose=1)
    test_pred = (preds_test > 0.5).astype(int)

    for i in range(X_test.shape[0]):
        predictions = np.repeat(test_pred[i,:,:], 3, axis=2)
        
        if Y_test == None: 
            concat_array = np.concatenate((X_test[i], predictions), axis=1)
        else:
            annotations = np.repeat(Y_test[i][...,np.newaxis], 3, axis=2)
            concat_array = np.concatenate((X_test[i], annotations, predictions), axis=1)

        concat_img = Image.fromarray((concat_array*255).astype(np.uint8), 'RGB')

        save_dir = os.path.join(log_dir, filename)
        utils.ensure_directory_existance(save_dir)
        if tile_name==None:
            concat_img.save(os.path.join(save_dir, str(i)+'.png'))
        else:
            concat_img.save(os.path.join(save_dir, tile_name[i]+'.png'))

    return



if __name__ == "__main__":
    print("Initializing")

    # dataset parameters
    dataset = 'dataset_input/' # name of folder 
    path_data = '/Users/Willem/Werk/510/510_cloud_detection/' + dataset #/Users/Willem/Werk/510

    model_name = '0121v0_dataset_50ep' # month, day, version, _model
    log_dir = '/Users/Willem/Werk/510/510_cloud_detection/log/' + model_name  #save weights, results & tensorboard
    checkpoint_path = '/Users/Willem/Werk/510/510_cloud_detection/saved_runs/0119v0_dataset_50ep.h5'
    # maxar_path = '/Users/Willem/Werk/510/510_cloud_detection/geotiff_examples/maxar_clouds_small.tif'
    maxar_path = '/Users/Willem/Werk/510/510_cloud_detection/geotiff_examples/maxar_clouds.tif'


    train = False
    resume = True
    test_model = False
    test_maxar = True
    size=256
    lr=1e-4


    train_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/train/'))
    train_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/train/'))
    val_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/val/'))
    val_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/val/'))
    test_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/test/'))
    test_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/test/'))
    
    # Build model
    print("Building model")
    model, pre_process = build_model(checkpoint_path, lr=lr, size=size, resume=resume)


    # Train model
    if train:
        print("Training")
        model = train_model(model, train_im, train_an, val_im, val_an, pre_process, log_dir, batch_size=4, epochs=50, size=256, augmentation=False)


    # Test model on test set or maxar TIF images
    if test_model:
        print("Testing on dataset")
        X_test, Y_test = load_test_images(test_im, test_an, size=256)
        predict(X_test, Y_test=Y_test, log_dir=log_dir, filename=dataset)


    if test_maxar:
        print("Testing Maxar file")
        downsized_img, X_test, tile_name = create_tiles_for_prediction(maxar_path, resize_factor=100, size=256, nr_channels=3) #resize factor depends on sensor resolution, landsat 30m, maxar 30cm
        predict(X_test, tile_name=tile_name, log_dir=log_dir, filename=os.path.splitext(os.path.basename(maxar_path))[0])


    


    






































