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
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

from skimage.io import imread, imshow
from skimage.transform import resize

import rasterio as rio
from rasterio.plot import show

import build_CloudXNet 
import build_UNet
import utils
import config
from metrics import *
from generator import DataGenerator





def pre_process_max(X):
    return X/np.max(X) #specify type?


def build_model(checkpoint_path, model_name='CloudXNet', lr=1e-4, size=256, resume=False, freeze_backbone = False):

    if model_name == 'CloudXNet':
        model, pre_process = build_CloudXNet.model_arch(input_rows=size, input_cols=size, num_of_channels=3, num_of_classes=1)
    elif model_name == 'UNet':
        model, pre_process = build_UNet.build_unet(input_size=(size,size,3), freeze_backbone=freeze_backbone)
    

    # print(model.summary())
    
    model.compile(optimizer = Adam(lr = lr), loss = fbeta_loss, metrics = [dice_score, fbeta_score, 'accuracy'])

    if resume:
        print('Resuming from weights')
        model.load_weights(checkpoint_path)

    return model, pre_process


def train_model(model, train_im, train_an, val_im, val_an, pre_process, log_dir, model_name, batch_size=4, epochs=50, size=256, augmentation=False):

    train_gen = DataGenerator(train_im, train_an, pre_process, batch_size=batch_size, width=size,
        height=size, augmentation=augmentation)
    val_gen = DataGenerator(val_im, val_an, pre_process, batch_size=batch_size, width=size,
        height=size, augmentation=augmentation)

    utils.ensure_directory_existance(log_dir)
    log_name = "tensorboard"

    if model_name == 'UNet':
        pre_run_epochs = 3
        print('Train with frozen backbone for nr epochs:', pre_run_epochs)

        # First run
        print(model.summary())
        utils.ensure_directory_existance(os.path.join(log_dir, 'run_1'))
        tensorboard = TensorBoard(log_dir=os.path.join(log_dir, 'run_1', log_name))
        checkpoint_path = os.path.join(log_dir, 'run_1', "weights.{epoch:02d}.hdf5")
        checkpointer = ModelCheckpoint(checkpoint_path, monitor= "val_loss", save_weights_only=True, save_freq='epoch')
        results_1 = model.fit(train_gen, validation_data=val_gen, epochs=pre_run_epochs, verbose=1, callbacks=[tensorboard, checkpointer])

        # Second run
        print('Building second model without frozen weights')
        utils.ensure_directory_existance(os.path.join(log_dir, 'run_2'))
        weights_path = os.path.join(log_dir, 'run_1', "weights.0{}.hdf5".format(pre_run_epochs))
        model_2, pre_process = build_model(weights_path, model_name=model_name, size=size, resume=True, freeze_backbone=False)
        print(model_2.summary())

        tensorboard = TensorBoard(log_dir=os.path.join(log_dir, 'run_2', log_name))
        checkpoint_path = os.path.join(log_dir, 'run_2', "weights.{epoch:02d}.hdf5")
        checkpointer = ModelCheckpoint(checkpoint_path, monitor= "val_loss", save_best_only = True, save_weights_only=True)

        print("Training second model for nr_epochs:", epochs-pre_run_epochs)
        results = model_2.fit(train_gen, validation_data=val_gen, epochs=epochs-pre_run_epochs, verbose=1, callbacks=[tensorboard, checkpointer])


    else:
        tensorboard = TensorBoard(log_dir=os.path.join(log_dir, log_name))
        checkpoint_path = os.path.join(log_dir, "weights.{epoch:02d}.hdf5")
        checkpointer = ModelCheckpoint(checkpoint_path, monitor= "val_loss", save_best_only = True, save_weights_only=True)#, save_freq='epoch')

        results = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, callbacks=[tensorboard, checkpointer])

    # Write metrics
    json_path = os.path.join(log_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(str(results.history), f)
        
    return model

# def predict(X_test, pre_process, log_dir, filename, Y_test=None, tile_name=None):
#     pre_processed_X = np.zeros(X_test.shape, dtype=np.float32)

#     for i in range(X_test.shape[0]):
#         pre_processed_X[i,:,:,:] = pre_process(X_test[i,:,:,:])

#     preds_test = model.predict(pre_processed_X, batch_size=2, verbose=1) # X_test between 0 and 1
#     test_pred = (preds_test > 0.5).astype(int)

#     for i in range(X_test.shape[0]):
#         predictions = np.repeat(test_pred[i,:,:], 3, axis=2)
        
#         if isinstance(Y_test, np.ndarray): 
#             annotations = np.repeat(Y_test[i][...,np.newaxis], 3, axis=2)
#             concat_array = np.concatenate((X_test[i], annotations*255, predictions*255), axis=1)
#         else:
#             concat_array = np.concatenate((X_test[i], predictions*255), axis=1)

#         concat_img = Image.fromarray(concat_array.astype(np.uint8), 'RGB')

#         save_dir = os.path.join(log_dir, filename + '_0.5')
#         utils.ensure_directory_existance(save_dir)

#         concat_img.save(os.path.join(save_dir, tile_name[i]+'.png'))
#     return


def predict(data_path, predictions_dir, pre_process, threshold = 0.5):
    # load tiles 
    scenes = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,x))]

    # run over each scene
    for i in tqdm(range(len(scenes))):
        # create output folder for scene i and get all files of that scene
        utils.ensure_directory_existance(os.path.join(predictions_dir, scenes[i]))
        scene_tiles = glob.glob(os.path.join(data_path, scenes[i], '*.png'))
        
        # create outpaths with tile name
        out_paths = []
        for j in range(len(scene_tiles)):
            out_path_tile = os.path.join(predictions_dir, scenes[i], os.path.basename(scene_tiles[j])) 
            out_paths.append(out_path_tile) 

        # predict
        predict_gen = DataGenerator(scene_tiles, annotations=None, preprocess_input=pre_process, batch_size=len(scene_tiles), to_fit=False, shuffle=False)
        probabilities = model.predict(predict_gen)
        pred = np.uint8((probabilities > threshold)*255)

        # save predictions for all tiles
        for j in range(len(scene_tiles)):
            save_tile = Image.fromarray(pred[j,:,:,0])
            save_tile.save(out_paths[j])
    

    return



if __name__ == "__main__":
    print("Initializing")
    args = config.configuration()

    # dataset parameters
    data_path = args['data_path']
    train_log_dir = args['checkpoint_path']
    predictions_dir = os.path.join(args['prediction_path'], args['run_name'])
    path_to_weights = args['weights_path']
    
    
    resume = not args['scratch']
    
    model_name = args['model_type'] #'CloudXNet', 'UNet'
    epoch=args['number_of_epochs']
    batch_size=args['batch_size']
    lr = args['learning_rate']
    pred_threshold = args['pred_threshold']
    size = 256

    # Build model
    print("Building model")
    model, pre_process = build_model(path_to_weights, model_name=model_name, lr=lr, size=size, resume=resume, freeze_backbone=True)


    # Train model
    if not args['inference']:
        print("Training")
        # Get data paths
        train_im = utils.get_list_of_files(os.path.join(data_path, 'rgb_images/train/'), ext='.png')
        train_an = utils.get_list_of_files(os.path.join(data_path, 'annotations/train/'), ext='.png')
        val_im = utils.get_list_of_files(os.path.join(data_path, 'rgb_images/val/'), ext='.png')
        val_an = utils.get_list_of_files(os.path.join(data_path, 'annotations/val/'), ext='.png')

        model = train_model(model, train_im, train_an, val_im, val_an, pre_process, log_dir, model_name, batch_size=batch_size, epochs=epoch, size=256, augmentation=False)

    # Inference
    if args['inference']:
        print("Predicting")

        #in case of testing CloudXNet model (preprocess test: /max, preprocess train: /255 (which is the returned preprocess function by build_model))
        if model_name == 'CloudXNet':
            pre_process = pre_process_max 

        predict(data_path, predictions_dir, pre_process, pred_threshold)














    # dataset = 'biome_input/' # name of folder 
    # path_data = '/' + dataset #/Users/Willem/Werk/510, /home/NC6user/510_cloud_detection/log/0209v0_biome_20ep

    # log_name = '0218v2_biome_30ep' # month, day, version, _model
    # log_dir = '/log/' + log_name  #save weights, results & tensorboard /home/NC6user/510_cloud_detection/log/
    # path_to_weights = '/Users/Willem/Werk/510/510_cloud_detection/log/0215v0_biome_30ep/run_2/weights.21.hdf5' #'/Users/Willem/Werk/510/510_cloud_detection/log/0124v0_dataset_38_10ep/weights.08.hdf5' #'/Users/Willem/Werk/510/510_cloud_detection/log/0119v0_dataset_50ep/0119v0_dataset_50ep.h5'
    # inference_path = '/Users/Willem/Werk/510/510_cloud_detection/geotiff_examples/maxar_white_buildings_1.tif'

    # train = True
    # resume = False
    # test_model = False
    # test_maxar = False 
    # model_name = 'UNet' #'CloudXNet', 'UNet'
    # epoch=30
    # batch_size=32
    # size=256
    # lr=1e-4
    # resize_factor = 100
    # freeze_backbone = True # only in case of UNet
    


    # train_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/train/'), ext='.png')
    # train_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/train/'), ext='.png')
    # val_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/val/'), ext='.png')
    # val_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/val/'), ext='.png')
    # test_im = utils.get_list_of_files(os.path.join(path_data, 'rgb_images/test/'), ext='.png')
    # test_an = utils.get_list_of_files(os.path.join(path_data, 'annotations/test/'), ext='.png')
    
    # # Build model
    # print("Building model")
    # model, pre_process = build_model(path_to_weights, model_name=model_name, lr=lr, size=size, resume=resume, freeze_backbone=freeze_backbone)


    # # Train model
    # if train:
    #     print("Training")
    #     model = train_model(model, train_im, train_an, val_im, val_an, pre_process, log_dir, model_name, batch_size=batch_size, epochs=epoch, size=size, augmentation=False)


    # # Test model on test set or maxar TIF images
    # if test_model:
    #     print("Testing on dataset")
    #     X_test, Y_test, image_name = load_test_images(test_im, test_an, size=size)
    #     predict(X_test, Y_test=Y_test, tile_name=image_name, log_dir=log_dir, pre_process=pre_process, filename=dataset)


    # if test_maxar:
    #     print("Testing Maxar file")
    #     # pre_process = pre_process_max #in case of testing CloudXNet model (preprocess train: /255, preprocess test: /max)
    #     downsized_img, X_test, tile_name = create_tiles_for_prediction(inference_path, resize_factor=resize_factor, size=size, nr_channels=3) #resize factor depends on sensor resolution, landsat 30m, sentinel 10m, maxar 30cm & dataset resized 512->256 & 384 ->256
    #     predict(X_test, pre_process=pre_process, tile_name=tile_name, log_dir=log_dir, filename=os.path.splitext(os.path.basename(inference_path))[0])


    


    







































