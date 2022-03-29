
import numpy as np
import argparse
import os
import glob
from PIL import Image
import rasterio as rio
from tqdm import tqdm 

import utils


def configuration():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("/workdir", "data", "1_input_scenes", "inference"),
        help="data path to train or inference folder containing the scenes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("/workdir", "data", "2_input_tiles", "inference"),
        help="output path to folder for preprocessed tiles",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        default=False,
        help="test and use the model on the inference set instead of training",
    )
    parser.add_argument(
        "--norm-percentile",
        type=int,
        default=0,
        help="percentile used for normilization, maxar images are already normalized, so for them set to 0. e.g. biome images are raw and need to be normilazed, regularly 10th percentile, see scale_pct",
    )
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=100,
        help="factor for resizing inference images to match resolution, model trained on res:30m, maxar res:1.24m",
    )
    args = parser.parse_args()

    arg_vars = vars(args)

    return arg_vars


def scale_pct(x, pct=10):
    '''
    In case of preprocessing raw data (e.g. biome), not needed for maxar data
    '''
    x_scaled = (x-np.nanpercentile(x, pct))/(np.nanpercentile(x, 100-pct) - np.nanpercentile(x, pct))
    x_scaled[x_scaled>1] = 1
    return (x_scaled*255).astype('uint8')


def preprocess_inference(tif_path, out_dir, pct=0, resize_factor=100, size=256, nr_channels=3):
    # create log dir
    scene_name = os.path.splitext(os.path.basename(tif_path))[0]
    scene_out_dir = os.path.join(out_dir, scene_name)
    utils.ensure_directory_existance(scene_out_dir)

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

    # apply normalization if input is not a maxar image and thus --norm-percentile is specified
    if pct != 0:
        red = scale_pct(red, pct)
        green = scale_pct(green, pct)
        blue = scale_pct(blue, pct)

    full_img = np.stack((red, green, blue), axis=2)
    
    # loop over image and create size*size tiles with padding
    for i in range(nr_tiles_h):
        for j in range(nr_tiles_w):
            tile_rgb = full_img[i*size:(i+1)*size,j*size:(j+1)*size,:]
            save_img = Image.fromarray(tile_rgb, 'RGB')
            save_img.save(os.path.join(scene_out_dir, str(i*size) + '_' + str(j*size) + '.png'))
            
    return 




if __name__ == "__main__":
    print("Preprocessing") 
    args = configuration()

    data_path = args['data_path']
    pct = args['norm_percentile']
    out_dir = args['out_dir']
    resize_factor = args['resize_factor']

    # get all scene paths
    if args['inference']:
        scene_paths = glob.glob(data_path +'/*.tif') + glob.glob(data_path +'/*.TIF')
        print('Preprocess' ,len(scene_paths) ,'scenes for inference')

        #loop over all scenes and preprocess them
        for i in tqdm(range(len(scene_paths))):
            preprocess_inference(scene_paths[i], out_dir, pct=pct, resize_factor=resize_factor, size=256, nr_channels=3)
            

    else:
        print('preprocess train scene')

        # In case of default data path, change input data to train directory
        if data_path == os.path.join("workdir", "data", "1_input_scenes", "inference"):
            data_path = os.path.join("workdir", "data", "1_input_scenes", "train")

        # In case of default output path, change output to train directory
        if data_path == os.path.join("workdir", "data", "2_input_tiles", "inference"):
            data_path = os.path.join("workdir", "data", "2_input_tiles", "train")
        
        # run preprocessing for train, should be adjusted for specific input data
        # preprocess_train(dir_path, path_data, size=256)













def preprocess_train(dir_path, path_data, size=256):
    
    scene_name = os.path.split(dir_path)[1]

    all_tif_paths = utils.get_list_of_files(dir_path, ext='.TIF')
    all_anno_paths = utils.get_list_of_files(dir_path, ext='.img')

    rgb_train_dir = os.path.join(path_data, 'rgb_images/train/')
    rgb_val_dir = os.path.join(path_data, 'rgb_images/val/')
    rgb_test_dir = os.path.join(path_data, 'rgb_images/test/')

    anno_train_dir = os.path.join(path_data, 'annotations/train/')
    anno_val_dir = os.path.join(path_data, 'annotations/val/')
    anno_test_dir = os.path.join(path_data, 'annotations/test/')

    for m in tqdm(range(len(all_anno_paths))):
        # Read annotation
        an_path = all_anno_paths[m]
        anno_open = rio.open(an_path)
        anno_classes = anno_open.read(1)
        
        # Read rgb
        file_name = os.path.split(os.path.dirname(an_path))[1]
        reader_r = rio.open([x for x in all_tif_paths if file_name + '_B4'in x][0])
        reader_g = rio.open([x for x in all_tif_paths if file_name + '_B3'in x][0])
        reader_b = rio.open([x for x in all_tif_paths if file_name + '_B2'in x][0])

        # combine bands in array and apply normalization
        scaled_r = scale_pct(reader_r.read(1))
        scaled_g = scale_pct(reader_g.read(1))
        scaled_b = scale_pct(reader_b.read(1))
        rgb_array = np.stack([scaled_r, scaled_g, scaled_b], axis=2)

        #Binarize classes, thin clouds(192) & clouds(255) as class
        anno_bin = (np.logical_or(anno_classes == 192, anno_classes == 255)).astype(np.uint8)


        # pad image
        height = rgb_array.shape[0]
        width = rgb_array.shape[1]
        nr_tiles_h = (height//size + 1)
        nr_tiles_w = (width//size + 1)
        pad_height = nr_tiles_h*size - height
        pad_width = nr_tiles_w*size - width

        red = np.pad(rgb_array[:,:,0], pad_width = ((0,pad_height), (0,pad_width)))
        green = np.pad(rgb_array[:,:,1], pad_width = ((0,pad_height), (0,pad_width)))
        blue = np.pad(rgb_array[:,:,2], pad_width = ((0,pad_height), (0,pad_width)))
        anno_pad = np.pad(anno_bin, pad_width = ((0,pad_height), (0,pad_width)))

        full_img = np.stack((red, green, blue), axis=2)

        # Create all dirs for each file name
        save_train_dir_rgb = os.path.join(rgb_train_dir, scene_name, file_name) + '/'
        save_val_dir_rgb = os.path.join(rgb_val_dir, scene_name, file_name) + '/'
        save_test_dir_rgb = os.path.join(rgb_test_dir, scene_name, file_name) + '/'
        save_train_dir_an = os.path.join(anno_train_dir, scene_name, file_name) + '/'
        save_val_dir_an = os.path.join(anno_val_dir, scene_name, file_name) + '/'
        save_test_dir_an = os.path.join(anno_test_dir, scene_name, file_name) + '/'

        utils.ensure_directory_existance(save_train_dir_rgb)
        utils.ensure_directory_existance(save_val_dir_rgb)
        utils.ensure_directory_existance(save_test_dir_rgb)
        utils.ensure_directory_existance(save_train_dir_an)
        utils.ensure_directory_existance(save_val_dir_an)
        utils.ensure_directory_existance(save_test_dir_an)

        
        # loop over image and create size*size tiles with padding
        for i in range(nr_tiles_h):
            for j in range(nr_tiles_w): #might need to swap nr_tiles_h/w
                tile_rgb = full_img[i*size:(i+1)*size,j*size:(j+1)*size,:]#/255 they will get preprocessed
                tile_anno = anno_pad[i*size:(i+1)*size,j*size:(j+1)*size]
                
                # save all 256x256 images & annotations when the rgb image is not empty
                if np.sum(tile_rgb) > 1000:
                    save_img = Image.fromarray(tile_rgb, 'RGB')
                    save_anno = Image.fromarray(tile_anno)

                    random_number = random.randrange(10)
                    if random_number < 8: #save as train
                        save_img.save(save_train_dir_rgb + str(i*size) + '_' + str(j*size) + '.png')
                        save_anno.save(save_train_dir_an + str(i*size) + '_' + str(j*size) + '.png')
                    elif random_number == 8: #save as val
                        save_img.save(save_val_dir_rgb + str(i*size) + '_' + str(j*size) + '.png')
                        save_anno.save(save_val_dir_an + str(i*size) + '_' + str(j*size) + '.png')
                    else: #save as test
                        save_img.save(save_test_dir_rgb + str(i*size) + '_' + str(j*size) + '.png')
                        save_anno.save(save_test_dir_an + str(i*size) + '_' + str(j*size) + '.png')


    return
