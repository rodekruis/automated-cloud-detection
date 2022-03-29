import numpy as np
import argparse
import os
import glob
from PIL import Image
import rasterio as rio
from tqdm import tqdm 

from osgeo import gdal, ogr

import utils
import gdal_polygonize
import gdal



def configuration():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "--input-path",
        type=str,
        default=os.path.join("/workdir", "data", "3_prediction_tiles"),
        help="path to the input tiles (predictions)",
    )
    parser.add_argument(
        "--tif-input-path",
        type=str,
        default=os.path.join("workdir", "data", "1_input_scenes", "inference"),
        help="path to the input tiles (predictions)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join("/workdir", "data", "4_prediction_scenes"),
        help="path to the output scene",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="name to identify execution, if not supplied will merge the predictions of all scenes",
    )
    parser.add_argument(
        "--return-png",
        action="store_true",
        default=False,
        help="returns output as a png file",
    )
    parser.add_argument(
        "--return-poly",
        action="store_true",
        default=False,
        help="returns a list of polygons of clouds",
    )    
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=100,
        help="factor for resizing the tiles back to the original tif size, model trained on res:30m, maxar res:1.24m",
    )

    args = parser.parse_args()

    arg_vars = vars(args)

    return arg_vars


def combine_tiles(input_path, tif_input_path, run_name, output_path, size, return_tif, return_poly, resize_factor):
    # load all scenes from run which we want to merge
    all_runs = utils.get_dirs_in_dir(input_path)
    run_to_merge = [x for x in all_runs if x == run_name]

    # if run to merge is not specified, merge all runs (seperately)
    if not run_to_merge:
        run_to_merge = all_runs
        
    # loop over all runs to merge them
    for run in run_to_merge:
        # get all scenes from this run and build output dir for this run
        all_scenes = utils.get_dirs_in_dir(os.path.join(input_path, run))
        utils.ensure_directory_existance(os.path.join(output_path, run))
        
        for i in tqdm(range(len(all_scenes))):
            print('Processing scene ', all_scenes[i])

            # get all tiles from this scene 
            all_tiles = os.listdir(os.path.join(input_path, run, all_scenes[i]))
                
            # get dimensions of entire scene and create empty scene
            x_max = max([int(s.split('_')[0]) for s in all_tiles])
            y_max = max([int(s.split('.')[0].split('_')[1]) for s in all_tiles])
            predicted_scene = np.zeros((x_max+size, y_max+size), dtype=np.int32)
            
            for tile in all_tiles:
                # read in tile
                tile_array = np.asarray(Image.open(os.path.join(input_path, run, all_scenes[i], tile)))
                [x, y] = os.path.splitext(tile)[0].split('_')
                predicted_scene[int(x):(int(x)+size), int(y):(int(y)+size)] = tile_array
                
            # if we want to return a tif file, read in the input tif from input_scenes to get the crs
            if return_tif:
                # get input tif paths
                all_input_tif_paths = utils.get_list_of_files(tif_input_path), '.tif')
                tif_path_list = [x for x in all_input_tif_paths if os.path.splitext(os.path.basename(x))[0] == all_scenes[i]]

                if len(tif_path_list) == 0:
                    print('the input tif scene to obtain the crs cannot be found of this scene:', all_scenes[i])
                    break
                else:
                    if len(tif_path_list) > 1:
                        print('The scene', all_scenes[i], 'is found', len(tif_path_list), 'times in the input folder. Taking the crs from the first file')

                    # Enlarge predicted image back to original maxar size (takes a long time) and divide predictions by 255 to reduce output size
                    predicted_scene = (predicted_scene/255).astype(np.uint8)
                    enlarged_pred = np.repeat(np.repeat(predicted_scene, resize_factor, axis=0), resize_factor, axis=1) 

                    open_tif = rio.open(tif_path_list[0])

                    with rio.open(os.path.join(output_path, run, all_scenes[i] + '.tif'), 'w', driver='GTiff', height=open_tif.height, 
                        width=open_tif.width, count=1, crs=open_tif.crs, dtype='uint8', transform=open_tif.transform) as dst:
                        dst.write(enlarged_pred[np.newaxis, 0:open_tif.height, 0:open_tif.width])

                    open_tif.close
            
            elif return_poly:
                # Enlarge predicted image back to original maxar size (takes a long time) and divide predictions by 255 to reduce output size
                predicted_scene = (predicted_scene/255).astype(np.uint8)
                enlarged_pred = np.repeat(np.repeat(predicted_scene, resize_factor, axis=0), resize_factor, axis=1) 

                #  create output datasource
                dst_layername = "POLYGONIZED_STUFF"
                drv = ogr.GetDriverByName("ESRI Shapefile")
                dst_ds = drv.CreateDataSource(dst_layername + ".shp")
                dst_layer = dst_ds.CreateLayer(dst_layername, srs = None)

                gdal.Polygonize(enlarged_pred, None, dst_layer, -1, [], callback=None)

            else:
                predicted_scene_out = Image.fromarray(predicted_scene)
                predicted_scene_out.save(os.path.join(output_path, run, all_scenes[i] + '.png'))
                
                
                

    return 
            




if __name__ == "__main__":
    print("Postprocessing") 
    args = configuration()

    input_path = args['input_path']
    run_name = args['run_name']
    output_path = args['output_path']
    return_tif != args['return_png']
    resize_factor = args['resize_factor']
    return_poly = args["return_poly"]
    tif_input_path = args['tif_input_path']

    size = 256

    combine_tiles(input_path, tif_input_path, run_name, output_path, size, return_tif, return_poly, resize_factor)















