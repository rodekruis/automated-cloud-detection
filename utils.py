
import os 
import numpy as np

import glob







def get_list_of_files(dir_path, sorted=True):
    """Get list of filepaths to png files within a directoy.

        Args:
            dir_path (str): path to directory
            sorted (bool, optional): True to sort the output filenames.

        Returns:
            all_files (list): list of all filepaths (str)
    """
    all_files = glob.glob(dir_path + '*.png')
    
    if sorted:
        all_files.sort()

    return all_files



def ensure_directory_existance(dir_path):
    """Check and/or create directory.

        Args:
            dir_path (str): path to directory

        Returns:
            None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)




