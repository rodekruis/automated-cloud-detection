import argparse
import os
import time
import re




def run_name_type(run_name):
    run_name = str(run_name)
    pattern = re.compile(r"^[a-zA-Z0-9_\.]{3,30}$")
    if not pattern.match(run_name):
        raise argparse.ArgumentTypeError(
            "Run name can contain only "
            + "alphanumeric, underscore (_) and dot (.) characters. "
            + "Must be at least 3 characters and at most 30 characters long."
        )
    return run_name

def make_directory(directoryPath):
    if not os.path.isdir(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath


def configuration():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join("/workdir", "runs"),
        help="save training output here",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("/workdir", "data", "2_input_tiles", "inference"),
        help="data path for train or inference folder",
    )
    parser.add_argument(
        "--prediction-path",
        type=str,
        default=os.path.join("/workdir", "data", "3_prediction_tiles"),
        help="path to the predictions made on the input tiles, in case of inference",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=os.path.join("/workdir", "weights", "UNet", "f2_all_0218v2.hdf5"),
        help="path to weights to resume from",
    )
    parser.add_argument(
        "--run-name",
        type=run_name_type,
        default="run_1",
        help="name to identify execution",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        help="model architecture to use, either UNet or CloudXNet, choose weight-path corresponding",
    )
    parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=30,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.5,
        help="threshold used to map probability predictions to integers",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="learning rate for training"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        default=False,
        help="test and use the model on the inference set instead of training",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        default=False,
        help="train model without continuing from loaded weights",
    )
    args = parser.parse_args()

    arg_vars = vars(args)
    arg_vars["model_name"] = arg_vars["run_name"]


    arg_vars["checkpoint_path"] = make_directory(
        os.path.join(arg_vars["checkpoint_path"], arg_vars["run_name"])
    )

    return arg_vars  #previously returned args