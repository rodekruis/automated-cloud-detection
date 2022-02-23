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
        default=os.path.join(".", "runs"),
        help="output path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(".", "input", "inference_preprocessed"),
        help="data path for train or inference folder",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=os.path.join(".", "weights", "UNet", "f2_all_0218v2.hdf5"),
        help="path to weights to resume from",
    )
    parser.add_argument(
        "--run-name",
        type=run_name_type,
        default="{:.0f}".format(time.time()),
        help="name to identify execution",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        help="model architecture to use, either UNet or CloudXNet, choose weight-path corresponding",
    )
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=100,
        help="factor for resizing inference images to match resolution, model trained on res:30m, maxar res:30cm",
    )
    parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=30,
        help="number of epochs for training",
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


    arg_vars[
        "model_directory"
    ] = "{}-nr_epochs_{}".format(
        arg_vars["run_name"],
        arg_vars["number_of_epochs"],
    )

    arg_vars["checkpoint_path"] = make_directory(
        os.path.join(arg_vars["checkpoint_path"], arg_vars["model_directory"])
    )
    arg_vars["trained_model_path"] = os.path.join(
        arg_vars["checkpoint_path"], "best_model_wts.pkl"
    )
    arg_vars["prediction_path"] = make_directory(
        os.path.join(arg_vars["checkpoint_path"], "predictions")
    )
    arg_vars["run_report_path"] = os.path.join(
        arg_vars["checkpoint_path"], "run_report.json"
    )


    return arg_vars  #previously returned args