# cloud_detection







## Structure
* `scripts` functions to preprocess data, build the model and apply inference
* `weights` contains the optimal trained weights
* `data` input and destination for intermediate results and output



If using Docker
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), to expose the GPUs to the docker container








## Getting started
### Using Docker
1. Install [Docker](https://www.docker.com/get-started).
2. Download the [latest Docker Image](https://hub.docker.com/r/rodekruis/automated-building-detection)
```
docker pull rodekruis/automated-building-detection
```
3. Create a docker container and connect it to a local directory (`<path-to-your-workspace>`)
```
docker run --name automated-building-detection -dit -v <path-to-your-workspace>:/workdir --ipc=host --gpus all -p 5000:5000 rodekruis/automated-building-detection
```
4. Access the container
```
docker exec -it automated-building-detection bash
```



## End-to-end inference example
Firstly, preprocess the scenes. The scenes should be either in .tif or .TIF formats and uploaded to the input/inference_scene directory. Running the following command will preprocess all scenes in the folder and output them in inference_preprocessed.
```
python scripts/pre_processing.py --inference
```

Secondly, run inference on the already trained model. By default the weights are from a model that penalizes false negatives more heavily (UNet/f2_all (using f2-score as loss)) but you could also change that into a more conservative model by adding '--weights-path ./weights/UNet/f1_all_0215v0.hdf5'. You could also use the CloudXNet model, then you should add '--model-type CloudXNet' and also include the path to the CloudXNet weights. The most easy way to run inference on the input tiles is by runnning:


```
python scripts/main.py --inference 
```


Thirdly, run the postprocessing script to merge all tiles into a scene level prediction. Specify --run-name to be the same as used in the previous inference step, when not specified, the tiles of all runs are merged (seperately per run). Furthermore add --return-tif if you want the output returned as a tif file (takes significantly longer than png). If used, check whether the input tif scene used in preprocessing is in the right directory (see --return-tif help) and use the same resize-factor as in preprocessing (only need to specify it, when you changed the value in preprocessing).
```
python scripts/post_processing.py --return-tif --run-name test_name 
```







For training, preprocessing can be skipped and you can directly download the already preprocessed tiles in input/train_preprocessed using this link. If you would like to train the model on an other dataset then biome, 

If you wish to train the model, omit the --inference argument. The model resumes from weigths given by --weights-path, if you don't want that, train the model from scratch (but still with pre-trained vgg weights in case of UNet) by adding --scratch.
