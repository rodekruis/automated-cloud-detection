# cloud_detection







## Structure
* `scripts` functions to preprocess data, build the model and apply inference
* `weights` contains the optimal trained weights
* `data` input and destination for intermediate results and output



If using Docker
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), to expose the GPUs to the docker container








## Getting started
1. Load data
```
scp -r <path to dir on localhost> user@<host>:data/1_input_scenes folder
```

2. Install [Docker](https://www.docker.com/get-started).

3. Download the [latest Docker Image](https://hub.docker.com/r/rodekruis/automated-building-detection)
```
docker pull rodekruis/cloud-detection
```
4. Create a docker container and connect it to a local directory (`<path-to-your-workspace>`)
```
docker run --name cloud-detection -dit -v <path-to-your-workspace>:/workdir --ipc=host --gpus all -p 5000:5000 rodekruis/cloud-detection
```
5. Access the container
```
docker exec -it cloud-detection bash
```


Firstly, upload the scenes for inference to the data/1_input_scenes folder. The scenes should be either in .tif or .TIF formats and RGB bands should be combined. If the scenes are in a local directory, that can be done by:
```
scp -r <path to dir on localhost> user@<host>:data/1_input_scenes folder
```


## End-to-end inference example
Firstly, preprocess the scenes in data/1_input_scenes folder. Running the following command will tile all scenes and output them in data/2_input_tiles.
```
python scripts/pre_processing.py --inference
```

Secondly, run inference on the tiles with the already trained model. By default the weights are from a model that penalizes false negatives more heavily (UNet/f2_all (using f2-score as loss)) but you could also change that into a more conservative model by adding '--weights-path ./weights/UNet/f1_all_0215v0.hdf5' (f1-loss). You could also use the CloudXNet model, then you should add '--model-type CloudXNet' and include the path to the CloudXNet weights. Depending on the model, a preprocessing step happens before inference on the input tiles (the pre-trained UNet requires the tiles to be preprocessed in the same way as for ImageNet). The default way to run inference on the input tiles is by runnning:


```
python scripts/main.py --inference 
```


Finally, run the postprocessing script to merge all tiles into a scene level prediction. Specify --run-name to be the same as used in the previous inference step, when not specified, the tiles of all runs are merged (seperately per run). Furthermore add --return-tif if you want the output returned as a tif file (takes significantly longer than png). If used, check whether the input tif scene used in preprocessing is in the right directory (see --return-tif help) and use the same resize-factor as in preprocessing (by default 100, as the default in preprocessing).
```
python scripts/post_processing.py --return-tif 
```




## End-to-end training example


For training, preprocessing can be skipped and you can directly download the already preprocessed tiles in input/train_preprocessed using this link. If you would like to train the model on an other dataset then biome, 

If you wish to train the model, omit the --inference argument. The model resumes from weigths given by --weights-path, if you don't want that, train the model from scratch (but still with pre-trained vgg weights in case of UNet) by adding --scratch.
