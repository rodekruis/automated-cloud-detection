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



## End-to-end example







If you wish to train the model, omit the --inference argument. The model resumes from weigths given by --weights-path, if you don't want that, train the model from scratch (but still with pre-trained vgg weights in case of UNet) by adding --scratch.
