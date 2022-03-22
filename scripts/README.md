# Config options
For each pre_processing, main and post_processing script seperate config options are available. In this readme a clarification is given for these config options. The default options can be found in each script.

## pre_processing
`--data-path` remote path to the data to train on, probably should start with /workdir. Preferably don't change the data locations. But should only be changed if inference data is not uploaded to the default /workdir/data/1_input_scenes/inference, or in case of training a new dataset (not Biome) and the data is not uploaded to /workdir/data/1_input_scenes/train.

`--out-dir` path to output directory. Should not be changed.

`--inference` add in case of inference.

`--norm-percentile` gives the percentile to normalize with. The Maxar images and the uploaded Biome dataset are already normalized so for them no extra normalization is needed. However, when training on a new dataset or inference with non Maxar images that need normalization it should be specified. Use a value of 2 for light normalization, 5 for medium and 10 for heavy. Since raw Biome data is significantly darker then Maxar data, a value of 10 is used previously.

`--resize-factor` this factor is used to downsize Maxar data, so in case of inference. The resolution for the Biome training data is 30m, while the Maxar data has a resolution of 30cm, so to match this the Maxar data is resized by factor 100. So when using a new training dataset this value should be changed according to the new resolution.


## main config
`checkpoint-path` the model weights and tensorboard output is saved here, so only applies for training. The UNet model is run 2 times, run_1 is with a frozen ImageNet backbone, to train the randomly initialized upsampling part of the UNet. After 3 epochs, when the upsampling weights are also a bit better, the whole model is trained in run_2. Only the best weights (lowest validation loss) are saved. So the last weights in run_2 are the best for the model.

`data-path` the path to the input tiles. Should not have to be changed. For inference /workdir/data/1_input_scenes/inference is the default location and for training, /workdir/data/1_input_scenes/train is default.

`prediction-path` this path leads to the directory where the predictions on the input tiles should be stored in case of inference. 

`weights-path` the path to the weights used to continue from. By default the weights are from a model that penalizes false negatives more heavily (UNet/f2_all (using f2-score as loss)) but you could also change that into a more conservative model by changing the path to /workdir/weights/UNet/f1_all_0215v0.hdf5 (f1-loss). You could also use the CloudXNet model, then you should add '--model-type CloudXNet' and include the path to the CloudXNet weights. 

`run-name` the name of the run to identify the execution. 

`model-type` the type of model to use. Either an UNet (default) or CloudXNet model. If you would like to use the CloudXNet architecture, you should also change the weights-path. When the UNet model is used, an extra preprocessing happens on the input tiles (the pre-trained UNet requires the tiles to be preprocessed in the same way as for ImageNet).

`pred-threshold` threshold used to map probability predictions to classification.

`inference` add --inference to apply inference, omit it to train the model.

`scratch` add --scratch to train the model from scratch, when omitted the model resumes for weights defined by `weights-path`. When the model used is the UNet model, the model will not train entirely from scratch, but will start with a pre-trained VGG16 backbone on ImageNet.


`number-of-epochs`, `batch-size` and `learning-rate` are also adjustable.



## post_processing












