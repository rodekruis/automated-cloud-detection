# Config options

For each pre_processing, main and post_processing script seperate config options are available. In this readme a clarification is given for these config options. The default options can be found in each script.

## pre_processing

`--data-path` remote path to the data to train on, probably should start with /workdir. Should only be changed if inference data is not uploaded to the default data/1_input_scenes/inference, or in case of training you want to train a on new dataset (not Biome).

`--out-dir` path to output directory. Only change in case of training on a new dataset, preferably to /workdir/data/2_input_tiles/train .

`--inference` add in case of inference.

`--norm-percentile` gives the percentile to normalize with. The Maxar images and the uploaded Biome dataset are already normalized so for them no extra normalization is needed. However, when training on a new dataset or inference with images that need normalization it should be specified. Use a value of 2 for light normalization, 5 for medium and 10 for heavy. Since raw Biome data is significantly darker then Maxar data, a value of 10 is used previously.

`--resize-factor` This factor is used to downsize Maxar data, so in case of inference. The resolution for the Biome training data is 30m, while the Maxar data has a resolution of 30cm, so to match this the Maxar data is resized by factor 100. So when using a new training dataset this value should be changed according to the new resolution.












