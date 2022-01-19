import random
import numpy as np
import skimage.io
from tensorflow.python.keras.utils.data_utils import Sequence
# import imgaug.augmenters as iaa
# import imgaug as ia

from PIL import Image
# import utils


# ia.seed(0)

class DataGenerator(Sequence):
    """Generates data for Keras

    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, images, annotations, preprocess_input, to_fit=True,
                 batch_size=32,width=256, height=256, n_channels=3, shuffle=True,
                 augmentation=False, dtype=np.float32):
        """Initialization

        Make sure that the provided class indices start at 1, because 0 is used
        for background predictions.

        :param data_dir: root directory of the images.
        :param data: List that contains both polygons and paths to the images
        :param classes: Dict that contains all the classes and their mask-indices
        :param preprocess_input: function for preprocessing image input
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param width: int indicating the input width of the model
        :param height: int indicating the input height of the model
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        :param augmentation: True to augment the images, False no augmentation
        :param dtype: data type of the generated batches
        """
        self.images = images
        self.annotations = annotations

        # # class variables
        # self.n_classes = len(classes)
        # self.class_codes = {}
        # for c_indx in classes.values():
        #     code = np.zeros(self.n_classes, dtype=dtype)
        #     code[c_indx] = 1
        #     self.class_codes[c_indx] = code

        self.preprocess_input = preprocess_input
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.dtype = dtype
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.images) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        if self.to_fit:
            return self._generate_X_y(indexes)
        else:
            return self._generate_X(indexes)


    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def _generate_X(self, indexes):
        """Generates data containing batch_size images
        :param indexes: list of ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels))

        # Generate data
        for i, im_id in enumerate(indexes):
            # Store sample
            X[i,] = self._load_image(self.images[im_id])
        X = self.preprocess_input(X)
        return X

    def _generate_X_y(self, indexes):
        """Generates data containing batch_size masks
        :param indexes: list of ids to load
        :return: batch of images and masks
        """
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels), dtype=self.dtype)
        _y = np.empty((self.batch_size, self.height, self.width, 1), dtype=self.dtype)

        # Generate data
        for i, im_id in enumerate(indexes):
            
            # Store data
            X[i,] = self._load_image(self.images[im_id])
            _y[i,] = self._load_annotation(self.annotations[im_id])

        if self.augmentation:
            X, _y = self._augment_batch(X, _y)

        # y = self._one_hot_encode(_y)
        X = self.preprocess_input(X)

        return X, _y


    def _augment_batch(self, X, _y):
        """Apply same augmentation on X and _y.
        :param X: images
        :param _y: segmentation masks
        """
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop((1, 150), keep_size=True),
            iaa.Affine(rotate=(-30, 30),shear=(-12, 12)),
            iaa.GammaContrast((0.9, 1.2)),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-20, 20)),
            # iaa.ElasticTransformation(alpha=2, sigma=1),
            iaa.AdditiveGaussianNoise(scale=(0,0.01*255))
        ])
        return seq(images=X, segmentation_maps=_y)


    # def _one_hot_encode(self, _y):
    #     y = np.empty((self.batch_size, self.height, self.width, self.n_classes), dtype=self.dtype)
    #     for i, anno in enumerate(_y):
    #         y[i,] = utils.one_hot_encode(anno.reshape((self.height, self.width)), self.class_codes, dtype=self.dtype)
    #     return y


    def _load_image(self, image_path):
        """Load image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = Image.open(image_path)
        return np.array(img, dtype=self.dtype)

    def _load_annotation(self, anno_path):
        """Load annotation
        :param anno_path: path to annotation to load
        :return: loaded annotation
        """
        an = Image.open(anno_path)
        anno = (np.array(an)/255).astype(np.uint8)
        return anno.reshape((anno.shape[0], anno.shape[1], 1)).astype(self.dtype)











