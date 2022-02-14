import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *

import utils


# ------------------------------------------------------------------------------
#  Blocks
# ------------------------------------------------------------------------------

def Conv3x3(x, filters, use_batchnorm, activation='relu', name='Conv3x3', bn_axis=3):
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        padding='same',
        activation=None,
        use_bias=not (use_batchnorm),
        kernel_initializer='he_uniform',
        name=name,
        data_format=DATA_FORMAT
    )(x)
    if use_batchnorm:
        x = BatchNormalization(axis=bn_axis, name=name + '_bn')(x)
    if activation:
        x = Activation(activation, name=name + '_act')(x)
    return x


def DecoderUpsamplingBlock(x, skip, filters, use_batchnorm, activation='relu',
                           name='decoder_block', concat_axis=3):
    up_name = name + '_up'
    concat_name = name + '_concat'
    conv1_name = name + '_conv1'
    conv2_name = name + '_conv2'

    x = UpSampling2D(size=(2,2), name=up_name, data_format=DATA_FORMAT)(x)

    if skip is not None:
        x = concatenate([x, skip], axis=concat_axis, name=concat_name)

    x = Conv3x3(x, filters, use_batchnorm, activation='relu', name=conv1_name, bn_axis=concat_axis)
    x = Conv3x3(x, filters, use_batchnorm, activation='relu', name=conv2_name, bn_axis=concat_axis)

    return x




DATA_FORMAT = 'channels_last'

# The layers that are used for the skip connections in Unet
SKIP_LAYERS = {
    'VGG16': ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'],
    'VGG19': ['block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'],
    'ResNet50': [142, 80, 38, 4]    # For some reason cannot load the ResNet50 backbone twice, unless using indices.
}

def get_backbone(backbone_name, input_shape, weights='imagenet', include_top=False, data_format='channels_last'):
    """Fetch a state-of-the-art backbone and the used image preprocessing function.

        Args:
            backbone_name (str): One of the following: VGG16, VGG19, ResNet50
            input_shape (tuple): Shape of model input (int, int, int)
            weights (str, optional): Either None or imagenet is supported
            include_top (bool, optional): Whether to include the top of the model.

        Returns:
            backbone (Model): backbone model
            preproces_func (function): image preprocessing function used to train the
                backbone with.
    """
    assert backbone_name in SKIP_LAYERS.keys(), "Unsupported backbone: {}".format(backbone_name)

    backbone, preproces_func = None, None
    if backbone_name == 'VGG16':
        backbone = VGG16(include_top=include_top, weights=weights, input_shape=input_shape)
        preprocess_func = vgg16.preprocess_input
    elif backbone_name == 'VGG19':
        backbone = VGG19(include_top=include_top, weights=weights, input_shape=input_shape)
        preprocess_func = vgg19.preprocess_input
    elif backbone_name == 'ResNet50':
        backbone = ResNet50(include_top=include_top, weights=weights, input_shape=input_shape)
        preprocess_func = resnet50.preprocess_input

    return backbone, preprocess_func








# ------------------------------------------------------------------------------
#  Model
# ------------------------------------------------------------------------------

def build_unet(backbone_name = 'VGG16', input_size=(256,256,3),
                     freeze_backbone=False, regularization=True, use_batchnorm=True):
    backbone, preprocess_func = get_backbone(backbone_name, input_size, data_format=DATA_FORMAT)


    # Freeze all layers, or certain blocks, works only for VGG16
    if freeze_backbone == True:
        for l in backbone.layers:
             l.trainable = False



    skips = [backbone.get_layer(name=i).output if isinstance(i, str)
        else backbone.get_layer(index=i).output for i in SKIP_LAYERS[backbone_name]]

    inputs = backbone.input
    x = backbone.output

    # Center
    x = Conv3x3(x, 512, use_batchnorm, activation = 'relu', name='center_block1_conv1')
    x = Conv3x3(x, 512, use_batchnorm, activation = 'relu', name='center_block1_conv2')

    # Decoder blocks
    for i, filters in enumerate([256,128,64,32,16]):
        skip = None
        if i < len(skips):
            skip = skips[i]
    
        x = DecoderUpsamplingBlock(x, skip, filters, use_batchnorm, activation='relu', name='decoder_block{}'.format(i+1))



    # Binary Clouds Head
    x = Conv3x3(x, 32, use_batchnorm, activation='relu', name='clouds_block1_conv1')
    x = Conv3x3(x, 16, use_batchnorm, activation = 'relu', name='clouds_block1_conv2')
    x = Conv2D(1, 1, activation = 'sigmoid', name='clouds', data_format=DATA_FORMAT)(x)
    model = Model(inputs = inputs, outputs = x)
    
    
    if regularization:
        for l in model.layers:
            l.kernel_regularizer = tf.keras.regularizers.l2(l=0.1)

    model.model_name = "unet_vgg16"
    model.output_width = input_size[0]
    model.output_height = input_size[1]
    model.input_width = input_size[0]
    model.input_height = input_size[1]

    return model, preprocess_func






