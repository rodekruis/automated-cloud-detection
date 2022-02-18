

import tensorflow as tf
from tensorflow.keras import backend as K


smooth = 0.0000001


def jacc_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    jacc = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    return jacc


def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersection + smooth)/(union + smooth)
    return dice

def fbeta_score(y_true, y_pred, beta=2., epsilon=K.epsilon()):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f * y_pred_f)
    predicted_positive = K.sum(y_pred_f)
    actual_positive = K.sum(y_true_f)
    print(predicted_positive)
    precision = tp/(predicted_positive+epsilon) # calculating precision
    recall = tp/(actual_positive+epsilon) # calculating recall
    
    # calculating fbeta
    beta_squared = K.square(beta)
    fb = (1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon)

    return fb

def fbeta_loss(y_true, y_pred, beta=2., epsilon=K.epsilon()):

    return 1 - fbeta_score(y_true, y_pred, beta=beta, epsilon=epsilon)


# def fbeta_loss(ytrue , ypred, beta=2, threshold=0.5, epsilon=K.epsilon()):
#     # epsilon is set so as to avoid division by zero error
    
#     beta_squared = beta**2 # squaring beta

#     # casting ytrue and ypred as float dtype
#     ytrue = tf.cast(ytrue, tf.float32)
#     ypred = tf.cast(ypred, tf.float32)

#     # setting values of ypred greater than the set threshold to 1 while those lesser to 0
#     ypred = tf.cast(tf.greater_equal(ypred, tf.constant(threshold)), tf.float32)

#     tp = tf.reduce_sum(ytrue*ypred) # calculating true positives
#     predicted_positive = tf.reduce_sum(ypred) # calculating predicted positives
#     actual_positive = tf.reduce_sum(ytrue) # calculating actual positives
    
#     precision = tp/(predicted_positive+epsilon) # calculating precision
#     recall = tp/(actual_positive+epsilon) # calculating recall
    
#     # calculating fbeta
#     fb = (1+beta_squared)*precision*recall / (beta_squared*precision + recall + epsilon)

#     return 1 - fb