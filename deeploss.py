import tensorflow as tf
import numpy as np
# vgg19 from keras
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.applications.vgg19 import VGG19
from tensorflow.contrib.keras.api.keras import backend as K


def preprocess_input(x):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input tensor, 4D in [-1,1]
    # Returns
        Preprocessed tensor.
    """
    # from [-1, 1] to [0,255.0]
    x = (x + 1.0) / 2.0 * 255.0
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    x = x - np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))
    return x


class VGG19Features(object):
    def __init__(self, session,
            feature_layers = None, feature_weights = None, gram_weights = None):
        K.set_session(session)
        self.base_model = VGG19(
                include_top = False,
                weights='imagenet')
        if feature_layers is None:
            feature_layers = [
                    "input_1",
                    "block1_conv2", "block2_conv2",
                    "block3_conv2", "block4_conv2",
                    "block5_conv2"]
        self.layer_names = [l.name for l in self.base_model.layers]
        for k in feature_layers:
            if not k in self.layer_names:
                raise KeyError(
                        "Invalid layer {}. Available layers: {}".format(
                            k, self.layer_names))
        features = [self.base_model.get_layer(k).output for k in feature_layers]
        self.model = Model(
                inputs = self.base_model.input,
                outputs = features)
        if feature_weights is None:
            feature_weights = len(feature_layers) * [1.0]
        if gram_weights is None:
            gram_weights = len(feature_layers) * [0.1]
        self.feature_weights = feature_weights
        self.gram_weights = gram_weights
        assert len(self.feature_weights) == len(features)
        self.use_gram = np.max(self.gram_weights) > 0.0

        self.variables = self.base_model.weights


    def extract_features(self, x):
        """x should be rgb in [-1,1]."""
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features


    def make_feature_ops(self, x):
        """x should be rgb tensor in [-1,1]."""
        x = preprocess_input(x)
        features = self.model(x)
        return features


    def grams(self, fs):
        gs = list()
        for f in fs:
            bs, h, w, c = f.shape.as_list()
            f = tf.reshape(f, [bs, h*w, c])
            ft = tf.transpose(f, [0,2,1])
            g = tf.matmul(ft, f)
            g = g / (4.0*h*w)
            gs.append(g)
        return gs


    def make_loss_op(self, x, y):
        """x, y should be rgb tensors in [-1,1]."""
        x = preprocess_input(x)
        x_features = self.model(x)

        y = preprocess_input(y)
        y_features = self.model(y)

        x_grams = self.grams(x_features)
        y_grams = self.grams(y_features)

        losses = [
                tf.reduce_mean(tf.abs(xf - yf)) for xf, yf in zip(
                    x_features, y_features)]
        gram_losses = [
                tf.reduce_mean(tf.abs(xg - yg)) for xg, yg in zip(
                    x_grams, y_grams)]

        for i in range(len(losses)):
            losses[i] = self.feature_weights[i] * losses[i]
            gram_losses[i] = self.gram_weights[i] * gram_losses[i]
        loss = tf.add_n(losses)
        if self.use_gram:
            loss = loss + tf.add_n(gram_losses)

        self.losses = losses
        self.gram_losses = gram_losses

        return loss
