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
    def __init__(self, session, feature_layers, feature_weights = None):
        K.set_session(session)
        self.base_model = VGG19(
                include_top = False,
                weights='imagenet')
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
        self.feature_weights = feature_weights
        assert len(self.feature_weights) == len(features)

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


    def make_loss_op(self, x, y):
        """x, y should be rgb tensors in [-1,1]."""
        x = preprocess_input(x)
        x_features = self.model(x)

        y = preprocess_input(y)
        y_features = self.model(y)

        losses = [
                tf.reduce_mean(tf.abs(xf - yf)) for xf, yf in zip(
                    x_features, y_features)]
        for i in range(len(losses)):
            losses[i] = self.feature_weights[i] * losses[i]
        loss = tf.add_n(losses)

        self.losses = losses

        return loss


if __name__ == "__main__":
    import sys
    from tensorflow.contrib.keras.api.keras.preprocessing import (
            image)

    s = tf.Session()

    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 * 2.0 - 1.0
    print(x.shape, np.min(x), np.max(x))
    x = tf.constant(x)

    feature_layers = [
            "input_1", "block1_conv1", "block1_conv2", "block1_pool", "block2_conv2",
            "block3_conv2", "block4_conv2", "block5_conv2"]
    vgg19 = VGG19Features(feature_layers)
    fmaps = vgg19.make_feature_ops(x)

    for i in range(len(fmaps)):
        print(i)
        f = fmaps[i].eval(session=s)
        print(f.shape)
