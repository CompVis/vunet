import tensorflow as tf
import numpy as np
# vgg19 from keras
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.applications.vgg19 import (
        VGG19)
from tensorflow.contrib.keras.api.keras.preprocessing import (
        image)


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
    def __init__(self, feature_layers):
        with tf.variable_scope("pretrained_VGG19"):
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
                tf.reduce_mean(tf.square(xf - yf)) for xf, yf in zip(
                    x_features, y_features)]
        loss = tf.add_n(losses) / len(losses)

        return loss



if __name__ == "__main__":
    import sys
    import tensorflow as tf

    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 * 2.0 - 1.0
    print(x.shape, np.min(x), np.max(x))

    feature_layers = ["input_1", "block1_conv2", "block2_conv2",
            "block3_conv2", "block4_conv2", "block5_conv2"]
    vgg19 = VGG19Features(feature_layers)
    fmaps = vgg19.extract_features(x)
    for fmap in fmaps:
        print(fmap.shape)

    x = tf.constant(x)
    fmaps = vgg19.make_feature_ops(x)
    print(fmaps)

    loss = vgg19.make_loss_op(x, x)
    print(loss)

    print(tf.trainable_variables())
    print("=====================")
    print(vgg19.variables)
