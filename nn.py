"""
modified from pixelcnn++
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def int_shape(x):
    return x.shape.as_list()


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def ce_loss(x, l):
    x = (x + 1.0) / 2.0
    x = tf.clip_by_value(255 * x, 0, 255)
    x = tf.cast(x, tf.int32)
    reconst_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(l, [-1, 256]),
                labels=tf.reshape(x, [-1])
                )
            )
    return reconst_cost


def ce_sample(logits, temp = 1.0):
    temp = tf.maximum(tf.convert_to_tensor(1e-5), tf.convert_to_tensor(temp))
    noise = tf.random_uniform(logits.shape, minval = 1e-5, maxval = 1.0 - 1e-5)
    pixels = tf.argmax(logits / temp - tf.log(-tf.log(noise)), 4)
    pixels = tf.cast(pixels, tf.float32) / 127.5 - 1.0
    return pixels


def discretized_mix_logistic_loss(x, l, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(
        x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -2.)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                    * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                    coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = tf.concat([tf.reshape(means[:, :, :, 0, :], [
                      xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = -mid_in - log_scales - 2. * tf.nn.softplus(-mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
                                                            tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    return -tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs), [1, 2]))


def sample_from_discretized_mix_logistic(l, nr_mix, temp1 = 1.0, temp2 = 1.0, mean = False):
    if mean:
        temp2 = 0.0
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if not mean:
        if temp1 < 1e-5:
            sel = tf.one_hot(tf.argmax(logit_probs, 3), depth=nr_mix, dtype=tf.float32)
        else:
            sel = tf.one_hot(tf.argmax(logit_probs/temp1 - tf.log(-tf.log(tf.random_uniform(
                logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    else:
        sel = tf.nn.softmax(logit_probs)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, 4), -2.)
    coeffs = tf.reduce_sum(tf.nn.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + temp2 * tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)


''' layers containing trainable variables. '''

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def dense(x, num_units, init_scale=1., counters={}, init=False, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if init:
            xs = x.shape.as_list()
            # data based initialization of parameters
            V = tf.get_variable('V', [xs[1], num_units], tf.float32, tf.random_normal_initializer(0, 0.05))
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init)
            x_init = tf.reshape(scale_init, [1, num_units]) * (x_init - tf.reshape(m_init, [1, num_units]))

            return x_init
        else:
            V = tf.get_variable("V")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            with tf.control_dependencies([tf.assert_variables_initialized([V, g, b])]):
                # use weight normalization (Salimans & Kingma, 2016)
                x = tf.matmul(x, V)
                scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

                return x


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, **kwargs):
    ''' convolutional layer '''
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        if init:
            xs = x.shape.as_list()
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [xs[-1], num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.05))
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, strides, pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer = scale_init)
            b = tf.get_variable('b', dtype=tf.float32, initializer = -m_init * scale_init)
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))

            return x_init
        else:
            V = tf.get_variable("V")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            with tf.control_dependencies([tf.assert_variables_initialized([V, g, b])]):
                # use weight normalization (Salimans & Kingma, 2016)
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

                # calculate convolutional layer output
                x = tf.nn.bias_add(tf.nn.conv2d(x, W, strides, pad), b)

                return x


@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, **kwargs):
    ''' transposed convolutional layer '''
    num_filters = int(num_filters)
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    strides = [1] + stride + [1]
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] -
                        1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [num_filters, xs[-1]], tf.float32, tf.random_normal_initializer(0, 0.05))
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, V_norm, target_shape, strides, padding=pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init)
            x_init = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))

            return x_init
        else:
            V = tf.get_variable("V")
            g = tf.get_variable("g")
            b = tf.get_variable("b")
            with tf.control_dependencies([tf.assert_variables_initialized([V, g, b])]):
                # use weight normalization (Salimans & Kingma, 2016)
                W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

                # calculate convolutional layer output
                x = tf.nn.conv2d_transpose(x, W, target_shape, strides, padding=pad)
                x = tf.nn.bias_add(x, b)

                return x


@add_arg_scope
def activate(x, activation, **kwargs):
    if activation == None:
        return x
    elif activation == "elu":
        return tf.nn.elu(x)
    else:
        raise NotImplemented(activation)


''' meta-layer consisting of multiple base layers '''

def nin(x, num_units):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units)
    return tf.reshape(x, s[:-1] + [num_units])


def downsample(x, num_units):
    return conv2d(x, num_units, stride = [2, 2])


def upsample(x, num_units, method = "subpixel"):
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride = [2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4*num_units)
        x = tf.depth_to_space(x, 2)
        return x


@add_arg_scope
def residual_block(x, a = None, conv=conv2d, init=False, dropout_p=0.0, gated = False, **kwargs):
    """Slight variation of original."""
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if a is not None:
        a = nin(activate(a), num_filters)
        residual = tf.concat([residual, a], axis = -1)
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob = 1.0 - dropout_p)
    residual = conv(residual, num_filters)
    if gated:
        residual = activate(residual)
        residual = tf.nn.dropout(residual, keep_prob = 1.0 - dropout_p)
        residual = conv(residual, 2*num_filters)
        a, b = tf.split(residual, 2, 3)
        residual = a * tf.nn.sigmoid(b)

    return x + residual


''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)


def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)


@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size,
                 pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [filter_size[1] - 1, 0], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size,
                 pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]


################################################################ random tf stuff


def make_linear_var(
        step,
        start, end,
        start_value, end_value,
        clip_min = 0.0, clip_max = 1.0):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha)/(b - a) * (x - a) + alpha"""
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)

"""Simple approximation of 2d gaussian kernel."""
k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
# normalize and extend to three independent input and output channels
kernel = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
def tf_gaussian_subsample(x):
    return tf.nn.conv2d(
            input = x,
            filter = kernel,
            strides = [1, 2, 2, 1],
            padding = "SAME")


# stride 2 subsampling
nnkernel = np.eye(3)[None,None,:,:]
def tf_subsample(x):
    return tf.nn.conv2d(
            input = x,
            filter = nnkernel,
            strides = [1, 2, 2, 1],
            padding = "SAME")


# replace
tf_downsample = tf_gaussian_subsample


def tf_pyramid(x, ps, p2 = None):
    """Pyramid of x, coarse to fine"""
    nd = round(math.log(ps, 2))
    assert 2**nd == ps
    xs = x.get_shape().as_list()
    assert len(xs) == 4
    b, h, w, c = xs
    assert h == w
    if p2 is None:
        p2 = round(math.log(h,ps))
        assert ps**p2 == h, "{}, {}, {}".format(ps, p2, h)
    pyramid = [x]
    for i in range(p2):
        p = pyramid[-1]
        for j in range(nd):
            p = tf_downsample(p)
        pyramid.append(p)
    return list(reversed(pyramid))


def tf_concat_coarse_cond(coarse, cond):
    groups = tf.concat([coarse] + 3* [tf.zeros_like(coarse)], axis = 3)
    coarse_up = tf.depth_to_space(groups, 2)
    return tf.concat([coarse_up, cond], axis = 3)


def np_concat_coarse_cond(coarse, cond):
    groups = np.concatenate([coarse] + 3* [np.zeros_like(coarse)], axis = 3)
    coarse_up = np_depth_to_space(groups, 2)
    return np.concatenate([coarse_up, cond], axis = 3)


smoothing1d = np.float32([1,2,1])
difference1d = np.float32([1,0,-1])
sobelx = np.outer(smoothing1d, difference1d)
sobely = np.transpose(sobelx)
# one dim for number of input channels
sobelx = sobelx[:,:,None]
sobely = sobely[:,:,None]
# stack along new dim for output channels
sobel = np.stack([sobelx, sobely], axis = -1)

fdx = np.zeros([3,3])
fdx[1,:] = difference1d
fdx = fdx[:,:,None]

fdy = np.zeros([3,3])
fdy[:,1] = difference1d
fdy = fdy[:,:,None]
fd = np.stack([fdx, fdy], axis = -1)
def tf_img_grad(x, use_sobel = True):
    """Sobel approximation of gradient."""
    gray = tf.reduce_mean(x, axis = -1, keep_dims = True)
    if use_sobel:
        filter_ = sobel
    else:
        filter_ = fd
    grad = tf.nn.conv2d(
            input = gray,
            filter = filter_,
            strides = 4*[1],
            padding = "SAME")
    return grad


def tf_grad_loss(x, y):
    """Mean squared L2 difference of gradients."""
    gx = tf_img_grad(x)
    gy = tf_img_grad(y)
    return tf.reduce_mean(tf.contrib.layers.flatten(tf.square(gx - gy)))


def tf_grad_mag(x):
    """Pointwise L2 norm of gradient."""
    gx = tf_img_grad(x)
    return tf.sqrt(tf.reduce_sum(tf.square(gx), axis = -1, keep_dims = True))


def tv_loss(x):
    h = 1.0 / x.shape.as_list()[1]
    g = tf_img_grad(x, use_sobel = False)
    hgl1 = h * tf.sqrt(
            tf.reduce_sum(
                tf.square(g),
                axis = 3))
    return tf.reduce_mean(
            tf.reduce_sum(
                hgl1,
                axis = [1,2]))



def likelihood_loss(target, tail_decoding, loss):
    if loss == "l2":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(target - tail_decoding)))
    elif loss == "l1":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.abs(target - tail_decoding)))
    elif loss == "h1":
        rec_loss = tf.reduce_mean(tf.contrib.layers.flatten(
            tf.square(target - tail_decoding)))
        rec_loss += tf_grad_loss(target, tail_decoding)
    else:
        raise NotImplemented("Unknown loss function: {}".format(loss))
    return rec_loss


def split_groups(x, bs = 2):
    return tf.split(tf.space_to_depth(x, bs), bs**2, axis = 3)


def merge_groups(xs, bs = 2):
    return tf.depth_to_space(tf.concat(xs, axis = 3), bs)
