from __future__ import division
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    net['conv5_3']=build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32),name='vgg_conv5_3')
    net['conv5_4']=build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34),name='vgg_conv5_4')
    net['pool5']=build_net('pool',net['conv5_4'])
    return net

def compute_error(real,fake):
    return tf.reduce_mean(
            tf.abs(fake - real),
            axis = [1,2,3])


class VGG19Features(object):

    def make_loss_op(self, x, y):
        """x, y should be rgb tensors in [-1,1]."""
        x = (x + 1.0) / 2.0 * 255.0
        y = (y + 1.0) / 2.0 * 255.0

        with tf.variable_scope("pretrained_vgg"):
            vgg_real=build_vgg19(x)
            vgg_fake=build_vgg19(y,reuse=True)
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pretrained_vgg')

        p0=compute_error(vgg_real['input'],vgg_fake['input'])
        p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'])
        p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'])
        p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'])
        p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'])
        p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10
        content_loss=p0+p1+p2+p3+p4+p5
        G_loss = tf.reduce_mean(content_loss,reduction_indices=0)

        self.losses = [
                tf.reduce_mean(p0),
                tf.reduce_mean(p1),
                tf.reduce_mean(p2),
                tf.reduce_mean(p3),
                tf.reduce_mean(p4),
                tf.reduce_mean(p5)]

        return G_loss
