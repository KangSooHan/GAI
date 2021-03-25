import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib

def truncated_normal_var(name ,shape, dtype):
    return(tf.get_variable(name = name, shape = shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=1e-2)))

def zero_var(name, shape, dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

def conv_layer( input_images, k_shape,k_size, k_stride, name):
        conv_kernel = truncated_normal_var(name=name+'_W', shape=np.concatenate([k_shape, [input_images.shape[3]], [k_size]]) , dtype=tf.float32)
        conv = tf.nn.conv2d(input_images, conv_kernel, k_stride, padding="SAME")
        conv_bias = zero_var(name=name+'_b', shape=[k_size], dtype=tf.float32)
        conv_add_bias = tf.nn.bias_add(conv, conv_bias)
        relu_conv = tf.nn.relu(conv_add_bias)
        return relu_conv

VGG_MEAN = [103.939, 116.779, 123.68]

class CNN():
    def __init__(self,  input_images):
        batch_size = input_images.shape[0]
        input_images_scaled = input_images * 255.0
        red, green, blue =  tf.split(axis=3, num_or_size_splits=3, value=input_images_scaled)
        bgr = tf.concat(axis=3, values=[blue-VGG_MEAN[0], green-VGG_MEAN[1], red-VGG_MEAN[2],])

        with tf.variable_scope('conv1_1',reuse= tf.AUTO_REUSE) as scope:
            self.conv1_1 = conv_layer(bgr, [3,3], 64, [1,1,1,1], 'conv1_1')

        with tf.variable_scope('conv1_2', reuse= tf.AUTO_REUSE) as scope:
            self.conv1_2 = conv_layer(self.conv1_1, [3,3], 64, [1,1,1,1], 'conv1_2')
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = 'pool_layer1')
        with tf.variable_scope('conv2_1', reuse= tf.AUTO_REUSE) as scope:
            self.conv2_1 = conv_layer(self.pool1, [3,3], 128, [1,1,1,1], 'conv2_1')
        with tf.variable_scope('conv2_2', reuse= tf.AUTO_REUSE) as scope:
            self.conv2_2 = conv_layer(self.conv2_1, [3,3], 128, [1,1,1,1], 'conv2_2')
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name = 'pool_layer2')

        with tf.variable_scope('conv3_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv3_1 = conv_layer(self.pool2, [3,3], 256, [1,1,1,1], 'conv3_1')

        with tf.variable_scope('conv3_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv3_2 = conv_layer(self.conv3_1, [3,3], 256, [1,1,1,1], 'conv3_2')

        with tf.variable_scope('conv3_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv3_3 = conv_layer(self.conv3_2, [3,3], 256, [1,1,1,1], 'conv3_3')
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = 'pool_layer3')

        with tf.variable_scope('conv4_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv4_1 = conv_layer(self.pool3, [3,3], 512, [1,1,1,1], 'conv4_1')

        with tf.variable_scope('conv4_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv4_2 = conv_layer(self.conv4_1, [3,3], 512, [1,1,1,1], 'conv4_2')

        with tf.variable_scope('conv4_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv4_3 = conv_layer(self.conv4_2, [3,3], 512, [1,1,1,1], 'conv4_3')
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = 'pool_layer4')

        with tf.variable_scope('conv5_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv5_1 = conv_layer(self.pool4, [3,3], 512, [1,1,1,1], 'conv5_1')
        with tf.variable_scope('conv5_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv5_2 = conv_layer(self.conv5_1, [3,3], 512, [1,1,1,1], 'conv5_2')
        with tf.variable_scope('conv5_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv5_3 = conv_layer(self.conv5_2, [3,3], 512, [1,1,1,1], 'conv5_3')
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = 'pool_layer5')

        reshaped_output = tf.reshape(self.pool5, [-1, 25088])
        reshaped_dim = reshaped_output.get_shape()[1].value
        with tf.variable_scope('fc6', reuse=tf.AUTO_REUSE) as scope:

            full_weight1 = truncated_normal_var(name="fc6_W", shape=[reshaped_dim, 4096], dtype=tf.float32)
            full_bias1 = zero_var(name='fc6_b', shape=[4096], dtype=tf.float32)
            self.full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))
        self.drop_layer1 = tf.nn.dropout(self.full_layer1, 0.5)

        with tf.variable_scope('fc7', reuse=tf.AUTO_REUSE) as scope:
            full_weight2 = truncated_normal_var(name="fc7_W", shape=[4096, 4096], dtype=tf.float32)
            full_bias2 = zero_var(name='fc7_b', shape=[4096], dtype=tf.float32)
            self.full_layer2 = tf.nn.relu(tf.add(tf.matmul(self.drop_layer1, full_weight2), full_bias2))
        self.drop_layer2 = tf.nn.dropout(self.full_layer2, 0.5)

        with tf.variable_scope('fc8', reuse=tf.AUTO_REUSE) as scope:
            full_weight3 = truncated_normal_var(name="fc8_W", shape=[4096, 1000], dtype=tf.float32)
            full_bias3 = zero_var(name='fc8_b', shape=[1000], dtype=tf.float32)
            self.full_layer3 = tf.nn.relu(tf.add(tf.matmul(self.drop_layer2, full_weight3), full_bias3))

       	self.prob = tf.nn.softmax(self.full_layer3, name="prob")

    def test5(self):
        return self.conv5_3

    def test4(self):
        return self.conv4_3

    def fc7(self):
        return self.full_layer2

    def prob1(self):
        return self.prob

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            k_root = k[:-2]
            v1 = sess.graph.get_tensor_by_name(k_root+'/'+k+':0')
            full = sess.run(tf.assign(v1, weights[k]))

