import tensorflow as tf


import tensorflow as tf
import tensorflow.contrib.slim as slim

def C3D(input, dimensions, dropout=False, regularizer=True):

    if regularizer: regularizer = tf.contrib.layers.l2_regularizer(0.005) # 0.005

    with tf.variable_scope('C3D'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            weights_initializer=tf.random_normal_initializer(stddev=0.01)
                            #weights_regularizer=slim.l2_regularizer(0.0005)
                            ):
            conv_1 = slim.conv3d(input, 64, 3, 1, scope='conv_1')
            maxpool_1 = slim.max_pool3d(conv_1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding='SAME', scope='maxpool_1')
            conv_2 = slim.conv3d(maxpool_1, 128, 3, 1, scope='conv_2')
            maxpool_2 = slim.max_pool3d(conv_2, [2, 2, 2], [2, 2, 2], padding='SAME', scope='maxpool_2')
            conv_3 = slim.conv3d(maxpool_2, 256, 3, 1, scope='conv_3')
            maxpool_3 = slim.max_pool3d(conv_3, [2, 2, 2], [2, 2, 2], padding='SAME', scope='maxpool_3')
            conv_4 = slim.conv3d(maxpool_3, 256, 3, 1, scope='conv_4')
            maxpool_4 = slim.max_pool3d(conv_4, [2, 2, 2], [2, 2, 2], padding='SAME', scope='maxpool_4')
            conv_5 = slim.conv3d(maxpool_4, 256, 3, 1, scope='conv_5')
            maxpool_5 = slim.max_pool3d(conv_5, [2, 2, 2], [2, 2, 2], padding='SAME', scope='maxpool_5')

        pool_shape = maxpool_5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] * pool_shape[4]  # 1 x 4 x 4 x 256 = 4096
        reshaped = tf.reshape(maxpool_5, [pool_shape[0], nodes])  # pool_shape[0] is N, batch_size

        with tf.variable_scope('fc6'):
            fc6_weight = tf.get_variable('weight', [nodes, 2048],
                                         initializer=tf.random_normal_initializer(stddev=0.005))
            if regularizer:
                tf.add_to_collection("weight_decay_loss", regularizer(fc6_weight))
            fc6_bias = tf.get_variable('bias', [2048], initializer=tf.constant_initializer(1.0))
            fc6 = tf.nn.relu(tf.matmul(reshaped, fc6_weight) + fc6_bias)
            if dropout:
                fc6 = tf.nn.dropout(fc6, 0.5)

        with tf.variable_scope('fc7'):
            fc7_weight = tf.get_variable('weight', [2048, 2048], initializer=tf.random_normal_initializer(stddev=0.005))
            if regularizer:
                tf.add_to_collection("weight_decay_loss", regularizer(fc7_weight))
            fc7_bias = tf.get_variable('bias', [2048], initializer=tf.constant_initializer(1.0))
            fc7 = tf.nn.relu(tf.matmul(fc6, fc7_weight) + fc7_bias)
            if dropout:
                fc7 = tf.nn.dropout(fc7, 0.5)

        with tf.variable_scope('fc8'):  # fc8
            out_weight = tf.get_variable('weight', [2048, dimensions], initializer=tf.random_normal_initializer(stddev=0.01))
            if regularizer:
                tf.add_to_collection("weight_decay_loss", regularizer(out_weight))
            out_bias = tf.get_variable('bias', [dimensions], initializer=tf.constant_initializer(0.0))
            out = tf.matmul(fc7, out_weight) + out_bias  # DO NOT ADD RELU!!!

        return out