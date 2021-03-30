import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

AVERAGE_POOLING = True
def Network(
                      image,
                      is_training=True,
                      keepprob=0.5,
                      class_num = 10,
                      scope='classfier3x3'):
    
    batch_norm_params = {   
        'is_training': True,
        'decay': 0.9,
        'epsilon': 1e-3,
        'scale': True,
        'updates_collections': None
                        }
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn = tf.nn.relu,
                weights_initializer = tf.contrib.keras.initializers.he_normal(),
                normalizer_fn = slim.batch_norm,
                normalizer_params = batch_norm_params):
            
            net = slim.conv2d(image, 64, [3, 3], stride = 2, padding = 'SAME', trainable = is_training, scope = 'conv1_1')#20x20
            net = slim.max_pool2d(net, [2,2])#10x10
            
            net = slim.conv2d(net, 128, [1, 1], padding = 'SAME', trainable = is_training, scope = 'conv2_1')
            net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', trainable = is_training, scope = 'conv2_2')
            net = slim.conv2d(net, 128, [3, 3], padding = 'SAME', trainable = is_training, scope = 'conv2_3')
            net = slim.max_pool2d(net, [2,2])#5x5
                

            net = slim.conv2d(net, 256, [1, 1], padding = 'SAME', trainable = is_training, scope = 'conv3_1')
            net = slim.conv2d(net, 256, [3, 3], padding = 'SAME', trainable = is_training, scope = 'conv3_2')
            net = slim.conv2d(net, 256, [2, 2], padding = 'VALID', trainable = is_training, scope = 'conv3_3')#4x4
            net = slim.max_pool2d(net, [2,2])#2x2
                
            net = slim.conv2d(net, 512, [1, 1], padding='VALID', 
                                  trainable = is_training, scope='conv4_1')
                
            if AVERAGE_POOLING:
                net = slim.avg_pool2d(net, [net.get_shape().as_list()[1], net.get_shape().as_list()[2]], scope='globel_pool')
                net = tf.reshape(net,[-1,512])
            else:
                net = tf.reshape(net,[-1,2048])
            net = slim.fully_connected(net, 10, scope='fc1') 
    return net
