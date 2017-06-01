from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def gazenet(inputs,
            head_inputs,
            head_loc,
            num_classes = 1000,
            is_training=True,
            dropout_keep_prob= 0.5,
            spatial_squeeze = True,
            scope = 'gazenet'):
    #input is a tensor of [batch_size, height, width, channels]
    with variable_scope.variable_scope(scope, "gazenet", [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with arg_scope(
            [layers.conv2d, layers_lib.fuly_connected, layers_lib.max_pool2d],
                outputs_collections =[end_points_collection]):
            saliency_net= layers.conv2d(inputs, 64, [11,11], 4, padding = 'VALID', scope = 'conv1')
            #56x56 , 52x52
            saliency_net = layers_lib.max_pool2d(saliency_net, [3,3], 2, scope = 'pool1')
            #28x28, 26x26
            saliency_net = layers.conv2d(saliency_net, 192, [5,5], scope = 'conv2')
            saliency_net = layers_lib.max_pool2d(saliency_net, [3,3], 2, scope = 'pool2')
            #14x14
            saliency_net = layers.conv2d(saliency_net, 384, [3,3], scope = 'conv3')
            saliency_net = layers.conv2d(saliency_net, 384, [3,3], scope = 'conv4')
            saliency_net = layers.conv2d(saliency_net, 256, [3,3], scope = 'conv5')
            # net = layers_lib.max_pool2d(net, [3,3], 2, scope = 'pool5')
            # last layer is a 1x1x 256 conv layer
            saliency_net = layers.conv2d(saliency_net, 1, [1,1], scope = 'conv6')
            #14x14, 13x13

            gaze_net = layers.conv2d(head_inputs, 64, [11,11], 4, padding ="VALID", scope = 'gaze_conv1')
            gaze_net = layers_lib.max_pool2d(gaze_net, [3,3], 2, scope = 'gaze_pool1')
            gaze_net = layers.conv2d(gaze_net, 192, [5,5], scope = 'gaze_conv2')
            gaze_net = layers_lib.max_pool2d(gaze_net, [3,3],2, scope = 'gaze_pool2')
            gaze_net = layers.conv2d(gaze_net, 384, [3,3], scope = 'gaze_conv3')
            gaze_net = layers.conv2d(gaze_net, 384, [3,3], scope = 'gaze_conv4')
            gaze_net = layers.conv2d(gaze_net, 256, [3,3], scope = 'gaze_conv5')

            gaze_net = layers.ops.fc(gaze_net, 100, scope = 'fc1')
            #Concatenate the location of the head
            head_loc = layers.flatten(head_loc)
            gaze_net = tf.concat(gaze_net, head_loc, 0)
            # to do: use tf.contrib.stack
            gaze_net = layers.ops.fc(gaze_net, 375, scope = 'fc2')
            gaze_net = layers.ops.fc(gaze_net, 200, scope = 'fc3')
            gaze_net = layers.ops.fc(gaze_net, 169, scope = 'fc4')


            product_mask = tf.multiply(gaze_net, layers.flatten(saliency_net))

            prediction_mask = layers.ops.fc(product_mask, 25, scope='prediction')




