#author:David Wong
#todo:Final Design

import tensorflow as tf
import tensorlayer as tl
import DiyLayers

def conv2d(input=None,
              act=tf.nn.relu,
              enable_bn=False,
              shape=[3,3,3,12],
              stride=1,
              padding='SAME',
              is_train=True,
              name='net_conv'):

    assert input is not None, 'User Error:layer:' + name + ' input is None...'

    if enable_bn:
        net = tl.layers.Conv2dLayer(input,
                                    act=tf.identity,
                                    shape=shape,
                                    strides=[1,stride,stride,1],
                                    padding=padding,
                                    W_init=tf.truncated_normal_initializer(stddev=0.01),
                                    b_init=None,
                                    name=name+'_conv')

        net = tl.layers.BatchNormLayer(net,
                                       is_train=is_train,
                                       beta_init=tf.constant_initializer(value=0.01),
                                       act=act,
                                       name=name+'_bn')
    else:
        net = tl.layers.Conv2dLayer(input,
                                    act=act,
                                    shape=shape,
                                    strides=[1,stride,stride,1],
                                    padding=padding,
                                    W_init=tf.truncated_normal_initializer(stddev=0.01),
                                    b_init=tf.constant_initializer(value=0.01),
                                    name=name+'_conv')
    return net

def Dense(input=None,
             act=tf.nn.relu,
             enable_bn=False,
             n_units=100,
             is_train=True,
             drop=0.8,
             name='dense'):

    assert input is not None,'User Error:layer:'+name+' input is None...'

    if enable_bn:
        net_op = tl.layers.DenseLayer(input,
                                   n_units=n_units,
                                   act=tf.identity,
                                   b_init=None,
                                   name=name+'_dense')
        net = tl.layers.BatchNormLayer(net_op,
                                       is_train=is_train,
                                       beta_init=tf.constant_initializer(value=0.01),
                                       act=act,
                                       name=name+'_bn')
    else:
        net = tl.layers.DenseLayer(input,
                                   n_units=n_units,
                                   act=act,
                                   name=name+'_dense')

    net = tl.layers.DropoutLayer(net,
                                 keep=drop,
                                 is_train=is_train,
                                 is_fix=True,
                                 name=name+'_dropout')
    return net

def Pool_max(input=None,size=3,stride=1,padding='SAME',name='pool_max'):

    assert input is not None, 'User Error:layer:' + name + ' input is None...'

    return tl.layers.PoolLayer(input,
                               ksize=[1,size,size,1],
                               strides=[1,stride,stride,1],
                               padding=padding,
                               pool=tf.nn.max_pool,
                               name=name+'_maxpool')

def Pool_avg(input=None,size=3,stride=1,padding='SAME',name='pool_avg'):

    assert input is not None, 'User Error:layer:' + name + ' input is None...'

    return tl.layers.PoolLayer(input,
                               ksize=[1,size,size,1],
                               strides=[1,stride,stride,1],
                               padding=padding,
                               pool=tf.nn.avg_pool,
                               name=name+'_avgpool')


def MobileV1_conv(input=None, is_train=True,kernel_shape=[3, 3, 1, 1], stride=1, padding='SAME',name='MobileV1_conv'):

    assert input is not None, 'User Error:layer:' + name + ' input is None...'

    # stride = 2 if downSample else 1
    dep = tl.layers.DepthwiseConv2d(input,
                                    channel_multiplier=1,
                                    shape=(kernel_shape[0],kernel_shape[1]),
                                    strides=(stride, stride),
                                    act=tf.identity,
                                    padding=padding,
                                    b_init=None,
                                    name=name + '_depthwise')
    bn1 = tl.layers.BatchNormLayer(dep,
                                   is_train=is_train,
                                   beta_init=tf.constant_initializer(value=0.01),
                                   act=tf.nn.relu,
                                   name=name + '_bn1')
    con = tl.layers.Conv2dLayer(bn1,
                                shape=[1,1,kernel_shape[2],kernel_shape[3]],
                                act=tf.identity,
                                strides=[1,1,1,1],
                                b_init=None,
                                name=name + '_con1x1')
    bn2 = tl.layers.BatchNormLayer(con,
                                   is_train=is_train,
                                   beta_init=tf.constant_initializer(value=0.01),
                                   act=tf.nn.relu,
                                   name=name + '_bn2')
    return bn2


def MobileV2_conv(input=None,
                  is_train=True,
                  in_out_channels=[3,3],
                  expansion=6,
                  strides=1,
                  name='MobileV2_conv'):

    assert input is not None, 'User Error:layer:' + name + ' input is None...'


    con1 = tl.layers.Conv2dLayer(input,
                                shape=[1,1,in_out_channels[0],in_out_channels[0]*expansion],
                                act=tf.identity,
                                strides=[1,1,1,1],
                                b_init=None,
                                name=name + '_con1')
    bn1 = tl.layers.BatchNormLayer(con1,
                                   is_train=is_train,
                                   beta_init=tf.constant_initializer(value=0.01),
                                   act=tf.nn.relu6,
                                   name=name + '_bn1')

    dw1 = tl.layers.DepthwiseConv2d(bn1,
                                    channel_multiplier=1,
                                    shape=(3,3),
                                    strides=(strides, strides),
                                    act=tf.identity,
                                    padding='SAME',
                                    b_init=None,
                                    name=name + '_dw1')
    bn2 = tl.layers.BatchNormLayer(dw1,
                                   is_train=is_train,
                                   beta_init=tf.constant_initializer(value=0.01),
                                   act=tf.nn.relu6,
                                   name=name + '_bn2')
    con2 = tl.layers.Conv2dLayer(bn2,
                                shape=[1,1,in_out_channels[0]*expansion,in_out_channels[1]],
                                act=tf.identity,
                                strides=[1,1,1,1],
                                b_init=None,
                                name=name + '_con2')
    bn3 = tl.layers.BatchNormLayer(con2,
                                   is_train=is_train,
                                   beta_init=tf.constant_initializer(value=0.01),
                                   act=tf.identity,
                                   name=name + '_bn3')


    if bn3.outputs.get_shape().as_list()[-1] == in_out_channels[0]:
        out = DiyLayers.AddLayer([bn3,input],name=name +'out')
    else:
        out = bn3

    return out

def center_loss(features, labels, alpha, num_classes):

    len_features = features.get_shape()[1]

    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)

    labels = tf.reshape(labels, [-1])

    labels = tf.cast(labels,tf.int32)

    centers_batch = tf.gather(centers, labels)

    loss = tf.nn.l2_loss(features - centers_batch)

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op
