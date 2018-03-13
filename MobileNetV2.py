#coding:utf-8
import tensorflow as tf
import tensorlayer as tl
import HighLayer as hl
from make_tfrecords import tfrecordOperator
import numpy as np
import Config as cfg
import cv2
import argparse as args
import time

def one_hot(dat):
    length = len(dat)
    labels = np.zeros((length, cfg.n_class))
    for i in range(length):
        temp = np.zeros((cfg.n_class,),dtype=np.uint8)
        temp[dat[i]]=1
        labels[i,:] = temp
    return labels

def getBatchFromOne(img):
    image_batch = np.zeros((1,cfg.image_size,cfg.image_size,1))
    temp = np.reshape(img,(cfg.image_size,cfg.image_size,1))
    image_batch[0,:,:,:] = temp
    return image_batch

def built_MobileNetV2(x_=None,y_=None,is_train=True,reuse=False):

    with tf.variable_scope("model",reuse=reuse):

        tl.layers.set_name_reuse(reuse)

        net = tl.layers.InputLayer(x_, name='input')

        net = hl.conv2d(input=net,enable_bn=True,shape=[3,3,1,32],stride=2,is_train=is_train,name='con1')
        #bottleneck1
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[32, 16], expansion=1, strides=1,
                               name='bottle1_1')
        # bottleneck2
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[16, 24], expansion=6, strides=2,
                               name='bottle2_1')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[24, 24], expansion=6, strides=1,
                               name='bottle2_2')
        # bottleneck3
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[24, 32], expansion=6, strides=2,
                               name='bottle3_1')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[32, 32], expansion=6, strides=1,
                               name='bottle3_2')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[32, 32], expansion=6, strides=1,
                               name='bottle3_3')
        # bottleneck4
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[32, 64], expansion=6, strides=2,
                               name='bottle4_1')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[64, 64], expansion=6, strides=1,
                               name='bottle4_2')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[64, 64], expansion=6, strides=1,
                               name='bottle4_3')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[64, 64], expansion=6, strides=1,
                               name='bottle4_4')
        # bottleneck5
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[64, 96], expansion=6, strides=1,
                               name='bottle5_1')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[96, 96], expansion=6, strides=1,
                               name='bottle5_2')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[96, 96], expansion=6, strides=1,
                               name='bottle5_3')
        # bottleneck6
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[96, 160], expansion=6, strides=2,
                               name='bottle6_1')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[160, 160], expansion=6, strides=1,
                               name='bottle6_2')
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[160, 160], expansion=6, strides=1,
                               name='bottle6_3')
        # bottleneck7
        net = hl.MobileV2_conv(input=net, is_train=is_train, in_out_channels=[160, 320], expansion=6, strides=1,
                               name='bottle7_1')
        #end
        net = hl.conv2d(input=net,enable_bn=True,shape=[1,1,320,1280],stride=1,is_train=is_train,name='con2')
        net = hl.Pool_avg(input=net,size=7,stride=1,padding='VALID',name='avg')

        net = tl.layers.FlattenLayer(net,name='flattern')
        net = tl.layers.DropoutLayer(net,keep=0.5,is_fix=True,is_train=is_train,name='drop1')
        net = hl.Dense(input=net,enable_bn=False,act=tf.identity,n_units=6,is_train=False,name='out_layer')

        net_out = net.outputs

        cost_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net_out))

        L2 = 0
        for w1 in tl.layers.get_variables_with_name('W_conv2d', True, False):
            L2+=tf.contrib.layers.l2_regularizer(0.001)(w1)
        for w2 in tl.layers.get_variables_with_name('W_sepconv2d', True, False):
            L2 += tf.contrib.layers.l2_regularizer(0.001)(w2)

        cost = cost_out + L2

        correct_prediction = tf.equal(tf.argmax(net_out, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net,cost,accuracy

if __name__ == '__main__':

    paraser = args.ArgumentParser()
    paraser.add_argument('--model',type=str)
    args = paraser.parse_args()

    x_ = tf.placeholder(tf.float32, [None,cfg.image_size,cfg.image_size,1],name='img')
    y_ = tf.placeholder(tf.float32, [None,cfg.n_class],name='label')
    lr_ = tf.placeholder(tf.float32,name='lr')

    network,train_cost,_ = built_MobileNetV2(x_,y_,is_train=True,reuse=False)

    network_test,test_cost,acc = built_MobileNetV2(x_,y_,is_train=False,reuse=True)

    train_step = tf.train.AdamOptimizer(lr_).minimize(train_cost)

    test_outputs = network_test.outputs

    tf.summary.scalar('lr',lr_)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('test_loss',train_cost)
    tf.summary.scalar('train_loss', test_cost)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=cfg.max_to_keep)

    merged = tf.summary.merge_all()
    print('params:',network.count_params())

    dataset = tfrecordOperator()

    train_img, train_labels = dataset.read_from_tfrecords(filename='test_CK+.tfrecords', imgShape=[cfg.image_size, cfg.image_size, 1])
    train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img, train_labels], batch_size=cfg.train_batch_size,
                                              num_threads=3,capacity=800, min_after_dequeue=100)

    test_img, test_labels = dataset.read_from_tfrecords(filename='train_CK+.tfrecords', imgShape=[cfg.image_size, cfg.image_size, 1])
    test_img_batch, test_label_batch = tf.train.shuffle_batch([test_img, test_labels], batch_size=cfg.test_batch_size,
                                              num_threads=3,capacity=100, min_after_dequeue=50)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    decay_power=0

    # train_writer = tf.summary.FileWriter('logs/train/',sess.graph)
    test_writer = tf.summary.FileWriter('logs/')

    if args.model is not None:
        saver.restore(sess,'./restore/model_'+args.model+'.ckpt')
        print('restore model file '+'./restore/model_'+args.model+'.ckpt')
    else:
        print('no model file restored.')

    for i in range(4000+1):

        train_a, train_b = sess.run([train_img_batch, train_label_batch])
        train_b = one_hot(train_b)

        test_a, test_b = sess.run([test_img_batch, test_label_batch])
        test_b = one_hot(test_b)

        if i%10==0:
            decay_power+=1
        temp_lr = cfg.lr*(cfg.decay_lr**decay_power)

        _,train_loss = sess.run([train_step,train_cost],feed_dict={x_: train_a, y_: train_b,lr_:temp_lr})

        summary,test_accuracy ,test_loss= sess.run([merged,acc,test_cost],feed_dict={x_: test_a, y_: test_b,lr_:temp_lr})

        test_writer.add_summary(summary, i)

        # if i % 20 == 0 and i>=500:
        #     saver.save(sess, "./model/model_" + "{:.3}".format(test_accuracy) + '_' + str(i)+".ckpt")

        print('step: '+str(i)+
              ', testing accuracy: '+'{:.3}'.format(test_accuracy)+
              ', train loss: ' + '{:.4}'.format(train_loss) +
              ', test loss: '+'{:.4}'.format(test_loss)+
              ', train/test: ' + '{:.4}'.format(train_loss/test_loss) +
              ', lr: '+'{:.4}'.format(temp_lr)
              )
    saver.save(sess, "./model/model_final.ckpt")
    sess.close()

