# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:18:14 2017

@author: Nguyen
"""

#Linear Algebra
import numpy as np

#Data Frames
import pandas as pd

#Neural Network
import tensorflow as tf

#Supporting Modules
from sklearn.metrics import log_loss


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.Graph().as_default()
    train = pd.read_json('../Data/train.json')
    labels = train.is_iceberg.values
    n_samples = labels.shape[0]
    batch_size = 100
    debug_step = 10
    x_band1 = np.array([np.array(band).astype(np.float32) for band in train['band_1']])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])
    #X_train = np.concatenate([x_band1[:,:,:,np.newaxis], x_band2[:,:,:,np.newaxis], ((x_band1+x_band1)/2)[:, :, :,np.newaxis]],axis=-1)
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    #Build our net
    x = tf.placeholder(tf.float32, shape=[None,5625])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    
    with tf.name_scope('Model'):
        with tf.name_scope('Conv1'):
            #The shape will be [[patch size], input channels, output channels]
            W_conv1 = weight_variable([3,3,1,32])
            b_conv1 = bias_variable([32])
            x_image = tf.reshape(x, [-1, 75,75,1])
            #Layer 1
            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope('Conv2'):
            #Layer 2
            W_conv2 = weight_variable([3,3,32,64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        with tf.name_scope('DenseLayer'):
            #Dense layer
            W_fc1 = weight_variable([19*19*64,1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1,19*19*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        with tf.name_scope('Dropout'):
            #Dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        with tf.name_scope('OutputLayer'):
            #Final layer (logits)
            W_fc2 = weight_variable([1024,2])
            b_fc2 = bias_variable([2])
            output_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = output_conv))
    opt = tf.train.AdamOptimizer(.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(output_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    probabilities = tf.nn.softmax(output_conv)
    tf.summary.FileWriterCache.clear()
    saver = tf.train.Saver()
    print('Beginning Session...')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graph', graph = sess.graph)
        sess.run(tf.global_variables_initializer())
        labels = labels.eval()
        for i in range(30):
            print('Current Epoch: ',i)
            for batch in range(int(n_samples/batch_size)):
                batch_x = x_band1[batch*batch_size : (1+batch) * batch_size]
                batch_y = labels[batch*batch_size : (1+batch) * batch_size]
                sess.run([opt], feed_dict={x:batch_x,y_:batch_y,keep_prob:.4})
            if i % debug_step == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch_x,y_:batch_y, keep_prob : 1.0})
                    ls = log_loss(batch_y, output_conv.eval(feed_dict={x:batch_x,y_:batch_y, keep_prob : 1.0}))
                    prob = probabilities.eval(feed_dict={x:batch_x,y_:batch_y, keep_prob : 1.0})
                    print('step %d, training accuracy %g, log loss %g' % (i, train_accuracy, ls))
                    print('Probabilities of each class: ', prob)
        train_accuracy = accuracy.eval(feed_dict={x:x_band1,y_:labels, keep_prob : 1.0})
        ls = log_loss(labels, output_conv.eval(feed_dict={x:x_band1,y_:labels, keep_prob : 1.0}))
        print('Final training accuracy %g, log loss %g' % (train_accuracy, ls))
        save_path = saver.save(sess, '../tmp/')
        print('Model saved in file: %s' % save_path)
    writer.close()
    print('End of program...')