# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:11:36 2017

@author: Nguyen
"""

#Linear Algebra
import numpy as np

#Data Frames
import pandas as pd

#Neural Network
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
    #Input Layer
    input_layer = tf.reshape(features['x'], [-1,75,75,1]) #( -1: infer shape of inputs, 75,75 pixels and 1 color range for grey)
    
    #Conv 1
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size = [3,3], padding = 'same', activation=tf.nn.relu) #OUTPUT TENSOR: [size, 75,75,32]
    #Pooling 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1,pool_size = [2,2], strides = 2) #OUTPUT TENSOR: [size,37,37,32 ]
    #Conv2
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = [3,3], padding = 'same',  activation = tf.nn.relu) #OUTPUT TENSOR: [size, 37,37,64]
    #Pooling 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2) #OUTPUT TENSOR:[size,18,18,64]
    #Dense Layer
    pool2_flattened = tf.reshape(pool2, [-1,18*18*64])#Find out why we reshape this
    dense = tf.layers.dense(inputs=pool2_flattened, units=1024, activation = tf.nn.relu) #unts are neurons
    #Apply dropout to avoid overfitting
    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training=mode ==tf.estimator.ModeKeys.TRAIN)
    
    #Logits Layer
    logits = tf.layers.dense(inputs = dropout, units = 2)
    
    predictions = {
            #Predictions for PREDICT and EVAL modes
            'classes': tf.argmax(input=logits, axis = 1),
            #Add softmax to the graph to predict
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #loss functions
    onehot_labels = tf.one_hot(indicies=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)
    
    #Training operation
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.GradientDescentOptimizer(0.001)
        train_op = opt.minimize(loss = loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op= train_op)
    eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels = labels, predictions = predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,eval_metric_ops= eval_metric_ops)
    
    

if __name__ == '__main__':
    #Ready data
    train = pd.read_json('../Data/train.json')
    labels = train.is_iceberg.values
    train = train.drop(['is_iceberg', 'id', 'band_2', 'inc_angle'], axis = 1)
    
    #Create classifier
    model_classifier = tf.estimator.Estimator(model_fn = cnn_model, model_dir='../tmp/conv_iceberg_model')
    
    #Create logging
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)
    
    #Train model
    training_model = tf.estimator.inputs.numpy_input_fn(x={'x':train.values}, y=labels, batch_size = 100, num_epochs = None, shuffle = True)
    model_classifier.train(input_fn = training_model, steps = 20000, hooks=[logging_hook])
    
    #Evaluate model
    
    