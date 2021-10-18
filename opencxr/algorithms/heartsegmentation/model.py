# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, \
    UpSampling2D, BatchNormalization
from keras.models import Model

'''=================================================================================================
Model architecture for heart segmentation
With the dropout settings you can specify the depth (-1 is the deepest layer) and the probability.
==================================================================================================='''


def unet(x_in, k_size=3, optimizer='adam', depth=4, downsize_filters_factor=1, batch_norm=False, activation='relu',
         initializer='glorot_uniform', seed=42, upsampling=False, dropout=[(-1, 0.5)], n_convs_per_layer=2, lr=False,
         loss='binary'):
    # Fix the seed into given argument.
    np.random.seed(seed)

    # if True, deepest layer with 0.5 probability.
    if dropout == True:
        dropout = [(-1, 0.5)]

    # Define the hyperparameter settings.
    # Max number of init filters: 64
    settings = {
        'n_classes': 2,  # nb of classes always 2.
        'depth': depth,  # unet depth 4
        'filters': 64 / downsize_filters_factor,  # unet filters 64
        'kernel_size': (k_size, k_size),  # unet kernel size (3,3)
        'pool_size': (2, 2),  # standard 2,2
        'n_convs_per_layer': n_convs_per_layer,  # standard perlayer 2 conv.
        'activation': activation,  # relu standard
        'kernel_initializer': initializer,  # he_normal standard
        'padding': 'same',  # valid standard
        'dropout': dropout,  # dropout is the deepest layer.
        'batch_norm': batch_norm,  # if to use batch norm or not.
        'upsampling': upsampling,  # if true upsampling, else conv2dtranspose.
    }

    # K.clear_session()
    # x_in: input layer, data format: (batch, height, width, channels)
    data = Input(shape=x_in)  # input layer
    layers = {}  # hold all the layers in a dictionary.
    l = data  # input dimension.

    def conv(filters):
        '''
        Return convolutional layer given setting arguments.
        '''
        return Conv2D(filters=filters,
                      kernel_size=settings['kernel_size'],
                      activation=settings['activation'],
                      kernel_initializer=settings['kernel_initializer'],
                      padding=settings['padding'])

    def dropout(rate):
        '''
        Return dropout layer given rate.
        '''
        return Dropout(rate)

    def batchnorm():
        '''
        Return batchnormalization layer.
        '''
        return BatchNormalization()

    def pool():
        '''
        Return pooling layer given setting argument.
        '''
        return MaxPooling2D(pool_size=settings['pool_size'])

    def concat():
        '''
        Return Concatenate function. 
        '''
        return Concatenate()

    # transpose convolutional filter.
    def t_conv(filters):

        if upsampling:
            # print('upsampling through neareset neighbor')
            return UpSampling2D(size=settings['pool_size'])

        else:
            # print('upsampling with conv2d learning transpose filter')
            return Conv2DTranspose(filters=filters,
                                   kernel_size=settings['pool_size'],
                                   strides=settings['pool_size'],
                                   kernel_initializer=settings['kernel_initializer'],
                                   padding=settings['padding'])

    # add a new layer to the existing architecture.
    def add(layer, l_in, name):
        '''
        Add a new layer to the dictionary given name.
        '''
        layers[name] = layer(l_in)
        return layers[name]

    depths = list(range(settings['depth']))  # [0,1,2,3]

    if settings['dropout'] != False:
        dropout_depths = list(range(settings['depth'] + 1))  # [1,2,3,4]
        dropouts = {dropout_depths[d]: rate for d, rate in
                    settings['dropout']}  # apply to the last layer with 0.5 rate..

    contracting_outputs = {}
    # contracting.
    for i in depths:
        for j in range(settings['n_convs_per_layer']):
            # number of filters.
            n = int(settings['filters'] * (2 ** i))
            # create the layer given previous , add it to dictionary of layers.
            l = add(conv(n), l, 'conv_down_{}_{}'.format(i, j))

            contracting_outputs[i] = l

            if settings['batch_norm'] == True:
                l = add(batchnorm(), l, 'batch_norm_{}_{}'.format(i, j))
                contracting_outputs[i] = l

            if settings['dropout'] != False:
                if i in dropouts:
                    l = add(dropout(dropouts[i]), l, 'dropout_{}_{}'.format(i, j))
                    contracting_outputs[i] = l

        l = add(pool(), l, 'pool_{}'.format(i))

    # bottom
    i = settings['depth']
    for j in range(settings['n_convs_per_layer']):
        n = int(settings['filters'] * 2 ** settings['depth'])
        l = add(conv(n), l, 'conv_{}_{}'.format(settings['depth'], j))
        # print('layer shape', l.shape)

        if settings['dropout'] != False:
            if i in dropouts:
                l = add(dropout(dropouts[i]), l, 'dropout_{}_{}'.format(i, j))

    # expanding
    for i in reversed(depths):
        n = int(settings['filters'] * 2 ** i)
        l = add(t_conv(n), l, 't_conv{}'.format(i))
        l = add(concat(), [l, contracting_outputs[i]], 'concat_{}'.format(i))
        for j in range(settings['n_convs_per_layer']):
            n = int(settings['filters'] * 2 ** i)
            l = add(conv(n), l, 'conv_up_{}_{}'.format(i, j))

            if settings['batch_norm'] == True:
                l = add(batchnorm(), l, 'batch_norm_{}_{}'.format(i, j))
                contracting_outputs[i] = l

    lr = float(lr)
    if lr != 'default':

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=lr, decay=1e-3)
        elif optimizer == 'adagrad':
            optimizer = keras.optimizers.Adagrad(learning_rate=lr)
        elif optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)

    else:  # use default values for the optimizers.
        if optimizer == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    if loss == 'binary':
        out = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')
        l = add(out, l, 'out')
        model = Model(data, l)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    else:  # if not binary classification
        out = Conv2D(filters=2, kernel_size=(1, 1), activation='softmax')
        l = add(out, l, 'out')
        model = Model(data, l)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model  # l, layers
