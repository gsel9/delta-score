import numpy as np 
import tensorflow as tf 


def logreg_multi(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                3, activation='softmax',
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed), 
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )
        ]
    )
    return model


def logreg(n_features=10, y=None, seed=42, l2_reg=0.01):

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                1, activation='sigmoid', 
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed), 
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )
        ],
        name="logreg"
    )
    return model


def mlp1(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        ],
        name="mlp1"
    )
    return model


def mlp2(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        ],
        name="mlp2"
    )
    return model


def mlp3(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(128, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(64, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        ],
        name="mlp3"
    )
    return model


def cnn(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    model = tf.keras.models.Sequential(
        [   
            tf.keras.layers.Dense(n_features, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed), 
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Convolution1D(64, 3, activation='relu', padding='same',
                                          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))
        ],
        name="cnn"
    )

    return model


def lstm(n_features=10, y=None, seed=42, l2_reg=0.01):
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(n_features, activation='relu', 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed), 
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(64, #input_shape=[3, n_features],
                                 kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
            tf.keras.layers.Dense(1, activation="sigmoid", 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))
        ],
        name="lstm"
    )
    return model