import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#__all__ = ['ConvBlock','repeat_elem','ResConvBlock','gating_signal','AttentionBlock','AttResUNet','Unet']

def ConvBlock(x, filter_size, filter_num, dropout=0, batch_norm=True):
    """
    inherited from @DigitalSreeni
    Standard 2xconvolution block inherited from DigitalSreeni
    :params x: input tensor
    :params filter_size: size of the square 2D convolution filters, goes by (filter_size, filter_size))
    :params filter_num: number of convolution filters
    :params dropout: dropout layer, default is 0 (no dropout)
    :params batch_norm: batch normalization layer
    :return: convolution block
    """
    # convolution layer
    conv = layers.Conv2D(filter_num, (filter_size,filter_size), padding='same')(x)
    # batch normalization
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    # consequtive conv layer
    # convolution layer
    conv = layers.Conv2D(filter_num, (filter_size,filter_size), padding='same')(conv)
    # batch normalization
    if batch_norm:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    return conv


def repeat_elem(tensor, rep, axis=3):
    """
    inherited from @DigitalSreeni
    Lambda function to repeat the elements of a tensor along an axis by a factor of rep
    :params tensor: input tensor
    :params rep: numper of repeat along axis: axis
    :params axis: specify which access to repeat, default=3
    :return: repeated tensor
    """
    return layers.Lambda(lambda x, repnum: keras.backend.repeat_elements(x, repnum, axis=axis),
                         arguments={'repnum': rep})(tensor)

def ResConvBlock(x, filter_size, filter_num, dropout=0, batch_norm=True):
    """
    inherited from @DigitalSreeni
    :params x: input tensor
    :params filter_size: size of the square 2D convolution filters, goes by (filter_size, filter_size))
    :params filter_num: number of convolution filters
    :params dropout: dropout layer, default is 0 (no dropout)
    :params batch_norm: batch normalization layer
    :return: residual convolution block
    """
    conv = ConvBlock(x, filter_size, filter_num, dropout, batch_norm)
    res_shortcut = layers.Conv2D(filter_num, kernel_size=(1,1), padding='same')(x)
    if batch_norm:
        res_shortcut = layers.BatchNormalization(axis=3)(res_shortcut)
    res_path = layers.add([conv, res_shortcut])
    res_path = layers.Activation('relu')(res_path)
    return res_path


def gating_signal(x, out_size, batch_norm=False):
    """
    inherited from @DigitalSreeni
    resize the down layer feature map into the same dimension as the up layer feature map using 1x1 convolution filter
    :params x: input tensor
    :params out_size: size of the up layer feature map
    :params batch_norm: batch normalization, default: False
    :return: gating feature map with the same dimension of the up layer feature map
    """
    
    out = layers.Conv2D(out_size, (1,1), padding='same')(x)
    if batch_norm:
        out = layers.BatchNormalization(axis=3)(out)
    out = layers.Activation('relu')(out)
    return out


def AttentionBlock(x, gating, inter_shape):
    """
    inherited from @DigitalSreeni
    attention block
    :params x: input tensor
    :params gating: gating unit
    :params inter_shape: number of features of the skip connection intermediate
    :return: gating feature map with the same dimension of the up layer feature map
    """
    
    # get the shape of the downsampling skip connection intermediate
    shape_x = keras.backend.int_shape(x)
    shape_g = keras.backend.int_shape(gating)

    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = keras.backend.int_shape(theta_x)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = keras.backend.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def AttResUNet(input_shape, 
               n_classes=1, 
               filter_num=64,
               filter_size=3,
               dropout_rate=0.0,
               up_sampling_size=2,
               activation='sigmoid',
               batch_norm=True):
    """
    inherited from @DigitalSreeni
    Residual UNet, with attention 
    :params n_classes: number of output classes
    :params filter_num: number of basic filters for the first layer
    :params filter_size: size of the convolutional filter
    :params dropout_rate: dropout rate
    :params batch_norm: batch normalization, default is True
    :params up_sampling_size: size of upsampling filters
    :params activation: activation function
    :return: gating feature map with the same dimension of the up layer feature map
    """
    
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # Down ResConv 1, double residual convolution + pooling
    conv_1 = ResConvBlock(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_1 = layers.MaxPooling2D(pool_size=(2,2))(conv_1)
    # Down ResConv 2, double residual convolution + pooling
    conv_2 = ResConvBlock(pool_1, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_2 = layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    # Down ResConv 3, double residual convolution + pooling
    conv_3 = ResConvBlock(pool_2, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_3 = layers.MaxPooling2D(pool_size=(2,2))(conv_3)
    # Down ResConv 4, double residual convolution + pooling
    conv_4 = ResConvBlock(pool_3, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_4 = layers.MaxPooling2D(pool_size=(2,2))(conv_4)
    # Down ResConv 5, convolution only
    conv_5 = ResConvBlock(pool_4, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers
    # Up AttRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_5_6 = gating_signal(conv_5, 8*filter_num, batch_norm)
    att_4_6 = AttentionBlock(conv_4, gating_5_6, 8*filter_num)
    up_6 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(conv_5)
    up_6 = layers.concatenate([up_6, att_4_6], axis=axis)
    up_conv_6 = ResConvBlock(up_6, filter_size, 8*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 7, attention gated concatenation + upsampling + double residual convolution
    gating_6_7 = gating_signal(up_conv_6, 4*filter_num, batch_norm)
    att_3_7 = AttentionBlock(conv_3, gating_6_7, 4*filter_num)
    up_7 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_6)
    up_7 = layers.concatenate([up_7, att_3_7], axis=axis)
    up_conv_7 = ResConvBlock(up_7, filter_size, 4*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 8, attention gated concatenation + upsampling + double residual convolution
    gating_7_8 = gating_signal(up_conv_7, 2*filter_num, batch_norm)
    att_2_8 = AttentionBlock(conv_2, gating_7_8, 2*filter_num)
    up_8 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_7)
    up_8 = layers.concatenate([up_8, att_2_8], axis=axis)
    up_conv_8 = ResConvBlock(up_8, filter_size, 2*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 9, attention gated concatenation + upsampling + double residual convolution
    gating_8_9 = gating_signal(up_conv_8, filter_num, batch_norm)
    att_1_9 = AttentionBlock(conv_1, gating_8_9, filter_num)
    up_9 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_8)
    up_9 = layers.concatenate([up_9, att_1_9], axis=axis)
    up_conv_9 = ResConvBlock(up_9, filter_size, filter_num, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(n_classes, kernel_size=(1,1))(up_conv_9)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation(activation)(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = keras.models.Model(inputs, conv_final, name="AttentionResUNet")
    return model


def Unet(input_shape, 
         n_classes=1, 
         filter_num=64,
         filter_size=3,
         dropout_rate=0.0,
         up_sampling_size=2,
         activation='sigmoid',
         batch_norm=True):
    """
    inherited from @DigitalSreeni
    Standard UNet, with attention 
    :params n_classes: number of output classes
    :params filter_num: number of basic filters for the first layer
    :params filter_size: size of the convolutional filter
    :params dropout_rate: dropout rate
    :params batch_norm: batch normalization, default is True
    :params up_sampling_size: size of upsampling filters
    :params activation: activation function
    :return: gating feature map with the same dimension of the up layer feature map
    """
    
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # Down ResConv 1, double residual convolution + pooling
    conv_1 = ConvBlock(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_1 = layers.MaxPooling2D(pool_size=(2,2))(conv_1)
    # Down ResConv 2, double residual convolution + pooling
    conv_2 = ConvBlock(pool_1, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_2 = layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    # Down ResConv 3, double residual convolution + pooling
    conv_3 = ConvBlock(pool_2, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_3 = layers.MaxPooling2D(pool_size=(2,2))(conv_3)
    # Down ResConv 4, double residual convolution + pooling
    conv_4 = ConvBlock(pool_3, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_4 = layers.MaxPooling2D(pool_size=(2,2))(conv_4)
    # Down ResConv 5, convolution only
    conv_5 = ConvBlock(pool_4, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers
    # Up AttRes 6, attention gated concatenation + upsampling + double residual convolution
    up_6 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(conv_5)
    up_6 = layers.concatenate([up_6, conv_4], axis=axis)
    up_conv_6 = ConvBlock(up_6, filter_size, 8*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 7, attention gated concatenation + upsampling + double residual convolution
    up_7 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_6)
    up_7 = layers.concatenate([up_7, conv_3], axis=axis)
    up_conv_7 = ConvBlock(up_7, filter_size, 4*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 8, attention gated concatenation + upsampling + double residual convolution
    up_8 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_7)
    up_8 = layers.concatenate([up_8, conv_2], axis=axis)
    up_conv_8 = ConvBlock(up_8, filter_size, 2*filter_num, dropout_rate, batch_norm)
    
    # Up AttRes 9, attention gated concatenation + upsampling + double residual convolution
    up_9 = layers.UpSampling2D(size=(up_sampling_size, up_sampling_size), data_format="channels_last")(up_conv_8)
    up_9 = layers.concatenate([up_9, conv_1], axis=axis)
    up_conv_9 = ConvBlock(up_9, filter_size, filter_num, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(n_classes, kernel_size=(1,1))(up_conv_9)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation(activation)(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = keras.models.Model(inputs, conv_final, name="UNet")
    return model