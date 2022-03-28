#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling , Normalization
from tensorflow.keras.layers import ZeroPadding2D , Conv2D , BatchNormalization , Activation , DepthwiseConv2D , GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape , Multiply , Dropout , Add , ZeroPadding2D , MaxPool2D , Dense
from tensorflow.keras.models import Model


# In[ ]:


def Evolved_Model(input_layer):
    
    rescaling = Rescaling(scale=1./255. , name='rescaling')(input_layer)
    normalization = Normalization(axis=-1 , name='normalization')(rescaling)
    stem_conv_pad = ZeroPadding2D(padding = (1,1) , data_format = 'channels_last' , name = 'stem_conv_pad')(normalization)
    stem_conv = Conv2D(16,kernel_size = (3,3),activation='relu' , name='stem_conv' ,padding = 'valid' , strides=(2, 2))(stem_conv_pad)
    stem_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name='stem_bn')(stem_conv)
    stem_activation = Activation('swish' , name='stem_activation')(stem_bn)
    # Block 1

    ## Sub Block1
    block1a_dwconv = DepthwiseConv2D(kernel_size = (3,3),strides=(1, 1),activation='swish' , padding = 'same' , data_format='channels_last', name='block1a_dwconv' )(stem_activation)
    block1a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1a_bn')(block1a_dwconv)
    block1a_activation = Activation('swish' , name='block1a_activation')(block1a_bn)
    block1a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block1a_se_squeeze')(block1a_activation)
    block1a_se_reshape = Reshape((1,1,-1) , name='block1a_se_reshape')(block1a_se_squeeze)
    block1a_se_reduce = Conv2D(4 , kernel_size = (1,1) , activation = 'swish' , name='block1a_se_reduce' , padding = 'same')(block1a_se_reshape)
    block1a_se_expand = Conv2D(16 , kernel_size = (1,1) , activation = 'sigmoid' , name='block1a_se_expand' , padding = 'same')(block1a_se_reduce)
    block1a_se_excite = Multiply(name = 'block1a_se_excite')([block1a_activation, block1a_se_expand])
    block1a_project_conv = Conv2D(8,kernel_size=(1,1),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (1,1) , name = 'block1a_project_conv')(block1a_se_excite)
    block1a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1a_project_bn')(block1a_project_conv)

    ## Sub Block3_1
    block1b_dwconv = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' , data_format='channels_last', name='block1b_dwconv' )(block1a_project_bn)
    block1b_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1b_bn')(block1b_dwconv)
    block1b_activation = Activation('swish' , name='block1b_activation')(block1b_bn)
    block1b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block1b_se_squeeze')(block1b_activation)
    block1b_se_reshape = Reshape((1,1,-1) , name='block1b_se_reshape')(block1b_se_squeeze)
    block1b_se_reduce = Conv2D(2 , kernel_size = (1,1) , activation = 'swish' , name='block1b_se_reduce' , padding = 'same')(block1b_se_reshape)
    block1b_se_expand = Conv2D(8 , kernel_size = (1,1) , activation = 'sigmoid' , name='block1b_se_expand' , padding = 'same')(block1b_se_reduce)
    block1b_se_excite = Multiply(name = 'block1b_se_excite')([block1b_activation , block1b_se_expand])
    block1b_project_conv = Conv2D(8,kernel_size=(1,1),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (1,1) , name = 'block1b_project_conv')(block1b_se_excite)
    block1b_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1b_project_bn')(block1b_project_conv)
    block1b_drop = Dropout(0.2 , name='block1b_drop')(block1b_project_bn)
    block1b_add = Add(name='block1b_add')([block1b_drop , block1a_project_bn])

    ## Sub block3_1
    block1c_dwconv = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' , data_format='channels_last', name='block1c_dwconv' )(block1b_add)
    block1c_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1c_bn')(block1c_dwconv)
    block1c_activation = Activation('swish' , name='block1c_activation')(block1c_bn)
    block1c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block1c_se_squeeze')(block1c_activation)
    block1c_se_reshape = Reshape((1,1,-1) , name='block1c_se_reshape')(block1c_se_squeeze)
    block1c_se_reduce = Conv2D(2 , kernel_size = (1,1) , activation = 'swish' , name='block1c_se_reduce' , padding = 'same')(block1c_se_reshape)
    block1c_se_expand = Conv2D(8 , kernel_size = (1,1) , activation = 'sigmoid' , name='block1c_se_expand' , padding = 'same')(block1c_se_reduce)
    block1c_se_excite = Multiply(name = 'block1c_se_excite')([block1c_activation , block1c_se_expand])
    block1c_project_conv = Conv2D(8,kernel_size=(1,1),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (1,1) , name = 'block1c_project_conv')(block1c_se_excite)
    block1c_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block1c_project_bn')(block1c_project_conv)
    block1c_drop = Dropout(0.2 , name='block1c_drop')(block1c_project_bn)
    block1c_add = Add(name='block1c_add')([block1c_drop , block1b_add])
    
    # Block 2
    ## Sub Block2_1
    block2a_expand_conv = Conv2D(48,kernel_size=(3,3),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (2,2) , name = 'block2a_expand_conv')(block1c_add)
    block2a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block2a_expand_bn')(block2a_expand_conv)
    block2a_expand_activation = Activation('swish' , name='block2a_expand_activation')(block2a_expand_bn)
    block2a_dwconv_pad = ZeroPadding2D(padding=1 , data_format = 'channels_last' , name = 'block2a_dwconv_pad')(block2a_expand_activation)
    block2a_dwconv = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'valid' , data_format='channels_last' ,strides = (1,1),name='block2a_dwconv' )(block2a_dwconv_pad)
    block2a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block2a_bn')(block2a_dwconv)
    block2a_activation = Activation('swish' , name='block2a_activation')(block2a_bn)
    block2a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block2a_se_squeeze')(block2a_activation)
    block2a_se_reshape = Reshape((1,1,-1) , name='block2a_se_reshape')(block2a_se_squeeze)
    block2a_se_reduce = Conv2D(2 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block2a_se_reduce')(block2a_se_reshape)
    block2a_se_expand = Conv2D(48 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block2a_se_expand')(block2a_se_reduce)
    block2a_se_excite = Multiply(name='block2a_se_excite')([block2a_activation,block2a_se_expand])
    block2a_project_conv = Conv2D(12 , kernel_size=(1,1) , activation ='swish', padding='same', name='block2a_project_conv')(block2a_se_excite)
    block2a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block2a_project_bn')(block2a_project_conv)


    ## Sub Block3_2
    block2b_expand_conv = Conv2D(72,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block2b_expand_conv')(block2a_project_bn)
    block2b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2b_expand_bn')(block2b_expand_conv)
    block2b_expand_activation = Activation('swish',name='block2b_expand_activation')(block2b_expand_bn)
    block2b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block2b_dwconv')(block2b_expand_activation)
    block2b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2b_bn')(block2b_dwconv)
    block2b_activation = Activation('swish',name='block2b_activation')(block2b_bn)
    block2b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block2b_se_squeeze')(block2b_activation)
    block2b_se_reshape = Reshape((1,1,-1),name='block2b_se_reshape')(block2b_se_squeeze)
    block2b_se_reduce = Conv2D(3,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block2b_se_reduce')(block2b_se_reshape)
    block2b_se_expand = Conv2D(72,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block2b_se_expand')(block2b_se_reduce)
    block2b_se_excite = Multiply(name='block2b_se_excite')([block2b_activation,block2b_se_expand])
    block2b_project_conv = Conv2D(12,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block2b_project_conv')(block2b_se_excite)
    block2b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2b_project_bn')(block2b_project_conv)
    block2b_drop = Dropout(0.2 , name='block2b_drop')(block2b_project_bn)
    block2b_add = Add(name='block2b_add')([block2b_drop,block2a_project_bn])


    ## Sub Block3_2
    block2c_expand_conv = Conv2D(72,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block2c_expand_conv')(block2b_project_bn)
    block2c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2c_expand_bn')(block2c_expand_conv)
    block2c_expand_activation = Activation('swish',name='block2c_expand_activation')(block2c_expand_bn)
    block2c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block2c_dwconv')(block2c_expand_activation)
    block2c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2c_bn')(block2c_dwconv)
    block2c_activation = Activation('swish',name='block2c_activation')(block2c_bn)
    block2c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block2c_se_squeeze')(block2c_activation)
    block2c_se_reshape = Reshape((1,1,-1),name='block2c_se_reshape')(block2c_se_squeeze)
    block2c_se_reduce = Conv2D(3,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block2c_se_reduce')(block2c_se_reshape)
    block2c_se_expand = Conv2D(72,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block2c_se_expand')(block2c_se_reduce)
    block2c_se_excite = Multiply(name='block2c_se_excite')([block2c_activation,block2c_se_expand])
    block2c_project_conv = Conv2D(12,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block2c_project_conv')(block2c_se_excite)
    block2c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block2c_project_bn')(block2c_project_conv)
    block2c_drop = Dropout(0.2 , name='block2c_drop')(block2c_project_bn)
    block2c_add = Add(name='block2c_add')([block2c_drop,block2b_project_bn])
    
    # Block3
    ## Sub Block2_1
    block3a_expand_conv = Conv2D(72,kernel_size=(5,5),data_format = 'channels_last',padding='same',activation='swish',strides = (2,2),name = 'block3a_expand_conv')(block2c_add)
    block3a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block3a_expand_bn')(block3a_expand_conv)
    block3a_expand_activation = Activation('swish' , name='block3a_expand_activation')(block3a_expand_bn)
    block3a_dwconv_pad = ZeroPadding2D(padding=1 , data_format = 'channels_last' , name = 'block3a_dwconv_pad')(block3a_expand_activation)
    block3a_dwconv = DepthwiseConv2D(kernel_size = 1,activation='swish' , padding = 'valid' , data_format='channels_last', strides = (1,1) ,name='block3a_dwconv' )(block3a_dwconv_pad)
    block3a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block3a_bn')(block3a_dwconv)
    block3a_activation = Activation('swish' , name='block3a_activation')(block3a_bn)
    block3a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block3a_se_squeeze')(block3a_activation)
    block3a_se_reshape = Reshape((1,1,-1) , name='block3a_se_reshape')(block3a_se_squeeze)
    block3a_se_reduce = Conv2D(3 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block3a_se_reduce')(block3a_se_reshape)
    block3a_se_expand = Conv2D(72 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block3a_se_expand')(block3a_se_reduce)
    block3a_se_excite = Multiply(name='block3a_se_excite')([block3a_activation,block3a_se_expand])
    block3a_project_conv = Conv2D(20 , kernel_size=(1,1) , activation ='swish', padding='same', name='block3a_project_conv')(block3a_se_excite)
    block3a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block3a_project_bn')(block3a_project_conv)


    ## Sub Block3_2
    block3b_expand_conv = Conv2D(120,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block3b_expand_conv')(block3a_project_bn)
    block3b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3b_expand_bn')(block3b_expand_conv)
    block3b_expand_activation = Activation('swish',name='block3b_expand_activation')(block3b_expand_bn)
    block3b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='valid',data_format='channels_last',name='block3b_dwconv')(block3b_expand_activation)
    block3b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3b_bn')(block3b_dwconv)
    block3b_activation = Activation('swish',name='block3b_activation')(block3b_bn)
    block3b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block3b_se_squeeze')(block3b_activation)
    block3b_se_reshape = Reshape((1,1,-1),name='block3b_se_reshape')(block3b_se_squeeze)
    block3b_se_reduce = Conv2D(5,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block3b_se_reduce')(block3b_se_reshape)
    block3b_se_expand = Conv2D(120,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block3b_se_expand')(block3b_se_reduce)
    block3b_se_excite = Multiply(name='block3b_se_excite')([block3b_activation,block3b_se_expand])
    block3b_project_conv = Conv2D(20,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block3b_project_conv')(block3b_se_excite)
    block3b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3b_project_bn')(block3b_project_conv)
    block3b_drop = Dropout(0.2 , name='block3b_drop')(block3b_project_bn)
    block3b_add = Add(name='block3b_add')([block3b_drop,block3a_project_bn])


    ## Sub Block3_2
    block3c_expand_conv = Conv2D(120,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block3c_expand_conv')(block3b_project_bn)
    block3c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3c_expand_bn')(block3c_expand_conv)
    block3c_expand_activation = Activation('swish',name='block3c_expand_activation')(block3c_expand_bn)
    block3c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block3c_dwconv')(block3c_expand_activation)
    block3c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3c_bn')(block3c_dwconv)
    block3c_activation = Activation('swish',name='block3c_activation')(block3c_bn)
    block3c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block3c_se_squeeze')(block3c_activation)
    block3c_se_reshape = Reshape((1,1,-1),name='block3c_se_reshape')(block3c_se_squeeze)
    block3c_se_reduce = Conv2D(5,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block3c_se_reduce')(block3c_se_reshape)
    block3c_se_expand = Conv2D(120,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block3c_se_expand')(block3c_se_reduce)
    block3c_se_excite = Multiply(name='block3c_se_excite')([block3c_activation,block3c_se_expand])
    block3c_project_conv = Conv2D(20,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block3c_project_conv')(block3c_se_excite)
    block3c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block3c_project_bn')(block3c_project_conv)
    block3c_drop = Dropout(0.2 , name='block3c_drop')(block3c_project_bn)
    block3c_add = Add(name='block3c_add')([block3c_drop,block3b_project_bn])
    
    # Block4
    ## Sub Block2_1
    block4a_expand_conv = Conv2D(120,kernel_size=(3,3),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (2,2) , name = 'block4a_expand_conv')(block3c_add)
    block4a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block4a_expand_bn')(block4a_expand_conv)
    block4a_expand_activation = Activation('swish' , name='block4a_expand_activation')(block4a_expand_bn)
    block4a_dwconv_pad = ZeroPadding2D(padding=1 , data_format = 'channels_last' , name = 'block4a_dwconv_pad')(block4a_expand_activation)
    block4a_dwconv = DepthwiseConv2D(kernel_size = 1,activation='swish' , padding = 'valid' , data_format='channels_last', strides = (1,1) ,name='block4a_dwconv' )(block4a_dwconv_pad)
    block4a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block4a_bn')(block4a_dwconv)
    block4a_activation = Activation('swish' , name='block4a_activation')(block4a_bn)
    block4a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block4a_se_squeeze')(block4a_activation)
    block4a_se_reshape = Reshape((1,1,-1) , name='block4a_se_reshape')(block4a_se_squeeze)
    block4a_se_reduce = Conv2D(5 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block4a_se_reduce')(block4a_se_reshape)
    block4a_se_expand = Conv2D(120 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block4a_se_expand')(block4a_se_reduce)
    block4a_se_excite = Multiply(name='block4a_se_excite')([block4a_activation,block4a_se_expand])
    block4a_project_conv = Conv2D(40 , kernel_size=(1,1) , activation ='swish', padding='same', name='block4a_project_conv')(block4a_se_excite)
    block4a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block4a_project_bn')(block4a_project_conv)


    ## Sub Block3_2
    block4b_expand_conv = Conv2D(240,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block4b_expand_conv')(block4a_project_bn)
    block4b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4b_expand_bn')(block4b_expand_conv)
    block4b_expand_activation = Activation('swish',name='block4b_expand_activation')(block4b_expand_bn)
    block4b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block4b_dwconv')(block4b_expand_activation)
    block4b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4b_bn')(block4b_dwconv)
    block4b_activation = Activation('swish',name='block4b_activation')(block4b_bn)
    block4b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block4b_se_squeeze')(block4b_activation)
    block4b_se_reshape = Reshape((1,1,-1),name='block4b_se_reshape')(block4b_se_squeeze)
    block4b_se_reduce = Conv2D(10,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block4b_se_reduce')(block4b_se_reshape)
    block4b_se_expand = Conv2D(240,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block4b_se_expand')(block4b_se_reduce)
    block4b_se_excite = Multiply(name='block4b_se_excite')([block4b_activation,block4b_se_expand])
    block4b_project_conv = Conv2D(40,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block4b_project_conv')(block4b_se_excite)
    block4b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4b_project_bn')(block4b_project_conv)
    block4b_drop = Dropout(0.2 , name='block4b_drop')(block4b_project_bn)
    block4b_add = Add(name='block4b_add')([block4b_drop,block4a_project_bn])


    ## Sub Block3_2
    block4c_expand_conv = Conv2D(240,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block4c_expand_conv')(block4b_project_bn)
    block4c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4c_expand_bn')(block4c_expand_conv)
    block4c_expand_activation = Activation('swish',name='block4c_expand_activation')(block4c_expand_bn)
    block4c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block4c_dwconv')(block4c_expand_activation)
    block4c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4c_bn')(block4c_dwconv)
    block4c_activation = Activation('swish',name='block4c_activation')(block4c_bn)
    block4c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block4c_se_squeeze')(block4c_activation)
    block4c_se_reshape = Reshape((1,1,-1),name='block4c_se_reshape')(block4c_se_squeeze)
    block4c_se_reduce = Conv2D(10,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block4c_se_reduce')(block4c_se_reshape)
    block4c_se_expand = Conv2D(240,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block4c_se_expand')(block4c_se_reduce)
    block4c_se_excite = Multiply(name='block4c_se_excite')([block4c_activation,block4c_se_expand])
    block4c_project_conv = Conv2D(40,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block4c_project_conv')(block4c_se_excite)
    block4c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block4c_project_bn')(block4c_project_conv)
    block4c_drop = Dropout(0.2 , name='block4c_drop')(block4c_project_bn)
    block4c_add = Add(name='block4c_add')([block4c_drop,block4b_project_bn])
    
    # Block5
    ## Sub Block2_2
    block5a_expand_conv = Conv2D(240,kernel_size=(5,5),strides=(1,1),data_format = 'channels_last' , padding='same' , activation='swish' , name = 'block5a_expand_conv')(block4c_add)
    block5a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block5a_expand_bn')(block5a_expand_conv)
    block5a_expand_activation = Activation('swish' , name='block5a_expand_activation')(block5a_expand_bn)
    block5a_dwconv = DepthwiseConv2D(kernel_size = 1,activation='swish' , padding = 'valid' , data_format='channels_last', strides = (1,1) ,name='block5a_dwconv' )(block5a_expand_activation)
    block5a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block5a_bn')(block5a_dwconv)
    block5a_activation = Activation('swish' , name='block5a_activation')(block5a_bn)
    block5a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block5a_se_squeeze')(block5a_activation)
    block5a_se_reshape = Reshape((1,1,-1) , name='block5a_se_reshape')(block5a_se_squeeze)
    block5a_se_reduce = Conv2D(10 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block5a_se_reduce')(block5a_se_reshape)
    block5a_se_expand = Conv2D(240 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block5a_se_expand')(block5a_se_reduce)
    block5a_se_excite = Multiply(name='block5a_se_excite')([block5a_activation,block5a_se_expand])
    block5a_project_conv = Conv2D(56 , kernel_size=(1,1) , activation ='swish', padding='same', name='block5a_project_conv')(block5a_se_excite)
    block5a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block5a_project_bn')(block5a_project_conv)


    ## Sub Block3_2
    block5b_expand_conv = Conv2D(336,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block5b_expand_conv')(block5a_project_bn)
    block5b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5b_expand_bn')(block5b_expand_conv)
    block5b_expand_activation = Activation('swish',name='block5b_expand_activation')(block5b_expand_bn)
    block5b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block5b_dwconv')(block5b_expand_activation)
    block5b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5b_bn')(block5b_dwconv)
    block5b_activation = Activation('swish',name='block5b_activation')(block5b_bn)
    block5b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block5b_se_squeeze')(block5b_activation)
    block5b_se_reshape = Reshape((1,1,-1),name='block5b_se_reshape')(block5b_se_squeeze)
    block5b_se_reduce = Conv2D(14,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block5b_se_reduce')(block5b_se_reshape)
    block5b_se_expand = Conv2D(336,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block5b_se_expand')(block5b_se_reduce)
    block5b_se_excite = Multiply(name='block5b_se_excite')([block5b_activation,block5b_se_expand])
    block5b_project_conv = Conv2D(56,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block5b_project_conv')(block5b_se_excite)
    block5b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5b_project_bn')(block5b_project_conv)
    block5b_drop = Dropout(0.2 , name='block5b_drop')(block5b_project_bn)
    block5b_add = Add(name='block5b_add')([block5b_drop,block5a_project_bn])


    ## Sub Block3_2
    block5c_expand_conv = Conv2D(336,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block5c_expand_conv')(block5b_project_bn)
    block5c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5c_expand_bn')(block5c_expand_conv)
    block5c_expand_activation = Activation('swish',name='block5c_expand_activation')(block5c_expand_bn)
    block5c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block5c_dwconv')(block5c_expand_activation)
    block5c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5c_bn')(block5c_dwconv)
    block5c_activation = Activation('swish',name='block5c_activation')(block5c_bn)
    block5c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block5c_se_squeeze')(block5c_activation)
    block5c_se_reshape = Reshape((1,1,-1),name='block5c_se_reshape')(block5c_se_squeeze)
    block5c_se_reduce = Conv2D(14,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block5c_se_reduce')(block5c_se_reshape)
    block5c_se_expand = Conv2D(336,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block5c_se_expand')(block5c_se_reduce)
    block5c_se_excite = Multiply(name='block5c_se_excite')([block5c_activation,block5c_se_expand])
    block5c_project_conv = Conv2D(56,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block5c_project_conv')(block5c_se_excite)
    block5c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block5c_project_bn')(block5c_project_conv)
    block5c_drop = Dropout(0.2 , name='block5c_drop')(block5c_project_bn)
    block5c_add = Add(name='block5c_add')([block5c_drop,block5b_project_bn])
    
    # Block6
    ## Sub Block2_1
    block6a_expand_conv = Conv2D(336,kernel_size=(5,5),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (2,2) , name = 'block6a_expand_conv')(block5c_add)
    block6a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block6a_expand_bn')(block6a_expand_conv)
    block6a_expand_activation = Activation('swish' , name='block6a_expand_activation')(block6a_expand_bn)
    block6a_dwconv_pad = ZeroPadding2D(padding=1 , data_format = 'channels_last' , name = 'block6a_dwconv_pad')(block6a_expand_activation)
    block6a_dwconv = DepthwiseConv2D(kernel_size = 1,activation='swish' , padding = 'valid' , data_format='channels_last', strides = (1,1) ,name='block6a_dwconv' )(block6a_dwconv_pad)
    block6a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block6a_bn')(block6a_dwconv)
    block6a_activation = Activation('swish' , name='block6a_activation')(block6a_bn)
    block6a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block6a_se_squeeze')(block6a_activation)
    block6a_se_reshape = Reshape((1,1,-1) , name='block6a_se_reshape')(block6a_se_squeeze)
    block6a_se_reduce = Conv2D(14 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block6a_se_reduce')(block6a_se_reshape)
    block6a_se_expand = Conv2D(336 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block6a_se_expand')(block6a_se_reduce)
    block6a_se_excite = Multiply(name='block6a_se_excite')([block6a_activation,block6a_se_expand])
    block6a_project_conv = Conv2D(96 , kernel_size=(1,1) , activation ='swish', padding='same', name='block6a_project_conv')(block6a_se_excite)
    block6a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block6a_project_bn')(block6a_project_conv)


    ## Sub Block3_2
    block6b_expand_conv = Conv2D(576,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block6b_expand_conv')(block6a_project_bn)
    block6b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6b_expand_bn')(block6b_expand_conv)
    block6b_expand_activation = Activation('swish',name='block6b_expand_activation')(block6b_expand_bn)
    block6b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block6b_dwconv')(block6b_expand_activation)
    block6b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6b_bn')(block6b_dwconv)
    block6b_activation = Activation('swish',name='block6b_activation')(block6b_bn)
    block6b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block6b_se_squeeze')(block6b_activation)
    block6b_se_reshape = Reshape((1,1,-1),name='block6b_se_reshape')(block6b_se_squeeze)
    block6b_se_reduce = Conv2D(24,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block6b_se_reduce')(block6b_se_reshape)
    block6b_se_expand = Conv2D(576,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block6b_se_expand')(block6b_se_reduce)
    block6b_se_excite = Multiply(name='block6b_se_excite')([block6b_activation,block6b_se_expand])
    block6b_project_conv = Conv2D(96,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block6b_project_conv')(block6b_se_excite)
    block6b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6b_project_bn')(block6b_project_conv)
    block6b_drop = Dropout(0.2 , name='block6b_drop')(block6b_project_bn)
    block6b_add = Add(name='block6b_add')([block6b_drop,block6a_project_bn])


    ## Sub Block3_2
    block6c_expand_conv = Conv2D(576,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block6c_expand_conv')(block6b_project_bn)
    block6c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6c_expand_bn')(block6c_expand_conv)
    block6c_expand_activation = Activation('swish',name='block6c_expand_activation')(block6c_expand_bn)
    block6c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block6c_dwconv')(block6c_expand_activation)
    block6c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6c_bn')(block6c_dwconv)
    block6c_activation = Activation('swish',name='block6c_activation')(block6c_bn)
    block6c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block6c_se_squeeze')(block6c_activation)
    block6c_se_reshape = Reshape((1,1,-1),name='block6c_se_reshape')(block6c_se_squeeze)
    block6c_se_reduce = Conv2D(24,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block6c_se_reduce')(block6c_se_reshape)
    block6c_se_expand = Conv2D(576,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block6c_se_expand')(block6c_se_reduce)
    block6c_se_excite = Multiply(name='block6c_se_excite')([block6c_activation,block6c_se_expand])
    block6c_project_conv = Conv2D(96,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block6c_project_conv')(block6c_se_excite)
    block6c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block6c_project_bn')(block6c_project_conv)
    block6c_drop = Dropout(0.2 , name='block6c_drop')(block6c_project_bn)
    block6c_add = Add(name='block6c_add')([block6c_drop,block6b_project_bn])
    
    # Block7
    ## Sub Block2_2
    block7a_expand_conv = Conv2D(576,kernel_size=(3,3),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (1,1) , name = 'block7a_expand_conv')(block6c_add)
    block7a_expand_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block7a_expand_bn')(block7a_expand_conv)
    block7a_expand_activation = Activation('swish' , name='block7a_expand_activation')(block7a_expand_bn)
    block7a_dwconv = DepthwiseConv2D(kernel_size = 1,activation='swish' , padding = 'valid' , data_format='channels_last', strides = (1,1) ,name='block7a_dwconv' )(block7a_expand_activation)
    block7a_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block7a_bn')(block7a_dwconv)
    block7a_activation = Activation('swish' , name='block7a_activation')(block7a_bn)
    block7a_se_squeeze = GlobalAveragePooling2D(data_format='channels_last' , name = 'block7a_se_squeeze')(block7a_activation)
    block7a_se_reshape = Reshape((1,1,-1) , name='block7a_se_reshape')(block7a_se_squeeze)
    block7a_se_reduce = Conv2D(24 , kernel_size = (1,1) , activation ='swish', padding='same', name = 'block7a_se_reduce')(block7a_se_reshape)
    block7a_se_expand = Conv2D(576 , kernel_size = (1,1) , activation ='sigmoid', padding='same', name = 'block7a_se_expand')(block7a_se_reduce)
    block7a_se_excite = Multiply(name='block7a_se_excite')([block7a_activation,block7a_se_expand])
    block7a_project_conv = Conv2D(160 , kernel_size=(1,1) , activation ='swish', padding='same', name='block7a_project_conv')(block7a_se_excite)
    block7a_project_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 , name = 'block7a_project_bn')(block7a_project_conv)


    ## Sub Block3_2
    block7b_expand_conv = Conv2D(960,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block7b_expand_conv')(block7a_project_bn)
    block7b_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7b_expand_bn')(block7b_expand_conv)
    block7b_expand_activation = Activation('swish',name='block7b_expand_activation')(block7b_expand_bn)
    block7b_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block7b_dwconv')(block7b_expand_activation)
    block7b_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7b_bn')(block7b_dwconv)
    block7b_activation = Activation('swish',name='block7b_activation')(block7b_bn)
    block7b_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block7b_se_squeeze')(block7b_activation)
    block7b_se_reshape = Reshape((1,1,-1),name='block7b_se_reshape')(block7b_se_squeeze)
    block7b_se_reduce = Conv2D(40,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block7b_se_reduce')(block7b_se_reshape)
    block7b_se_expand = Conv2D(960,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block7b_se_expand')(block7b_se_reduce)
    block7b_se_excite = Multiply(name='block7b_se_excite')([block7b_activation,block7b_se_expand])
    block7b_project_conv = Conv2D(160,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block7b_project_conv')(block7b_se_excite)
    block7b_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7b_project_bn')(block7b_project_conv)
    block7b_drop = Dropout(0.2 , name='block7b_drop')(block7b_project_bn)
    block7b_add = Add(name='block7b_add')([block7b_drop,block7a_project_bn])


    ## Sub Block3_2
    block7c_expand_conv = Conv2D(960,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',strides=(1,1),name='block7c_expand_conv')(block7b_project_bn)
    block7c_expand_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7c_expand_bn')(block7c_expand_conv)
    block7c_expand_activation = Activation('swish',name='block7c_expand_activation')(block7c_expand_bn)
    block7c_dwconv = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',name='block7c_dwconv')(block7c_expand_activation)
    block7c_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7c_bn')(block7c_dwconv)
    block7c_activation = Activation('swish',name='block7c_activation')(block7c_bn)
    block7c_se_squeeze = GlobalAveragePooling2D(data_format='channels_last',name='block7c_se_squeeze')(block7c_activation)
    block7c_se_reshape = Reshape((1,1,-1),name='block7c_se_reshape')(block7c_se_squeeze)
    block7c_se_reduce = Conv2D(40,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block7c_se_reduce')(block7c_se_reshape)
    block7c_se_expand = Conv2D(960,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',name='block7c_se_expand')(block7c_se_reduce)
    block7c_se_excite = Multiply(name='block7c_se_excite')([block7c_activation,block7c_se_expand])
    block7c_project_conv = Conv2D(160,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',name='block7c_project_conv')(block7c_se_excite)
    block7c_project_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='block7c_project_bn')(block7c_project_conv)
    block7c_drop = Dropout(0.2 , name='block7c_drop')(block7c_project_bn)
    block7c_add = Add(name='block7c_add')([block7c_drop,block7b_project_bn])

    # Final top
    top_conv = Conv2D(640 , (1,1),activation='relu',padding='same',name='top_conv')(block7c_add)
    top_bn = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,name='top_bn')(top_conv)
    top_activation = Activation('relu',name='top_activation')(top_bn)

    return Model(input_layer , top_activation)

