#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling , Normalization
from tensorflow.keras.layers import ZeroPadding2D , Conv2D , BatchNormalization , Activation , DepthwiseConv2D , GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape , Multiply , Dropout , Add , ZeroPadding2D , MaxPool2D , Dense
from tensorflow.keras.models import Model


# In[ ]:


def Stem_Block(input_layer , filter1):
    
    rescaling = Rescaling(scale=1./255. , name='rescaling')(input_layer)
    normalization = Normalization(axis=-1 , name='normalization')(rescaling)
    stem_conv_pad = ZeroPadding2D(padding = (1,1) , data_format = 'channels_last' , name = 'stem_conv_pad')(normalization)
    stem_conv = Conv2D(filter1,kernel_size = (3,3),activation='swish' , name='stem_conv' ,padding = 'same' , strides=(2, 2) , use_bias=False)(stem_conv_pad)
    stem_bn = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01, center=False, scale=False , name='stem_bn')(stem_conv)
    stem_activation = Activation('swish' , name='stem_activation')(stem_bn)
    
    return Model(input_layer,stem_activation)


# In[ ]:


def Sub_Blick1(model , num , filter1 , filter2 , filter3):
    input_layer = model.input
    connect_layer = model.output
    
    globals()['block'+num+'_dwconv'] = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' , data_format='channels_last', name='block'+num+'_dwconv' , use_bias=False)(connect_layer)
    globals()['block'+num+'_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01, center=False, scale=False , name = 'block'+num+'_bn')(globals()['block'+num+'_dwconv'])
    globals()['block'+num+'_activation'] = Activation('swish' , name='block'+num+'_activation')(globals()['block'+num+'_bn'])
    globals()['block'+num+'_se_squeeze'] = GlobalAveragePooling2D(data_format='channels_last' , name = 'block'+num+'_se_squeeze')(globals()['block'+num+'_activation'])
    globals()['block'+num+'_se_reshape'] = Reshape((1,1,-1) , name='block'+num+'_se_reshape')(globals()['block'+num+'_se_squeeze'])
    globals()['block'+num+'_se_reduce'] = Conv2D(filter1 , kernel_size = (1,1) , activation = 'swish' , name='block'+num+'_se_reduce' , padding = 'same' , use_bias=False)(globals()['block'+num+'_se_reshape'])
    globals()['block'+num+'_se_expand'] = Conv2D(filter2 , kernel_size = (1,1) , activation = 'sigmoid' , name='block'+num+'_se_expand' , padding = 'same' , use_bias=False)(globals()['block'+num+'_se_reduce'])
    globals()['block'+num+'_se_excite'] = Multiply(name = 'block'+num+'_se_excite')([globals()['block'+num+'_activation'], globals()['block'+num+'_se_expand']])
    globals()['block'+num+'_project_conv'] = Conv2D(filter3,kernel_size=(1,1),data_format = 'channels_last' , padding='same' , activation='swish' , strides = (1,1) ,use_bias=False , name = 'block'+num+'_project_conv')(globals()['block'+num+'_se_excite'])
    globals()['block'+num+'_project_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_project_bn')(globals()['block'+num+'_project_conv'])
    
    return Model(input_layer , globals()['block'+num+'_project_bn'])


# In[ ]:


def Sub_Block2_1(model , num , kernel , stride , filter1 , filter2 , filter3):
    input_layer = model.input
    connect_layer = model.output
    
    globals()['block'+num+'_expand_conv'] = Conv2D(filter1,kernel_size=kernel,data_format = 'channels_last' , padding='same' , activation='swish' , strides = stride , use_bias=False , name = 'block'+num+'_expand_conv')(connect_layer)
    globals()['block'+num+'_expand_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_expand_bn')(globals()['block'+num+'_expand_conv'])
    globals()['block'+num+'_expand_activation'] = Activation('swish' , name='block'+num+'_expand_activation')(globals()['block'+num+'_expand_bn'])
    globals()['block'+num+'_dwconv_pad'] = ZeroPadding2D(padding=1 , data_format = 'channels_last' , name = 'block'+num+'_dwconv_pad')(globals()['block'+num+'_expand_activation'])
    globals()['block'+num+'_dwconv'] = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' , data_format='channels_last', strides = (1,1), use_bias=False ,name='block'+num+'_dwconv' )(globals()['block'+num+'_dwconv_pad'])
    globals()['block'+num+'_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_bn')(globals()['block'+num+'_dwconv'])
    globals()['block'+num+'_activation'] = Activation('swish' , name='block'+num+'_activation')(globals()['block'+num+'_bn'])
    globals()['block'+num+'_se_squeeze'] = GlobalAveragePooling2D(data_format='channels_last' , name = 'block'+num+'_se_squeeze')(globals()['block'+num+'_activation'])
    globals()['block'+num+'_se_reshape'] = Reshape((1,1,-1) , name='block'+num+'_se_reshape')(globals()['block'+num+'_se_squeeze'])
    globals()['block'+num+'_se_reduce'] = Conv2D(filter2 , kernel_size = (1,1) , activation ='swish', padding='same',use_bias=False, name = 'block'+num+'_se_reduce')(globals()['block'+num+'_se_reshape'])
    globals()['block'+num+'_se_expand'] = Conv2D(filter1 , kernel_size = (1,1) , activation ='sigmoid', padding='same' ,use_bias=False, name = 'block'+num+'_se_expand')(globals()['block'+num+'_se_reduce'])
    globals()['block'+num+'_se_excite'] = Multiply(name='block'+num+'_se_excite')([globals()['block'+num+'_activation'],globals()['block'+num+'_se_expand']])
    globals()['block'+num+'_project_conv'] = Conv2D(filter3 , kernel_size=(1,1) , activation ='swish', padding='same',use_bias=False, name='block'+num+'_project_conv')(globals()['block'+num+'_se_excite'])
    globals()['block'+num+'_project_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_project_bn')(globals()['block'+num+'_project_conv'])
    
    return Model(input_layer , globals()['block'+num+'_project_bn'])


# In[ ]:


# Block5 and Block7 TOP sub block use
def Sub_Block2_2(model , num , kernel , stride , filter1 , filter2 , filter3):
    input_layer = model.input
    connect_layer = model.output
    
    globals()['block'+num+'_expand_conv'] = Conv2D(filter1,kernel_size=kernel,data_format = 'channels_last' , padding='same' ,use_bias=False, activation='swish' , strides = stride , name = 'block'+num+'_expand_conv')(connect_layer)
    globals()['block'+num+'_expand_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_expand_bn')(globals()['block'+num+'_expand_conv'])
    globals()['block'+num+'_expand_activation'] = Activation('swish' , name='block'+num+'_expand_activation')(globals()['block'+num+'_expand_bn'])
    globals()['block'+num+'_dwconv'] = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' , data_format='channels_last', strides = (1,1) ,use_bias=False,name='block'+num+'_dwconv' )(globals()['block'+num+'_expand_activation'])
    globals()['block'+num+'_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_bn')(globals()['block'+num+'_dwconv'])
    globals()['block'+num+'_activation'] = Activation('swish' , name='block'+num+'_activation')(globals()['block'+num+'_bn'])
    globals()['block'+num+'_se_squeeze'] = GlobalAveragePooling2D(data_format='channels_last' , name = 'block'+num+'_se_squeeze')(globals()['block'+num+'_activation'])
    globals()['block'+num+'_se_reshape'] = Reshape((1,1,-1) , name='block'+num+'_se_reshape')(globals()['block'+num+'_se_squeeze'])
    globals()['block'+num+'_se_reduce'] = Conv2D(filter2 , kernel_size = (1,1) , activation ='swish', padding='same',use_bias=False, name = 'block'+num+'_se_reduce')(globals()['block'+num+'_se_reshape'])
    globals()['block'+num+'_se_expand'] = Conv2D(filter1 , kernel_size = (1,1) , activation ='sigmoid', padding='same',use_bias=False, name = 'block'+num+'_se_expand')(globals()['block'+num+'_se_reduce'])
    globals()['block'+num+'_se_excite'] = Multiply(name='block'+num+'_se_excite')([globals()['block'+num+'_activation'],globals()['block'+num+'_se_expand']])
    globals()['block'+num+'_project_conv'] = Conv2D(filter3 , kernel_size=(1,1) , activation ='swish', padding='same',use_bias=False, name='block'+num+'_project_conv')(globals()['block'+num+'_se_excite'])
    globals()['block'+num+'_project_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_project_bn')(globals()['block'+num+'_project_conv'])
    
    return Model(input_layer , globals()['block'+num+'_project_bn'])


# In[ ]:


# use in Block 1's Sub Block3
def Sub_Block3_1(model , num , filter1 , filter2):
    input_layer = model.input
    connect_layer = model.output
    
    globals()['block'+num+'_dwconv'] = DepthwiseConv2D(kernel_size = (1,1),activation='swish' , padding = 'same' ,use_bias=False,data_format='channels_last', name='block'+num+'_dwconv' )(connect_layer)
    globals()['block'+num+'_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_bn')(globals()['block'+num+'_dwconv'])
    globals()['block'+num+'_activation'] = Activation('swish' , name='block'+num+'_activation')(globals()['block'+num+'_bn'])
    globals()['block'+num+'_se_squeeze'] = GlobalAveragePooling2D(data_format='channels_last' , name = 'block'+num+'_se_squeeze')(globals()['block'+num+'_activation'])
    globals()['block'+num+'_se_reshape'] = Reshape((1,1,-1) , name='block'+num+'_se_reshape')(globals()['block'+num+'_se_squeeze'])
    globals()['block'+num+'_se_reduce'] = Conv2D(filter1 , kernel_size = (1,1) , activation = 'swish',use_bias=False , name='block'+num+'_se_reduce' , padding = 'same')(globals()['block'+num+'_se_reshape'])
    globals()['block'+num+'_se_expand'] = Conv2D(filter2 , kernel_size = (1,1) , activation = 'sigmoid' ,use_bias=False, name='block'+num+'_se_expand' , padding = 'same')(globals()['block'+num+'_se_reduce'])
    globals()['block'+num+'_se_excite'] = Multiply(name = 'block'+num+'_se_excite')([globals()['block'+num+'_activation'] , globals()['block'+num+'_se_expand']])
    globals()['block'+num+'_project_conv'] = Conv2D(filter2,kernel_size=(1,1),data_format = 'channels_last' , padding='same' ,use_bias=False, activation='swish' , strides = (1,1) , name = 'block'+num+'_project_conv')(globals()['block'+num+'_se_excite'])
    globals()['block'+num+'_project_bn'] = BatchNormalization(axis=-1,  momentum=0.1, epsilon=0.01,center=False, scale=False  , name = 'block'+num+'_project_bn')(globals()['block'+num+'_project_conv'])
    globals()['block'+num+'_drop'] = Dropout(0.2 , name='block'+num+'_drop')(globals()['block'+num+'_project_bn'])
    globals()['block'+num+'_add'] = Add(name='block'+num+'_add')([globals()['block'+num+'_drop'] , connect_layer])
    
    return Model(input_layer , globals()['block'+num+'_add'])


# In[ ]:


# use other Block's Sub Block3
def Sub_Block3_2(model , num , filter1 , filter2 , filter3):
    input_layer = model.input
    connect_layer = model.output
    
    globals()['block'+num+'_expand_conv'] = Conv2D(filter1,kernel_size=(1,1),data_format='channels_last',padding='same',activation='swish',use_bias=False,strides=(1,1),name='block'+num+'_expand_conv')(connect_layer)
    globals()['block'+num+'_expand_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False ,name='block'+num+'_expand_bn')(globals()['block'+num+'_expand_conv'])
    globals()['block'+num+'_expand_activation'] = Activation('swish',name='block'+num+'_expand_activation')(globals()['block'+num+'_expand_bn'])
    globals()['block'+num+'_dwconv'] = DepthwiseConv2D(kernel_size=1,activation='swish',padding='same',data_format='channels_last',use_bias=False,name='block'+num+'_dwconv')(globals()['block'+num+'_expand_activation'])
    globals()['block'+num+'_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False ,name='block'+num+'_bn')(globals()['block'+num+'_dwconv'])
    globals()['block'+num+'_activation'] = Activation('swish',name='block'+num+'_activation')(globals()['block'+num+'_bn'])
    globals()['block'+num+'_se_squeeze'] = GlobalAveragePooling2D(data_format='channels_last',name='block'+num+'_se_squeeze')(globals()['block'+num+'_activation'])
    globals()['block'+num+'_se_reshape'] = Reshape((1,1,-1),name='block'+num+'_se_reshape')(globals()['block'+num+'_se_squeeze'])
    globals()['block'+num+'_se_reduce'] = Conv2D(filter2,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',use_bias=False,name='block'+num+'_se_reduce')(globals()['block'+num+'_se_reshape'])
    globals()['block'+num+'_se_expand'] = Conv2D(filter1,kernel_size=(1,1),activation='sigmoid',padding='same',data_format='channels_last',use_bias=False,name='block'+num+'_se_expand')(globals()['block'+num+'_se_reduce'])
    globals()['block'+num+'_se_excite'] = Multiply(name='block'+num+'_se_excite')([globals()['block'+num+'_activation'],globals()['block'+num+'_se_expand']])
    globals()['block'+num+'_project_conv'] = Conv2D(filter3,kernel_size=(1,1),activation='swish',padding='same',data_format='channels_last',use_bias=False,name='block'+num+'_project_conv')(globals()['block'+num+'_se_excite'])
    globals()['block'+num+'_project_bn'] = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False ,name='block'+num+'_project_bn')(globals()['block'+num+'_project_conv'])
    globals()['block'+num+'_drop'] = Dropout(0.2 , name='block'+num+'_drop')(globals()['block'+num+'_project_bn'])
    globals()['block'+num+'_add'] = Add(name='block'+num+'_add')([globals()['block'+num+'_drop'] , connect_layer])
    
    return Model(input_layer , globals()['block'+num+'_add'])


# In[ ]:


def Final_Block(model , filter1):
    input_layer = model.input
    connect_layer = model.output
    
    top_conv = Conv2D(filter1 , (1,1),activation='swish',padding='same',use_bias=False,name='top_conv')(connect_layer)
    top_bn = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.01,center=False, scale=False ,name='top_bn')(top_conv)
    top_activation = Activation('swish',name='top_activation')(top_bn)
#     top_dopout = Dropout(0.2)(top_activation)
#     top_se_squeeze = GlobalAveragePooling2D(name='top_se_squeeze')(top_dopout)
#     top_dense = Dense(8, activation='softmax', name='softmax')(top_se_squeeze)
    
    return Model(input_layer , top_activation)

