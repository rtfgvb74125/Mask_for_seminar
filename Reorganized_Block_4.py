#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from tensorflow.keras.layers import Input
import EfficientNet_Block as ef_block
def Evolved_41(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 64)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 16 , 64 , 32)
    model = ef_block.Sub_Block3_1(model , '1b' , 8 , 32)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),192,8,48)
    model = ef_block.Sub_Block3_2(model,'2b',288,12,48)
    model = ef_block.Sub_Block3_2(model,'2c',288,12,48)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),288,12,80)
    model = ef_block.Sub_Block3_2(model,'3b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3c',480,20,80)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),480,20,160)
    model = ef_block.Sub_Block3_2(model,'4b',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4c',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4d',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4e',960,40,160)
        
#     # Final
    
    model = ef_block.Final_Block(model ,224)
    
    return model





def Evolved_42(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)
    model = ef_block.Sub_Block3_1(model , '1b' , 4 , 16)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
    model = ef_block.Sub_Block3_2(model,'2c',144,6,24)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)
    model = ef_block.Sub_Block3_2(model,'3c',240,10,40)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4d',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4e',480,20,80)

    
#     # Final
    
    model = ef_block.Final_Block(model ,112)
    
    return model

def Evolved_43(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)
    model = ef_block.Sub_Block3_1(model , '1b' , 4 , 16)
    model = ef_block.Sub_Block3_1(model , '1c' , 4 , 16)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
    model = ef_block.Sub_Block3_2(model,'2c',144,6,24)
    model = ef_block.Sub_Block3_2(model,'2d',144,6,24)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)
    model = ef_block.Sub_Block3_2(model,'3c',240,10,40)
    model = ef_block.Sub_Block3_2(model,'3d',240,10,40)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4d',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4e',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4f',480,20,80)
    
#     # Final
    
    model = ef_block.Final_Block(model ,112)
    
    return model
def Evolved_44(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 64)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 16 , 64 , 32)
    model = ef_block.Sub_Block3_1(model , '1b' , 8 , 32)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),192,8,48)
    model = ef_block.Sub_Block3_2(model,'2b',288,12,48)
    model = ef_block.Sub_Block3_2(model,'2c',288,12,48)
    model = ef_block.Sub_Block3_2(model,'2d',288,12,48)

    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),288,12,80)
    model = ef_block.Sub_Block3_2(model,'3b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3c',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3d',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3e',480,20,80)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),480,20,160)
    model = ef_block.Sub_Block3_2(model,'4b',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4c',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4d',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4e',960,40,160)
  
    # Final
    
    model = ef_block.Final_Block(model ,224)
    
    return model

def Evolved_45(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 16)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 4 , 16 , 8)
    model = ef_block.Sub_Block3_1(model , '1b' , 2 , 8)
#     model = ef_block.Sub_Block3_1(model , '1c' , 2 , 8)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),48,2,12)
    model = ef_block.Sub_Block3_2(model,'2b',72,3,12)
    model = ef_block.Sub_Block3_2(model,'2c',72,3,12)
    model = ef_block.Sub_Block3_2(model,'2d',72,3,12)
#     model = ef_block.Sub_Block3_2(model,'2e',72,3,12)
#     model = ef_block.Sub_Block3_2(model,'2f',72,3,12)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),72,3,20)
    model = ef_block.Sub_Block3_2(model,'3b',120,5,20)
    model = ef_block.Sub_Block3_2(model,'3c',120,5,20)
    model = ef_block.Sub_Block3_2(model,'3d',120,5,20)
    model = ef_block.Sub_Block3_2(model,'3e',120,5,20)
#     model = ef_block.Sub_Block3_2(model,'3f',120,5,20)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),120,5,40)
    model = ef_block.Sub_Block3_2(model,'4b',240,10,40)
    model = ef_block.Sub_Block3_2(model,'4c',240,10,40)
    model = ef_block.Sub_Block3_2(model,'4d',240,10,40)
    model = ef_block.Sub_Block3_2(model,'4e',240,10,40)
    
#     # Final
    
    model = ef_block.Final_Block(model ,56)
    
    return model

def Evolved_46(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)

    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
    model = ef_block.Sub_Block3_2(model,'2c',144,6,24)

    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)
    model = ef_block.Sub_Block3_2(model,'3c',240,10,40)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)
    

#     # Final
    
    model = ef_block.Final_Block(model ,112)
    
    return model


def Evolved_47(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)

    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)

    
    
#     # Final
    
    model = ef_block.Final_Block(model ,112)
    
    return model