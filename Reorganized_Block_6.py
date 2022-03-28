#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Input
import EfficientNet_Block as ef_block
def Evolved_61(input_layer):
    
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
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),960,40,224)
    model = ef_block.Sub_Block3_2(model,'5b',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5c',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5d',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5e',1344,56,224)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),1344,56,384)
    model = ef_block.Sub_Block3_2(model,'6b',2304,96,384)
    
#     # Final
    
    model = ef_block.Final_Block(model ,1536)
    
    return model





def Evolved_62(input_layer):
    
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
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),480,20,112)
    model = ef_block.Sub_Block3_2(model,'5b',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5c',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5d',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5e',672,28,112)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),672,28,192)
    model = ef_block.Sub_Block3_2(model,'6b',1152,48,192)
    
#     # Final
    
    model = ef_block.Final_Block(model ,728)
    
    return model

def Evolved_63(input_layer):
    
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
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),480,20,112)
    model = ef_block.Sub_Block3_2(model,'5b',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5c',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5d',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5e',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5f',672,28,112)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),672,28,192)
    model = ef_block.Sub_Block3_2(model,'6b',1152,48,192)
    model = ef_block.Sub_Block3_2(model,'6c',1152,48,192)
    
#     # Final
    
    model = ef_block.Final_Block(model ,728)
    
    return model


def Evolved_64(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 64)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 16 , 64 , 32)
    model = ef_block.Sub_Block3_1(model , '1b' , 8 , 32)
#     model = ef_block.Sub_Block3_1(model , '1c' , 8 , 32)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),192,8,48)
    model = ef_block.Sub_Block3_2(model,'2b',288,12,48)
    model = ef_block.Sub_Block3_2(model,'2c',288,12,48)
    model = ef_block.Sub_Block3_2(model,'2d',288,12,48)
#     model = ef_block.Sub_Block3_2(model,'2e',288,12,48)
#     model = ef_block.Sub_Block3_2(model,'2f',288,12,48)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),288,12,80)
    model = ef_block.Sub_Block3_2(model,'3b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3c',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3d',480,20,80)
    model = ef_block.Sub_Block3_2(model,'3e',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'3f',480,20,80)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),480,20,160)
    model = ef_block.Sub_Block3_2(model,'4b',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4c',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4d',960,40,160)
    model = ef_block.Sub_Block3_2(model,'4e',960,40,160)
#     model = ef_block.Sub_Block3_2(model,'4f',960,40,160)
#     model = ef_block.Sub_Block3_2(model,'4g',960,40,160)
#     model = ef_block.Sub_Block3_2(model,'4h',960,40,160)
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),960,40,224)
    model = ef_block.Sub_Block3_2(model,'5b',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5c',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5d',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5e',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5f',1344,56,224)
    model = ef_block.Sub_Block3_2(model,'5g',1344,56,224)
#     model = ef_block.Sub_Block3_2(model,'5h',1344,56,224)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),1344,56,384)
    model = ef_block.Sub_Block3_2(model,'6b',2304,96,384)
#     model = ef_block.Sub_Block3_2(model,'6c',2304,96,384)
    
#     # Final
    
    model = ef_block.Final_Block(model ,1536)
    
    return model

def Evolved_65(input_layer):
    
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
#     model = ef_block.Sub_Block3_2(model,'4f',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'4g',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'4h',240,10,40)
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),240,10,56)
    model = ef_block.Sub_Block3_2(model,'5b',336,14,56)
    model = ef_block.Sub_Block3_2(model,'5c',336,14,56)
    model = ef_block.Sub_Block3_2(model,'5d',336,14,56)
    model = ef_block.Sub_Block3_2(model,'5e',336,14,56)
    model = ef_block.Sub_Block3_2(model,'5f',336,14,56)
    model = ef_block.Sub_Block3_2(model,'5g',336,14,56)
#     model = ef_block.Sub_Block3_2(model,'5h',336,14,56)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),336,14,96)
    model = ef_block.Sub_Block3_2(model,'6b',576,24,96)
#     model = ef_block.Sub_Block3_2(model,'6c',576,24,96)
    
#     # Final
    
    model = ef_block.Final_Block(model ,364)
    
    return model

def Evolved_66(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)
#     model = ef_block.Sub_Block3_1(model , '1b' , 4 , 16)
#     model = ef_block.Sub_Block3_1(model , '1c' , 4 , 16)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
    model = ef_block.Sub_Block3_2(model,'2c',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2d',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2e',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2f',144,6,24)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)
    model = ef_block.Sub_Block3_2(model,'3c',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3d',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3e',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3f',240,10,40)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4d',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4e',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4f',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4g',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4h',480,20,80)
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),480,20,112)
    model = ef_block.Sub_Block3_2(model,'5b',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5c',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5d',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5e',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5f',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5g',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5h',672,28,112)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),672,28,192)
#     model = ef_block.Sub_Block3_2(model,'6b',1152,48,192)
#     model = ef_block.Sub_Block3_2(model,'6c',1152,48,192)
    
#     # Final
    
    model = ef_block.Final_Block(model ,728)
    
    return model


def Evolved_67(input_layer):
    
    model = ef_block.Stem_Block(input_layer , 32)
    
    # Block 1

    model = ef_block.Sub_Blick1(model , '1a' , 8 , 32 , 16)
#     model = ef_block.Sub_Block3_1(model , '1b' , 4 , 16)
#     model = ef_block.Sub_Block3_1(model , '1c' , 4 , 16)
    
    # Block 2
    
    model = ef_block.Sub_Block2_1(model ,'2a',(3,3),(1,1),96,4,24)
    model = ef_block.Sub_Block3_2(model,'2b',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2c',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2d',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2e',144,6,24)
#     model = ef_block.Sub_Block3_2(model,'2f',144,6,24)
    
#     # Block 3
    
    model = ef_block.Sub_Block2_1(model ,'3a',(5,5),(2,2),144,6,40)
    model = ef_block.Sub_Block3_2(model,'3b',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3c',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3d',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3e',240,10,40)
#     model = ef_block.Sub_Block3_2(model,'3f',240,10,40)
    
#     # Block 4
    
    model = ef_block.Sub_Block2_1(model ,'4a',(3,3),(2,2),240,10,80)
    model = ef_block.Sub_Block3_2(model,'4b',480,20,80)
    model = ef_block.Sub_Block3_2(model,'4c',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4d',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4e',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4f',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4g',480,20,80)
#     model = ef_block.Sub_Block3_2(model,'4h',480,20,80)
    
#     # Block 5
    
    model = ef_block.Sub_Block2_2(model ,'5a',(5,5),(1,1),480,20,112)
    model = ef_block.Sub_Block3_2(model,'5b',672,28,112)
    model = ef_block.Sub_Block3_2(model,'5c',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5d',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5e',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5f',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5g',672,28,112)
#     model = ef_block.Sub_Block3_2(model,'5h',672,28,112)

#     # Block 6
    
    model = ef_block.Sub_Block2_1(model ,'6a',(5,5),(2,2),672,28,192)
#     model = ef_block.Sub_Block3_2(model,'6b',1152,48,192)
#     model = ef_block.Sub_Block3_2(model,'6c',1152,48,192)
    
#     # Final
    
    model = ef_block.Final_Block(model ,728)
    
    return model