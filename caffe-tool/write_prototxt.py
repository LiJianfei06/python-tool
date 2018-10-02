# -*- coding: utf-8 -*-
"""
Created on Sun May 20 23:39:19 2018
直接生成 train.prototxt 和 test.prototxt 不需要以来caffe的python接口
若有新的层慢慢添加，积少成多
@author: Administrator
"""

import sys
import os
import time


''' data层 '''
def data(network,name="Data1",mirror=False,scale=1.0,crop_size=224,batch_size=32,backend="LMDB",shuffle=True,datasets_path=''):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Data"'+'\n'
    network=network+'  top: "data"'+'\n'
    network=network+'  top: "label"'+'\n'
    
    network=network+'  transform_param {'+'\n'
    network=network+'    scale: %s'%str(scale)+'\n'
    network=network+'    mirror: %s'%str(mirror)+'\n'
    network=network+'    crop_size: %s'%str(crop_size)+'\n'
    network=network+'  }'+'\n'
    
    network=network+'  data_param {'+'\n'  
    network=network+'    source: "%s"'%str(datasets_path)+'\n'
    network=network+'    batch_size: %s'%str(batch_size)+'\n'
    network=network+'    backend: %s'%str(backend)+'\n'
    network=network+'  }'+'\n'
    
    network=network+'  image_data_param {'+'\n'  
    network=network+'    shuffle: %s'%str(shuffle)+'\n'
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,"data"

''' Convolution层 '''
def Convolution(network,name="conv1",bottom_name='',top_name='',num_output=16,bias_term=False,pad=1,kernel_size=3,stride=1,
                 weight_type='msra',bias_type='constant',group=1):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Convolution"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 1'+'\n'
    network=network+'    decay_mult: 1'+'\n'
    network=network+'  }'+'\n'
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 2'+'\n'
    network=network+'    decay_mult: 0'+'\n'
    network=network+'  }'+'\n'       
    
    network=network+'  convolution_param {'+'\n'
    network=network+'    num_output: %s'%str(num_output)+'\n'
    network=network+'    bias_term: %s'%str(bias_term)+'\n'
    network=network+'    pad: %s'%str(pad)+'\n'
    if group>1:
        network=network+'    group: %s'%str(group)+'\n'    
    network=network+'    kernel_size: %s'%str(kernel_size)+'\n'
    network=network+'    stride: %s'%str(stride)+'\n'
    network=network+'    weight_filler {'+'\n'  
    network=network+'      type: "%s"'%str(weight_type)+'\n'
    network=network+'    }'+'\n'
    if bias_term==True:
        network=network+'    bias_filler {'+'\n'  
        network=network+'      type: "%s"'%str(bias_type)+'\n'
        network=network+'      value: 0'+'\n'
        network=network+'    }'+'\n'      
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,top_name


''' ConvolutionDepthwise层 '''
def ConvolutionDepthwise(network,name="conv1",bottom_name='',top_name='',num_output=16,bias_term=False,pad=1,kernel_size=3,stride=1,
                 weight_type='msra',bias_type='constant'):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "ConvolutionDepthwise"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 1'+'\n'
    network=network+'    decay_mult: 1'+'\n'
    network=network+'  }'+'\n'
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 2'+'\n'
    network=network+'    decay_mult: 0'+'\n'
    network=network+'  }'+'\n'     
    
    network=network+'  convolution_param {'+'\n'
    network=network+'    num_output: %s'%str(num_output)+'\n'
    network=network+'    bias_term: %s'%str(bias_term)+'\n'
    network=network+'    pad: %s'%str(pad)+'\n'
    network=network+'    kernel_size: %s'%str(kernel_size)+'\n'
    network=network+'    stride: %s'%str(stride)+'\n'
    network=network+'    weight_filler {'+'\n'  
    network=network+'      type: "%s"'%str(weight_type)+'\n'
    network=network+'    }'+'\n'
    
    if bias_term==True:
        network=network+'    bias_filler {'+'\n'  
        network=network+'      type: "%s"'%str(bias_type)+'\n'
        network=network+'      value: 0'+'\n'
        network=network+'    }'+'\n'      

    
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,top_name





''' ShuffleChannel层 '''
def ShuffleChannel(network,name="conv1",bottom_name='',top_name='',group=3):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "ShuffleChannel"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  shuffle_channel_param {'+'\n'
    network=network+'    group: %s'%str(group)+'\n'
 
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,top_name



''' BatchNorm层 '''
def BatchNorm(network,name_bn="bn1",name_scale="scale1",bottom_name='',top_name='',use_global_stats=False):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name_bn
    network=network+'  type: "BatchNorm"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 0'+'\n'
    network=network+'    decay_mult: 0'+'\n'
    network=network+'  }'+'\n'
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 0'+'\n'
    network=network+'    decay_mult: 0'+'\n'
    network=network+'  }'+'\n'
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 0'+'\n'
    network=network+'    decay_mult: 0'+'\n'
    network=network+'  }'+'\n'      
    
    network=network+'  batch_norm_param {'+'\n'
    network=network+'    use_global_stats: %s'%str(use_global_stats)+'\n'
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name_scale
    network=network+'  type: "Scale"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  scale_param {'+'\n'
    network=network+'    bias_term: true\n'
    network=network+'  }'+'\n'
    network=network+'}'+'\n'    
    
      
    return network,top_name


''' ReLU层 '''
def ReLU(network,name="ReLU1",bottom_name='',top_name=''):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "ReLU"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n' 

    network=network+'}'+'\n'

      
    return network,top_name


''' Pooling层 '''
def Pooling(network,name="Pooling1",bottom_name='',top_name='',pool='MAX',kernel_size=2,stride=2,global_pooling=False):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Pooling"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'  pooling_param {'+'\n'
    network=network+'    pool: %s'%str(pool)+'\n'
    if(global_pooling==False):network=network+'    kernel_size: %s'%str(kernel_size)+'\n'
    if(global_pooling==False):network=network+'    stride: %s'%str(stride)+'\n'
    if(global_pooling==True):network=network+'    global_pooling: %s'%str(global_pooling)+'\n' 
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,top_name


''' InnerProduct层 '''
def InnerProduct(network,name="fc1",bottom_name='',top_name='',num_output=10,
                 weight_type='xavier',bias_type='constant'):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "InnerProduct"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'

    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 1'+'\n'
    network=network+'    decay_mult: 1'+'\n'
    network=network+'  }'+'\n'
    network=network+'  param {'+'\n'
    network=network+'    lr_mult: 2'+'\n'
    network=network+'    decay_mult: 1'+'\n'
    network=network+'  }'+'\n' 
    
    network=network+'  inner_product_param {'+'\n'
    network=network+'    num_output: %s'%str(num_output)+'\n'
    network=network+'    weight_filler {'+'\n'  
    network=network+'      type: "%s"'%str(weight_type)+'\n'
    network=network+'    }'+'\n'
    network=network+'    bias_filler {'+'\n'  
    network=network+'      type: "%s"'%str(bias_type)+'\n'
    network=network+'      value: 0'+'\n'
    
    network=network+'    }'+'\n'
    network=network+'  }'+'\n'
    network=network+'}'+'\n'
    
    return network,top_name



''' Eltwise层 '''
def Eltwise(network,name="Eltwise1",bottom_name1='',bottom_name2='',top_name='',operation='SUM'):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Eltwise"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name1+'\n'
    network=network+'  bottom: "%s"'%bottom_name2+'\n' 
    network=network+'  top: "%s"'%top_name+'\n' 
    if operation!='':
        network=network+'  eltwise_param {'+'\n'
        network=network+'    operation: %s'%str(operation)+'\n'
        network=network+'  }'+'\n'
    
    network=network+'}'+'\n'

      
    return network,top_name


''' Concat层 '''
def Concat(network,name="Concat1",bottom_name1='',bottom_name2='',top_name=''):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Concat"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name1+'\n'
    network=network+'  bottom: "%s"'%bottom_name2+'\n' 
    network=network+'  top: "%s"'%top_name+'\n' 
    network=network+'}'+'\n'

      
    return network,top_name


''' Dropout层 '''
def Dropout(network,name="Dropout1",bottom_name1='',top_name='',dropout_ratio=1.0):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Dropout"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name1+'\n'
    network=network+'  top: "%s"'%top_name+'\n' 
    network=network+'  dropout_param {'+'\n'
    network=network+'    dropout_ratio: %s'%str(dropout_ratio)+'\n'
    network=network+'  }'+'\n'       
    network=network+'}'+'\n'

      
    return network,top_name




  
''' SoftmaxWithLoss层 '''
def SoftmaxWithLoss(network,name="Softmax1",bottom_name1='',bottom_name2='',top_name=''):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "SoftmaxWithLoss"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name1+'\n'
    network=network+'  bottom: "%s"'%bottom_name2+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'}'+'\n'
    
    return network,top_name


''' Accuracy层 '''
def Accuracy(network,name="prob",bottom_name1='',bottom_name2='',top_name='prob'):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Accuracy"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name1+'\n'
    network=network+'  bottom: "%s"'%bottom_name2+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'}'+'\n'
    
    return network,top_name


''' Softmax层 '''
def Softmax(network,name="prob",bottom_name='',top_name='prob'):
    network=network+'layer {'+'\n'
    network=network+'  name: "%s"\n'%name
    network=network+'  type: "Softmax"'+'\n'
    network=network+'  bottom: "%s"'%bottom_name+'\n'
    network=network+'  top: "%s"'%top_name+'\n'
    
    network=network+'}'+'\n'
    
    return network,top_name






    
    