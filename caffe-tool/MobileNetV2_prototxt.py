# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:00:13 2018
cifar 10 的mobilenetV2
@author: Administrator
"""

import write_prototxt

def MobileV2_Unit(network,last_name='',block_name='conv2_',block_n=3,num_out0=16,num_out1=16,num_out2=16,downsampling=False,down_method='pooling',use_global_stats='False'):
    
    input_name=last_name








        
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1
        
        
        
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+'0'+'/expand',bottom_name=last_name,top_name=block_name+'0'+'/expand',num_output=num_out1,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'0'+"/bn/expand",name_scale=block_name+'0'+"/scale/expand",
                          bottom_name=block_name+'0'+'/expand',top_name=block_name+'0'+'/expand',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+'0'+"ReLU/dw",bottom_name=block_name+'0'+'/expand',top_name=block_name+'0'+'/expand')
        
        
        #ConvolutionDepthwise
        
        network,last_name=write_prototxt.ConvolutionDepthwise(network,name=block_name+'0'+'/dw',bottom_name=last_name,top_name=block_name+'0'+'/dw',num_output=num_out1,
                                                             bias_term=False,pad=1,kernel_size=3,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'0'+'/bn/dw',name_scale=block_name+'0'+"/scale/dw",
                          bottom_name=block_name+'0'+'/dw',top_name=block_name+'0'+'/dw',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+'0'+"ReLU/dw",bottom_name=block_name+'0'+'/dw',top_name=block_name+'0'+'/dw')
    
    
        network,last_name=write_prototxt.Convolution(network,name=block_name+'0'+'/linear',bottom_name=last_name,top_name=block_name+'0'+'/linear',num_output=num_out2,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'0'+"/bn/linear",name_scale=block_name+'0'+"/scale/linear",
                          bottom_name=block_name+'0'+'/linear',top_name=block_name+'0'+'/linear',use_global_stats=use_global_stats)
        #network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+'0'+"ReLU/dw",bottom_name=block_name+'0'+'/sep',top_name=block_name+'0'+'/sep')
    
        if num_out0==num_out2:
            network,last_name = write_prototxt.Eltwise(network,name="block_"+block_name+'0',bottom_name1=input_name,bottom_name2=last_name,top_name="block_"+block_name,operation='')
 
    return network,last_name



def Net( mode='train',root_path='',batch_size=32):


    network='name:"MobileNetV2"'+'\n'
    
    if mode=='train':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=True,crop_size=28,batch_size=batch_size,backend="LMDB",shuffle=True,datasets_path=root_path)
        use_global_stats=False
    elif mode=='test':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=False,crop_size=28,batch_size=batch_size,backend="LMDB",shuffle=False,datasets_path=root_path)
        use_global_stats=True
    
    
    network,last_name=write_prototxt.Convolution(network,name="conv1",bottom_name=last_name,top_name='conv1',num_output=32,
                        bias_term=False,pad=1,kernel_size=3,stride=1,weight_type='msra',bias_type='constant')
    network,last_name=write_prototxt.BatchNorm(network,name_bn="conv1_bn",name_scale="conv1__scale",
                          bottom_name="conv1",top_name="conv1",use_global_stats=use_global_stats)
    network,last_name=write_prototxt.ReLU(network,name="conv1_ReLU",bottom_name='conv1',top_name='conv1')
         

    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv2_',
                                  block_n=1,num_out0=32,num_out1=32,num_out2=16,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv3_',
                                  block_n=1,num_out0=16,num_out1=96,num_out2=24,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv4_',
                                  block_n=1,num_out0=24,num_out1=144,num_out2=24,downsampling=False,use_global_stats=use_global_stats)
   
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv5_',
                                  block_n=1,num_out0=24,num_out1=144,num_out2=32,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv6_',
                                  block_n=1,num_out0=32,num_out1=144,num_out2=32,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv7_',
                                  block_n=1,num_out0=32,num_out1=192,num_out2=32,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv8_',
                                  block_n=1,num_out0=32,num_out1=192,num_out2=32,downsampling=False,use_global_stats=use_global_stats)

    
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv9_',
                                  block_n=1,num_out0=32,num_out1=192,num_out2=48,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv10_',
                                  block_n=1,num_out0=48,num_out1=384,num_out2=48,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv11_',
                                  block_n=1,num_out0=48,num_out1=384,num_out2=48,downsampling=False,use_global_stats=use_global_stats)

    network,last_name=MobileV2_Unit(network,last_name=last_name,block_name='conv12_',
                                  block_n=1,num_out0=48,num_out1=576,num_out2=64,downsampling=False,use_global_stats=use_global_stats)

    
    network,last_name=write_prototxt.Pooling(network,name="Pooling1",bottom_name=last_name,top_name='Pooling1',pool='AVE',global_pooling=True)
   
    
  
    network,last_name=write_prototxt.InnerProduct(network,name="fc1",bottom_name=last_name,top_name='fc1',num_output=10,weight_type='xavier',bias_type='constant')
    if mode=='train':
        network,last_name=write_prototxt.SoftmaxWithLoss(network,name="Softmax1",bottom_name1='fc1',bottom_name2='Data2',top_name='Softmax1')
    if mode=='test':
        network,last_name=write_prototxt.Accuracy(network,name="prob",bottom_name1='fc1',bottom_name2='Data2',top_name='prob')
#    
#   
    print network
    
    
    return network





if __name__ == '__main__':
    root_path_train="./examples/ljftest_cifar10_MobileNetV2/train_lmdb"
    root_path_test="./examples/ljftest_cifar10_MobileNetV2/test_lmdb"
    

     
    with open("train.prototxt", 'w') as f:
        f.write(str(Net(mode='train',root_path=root_path_train,batch_size=8)))#创建 train.prototxt
    with open("test.prototxt", 'w') as f:
        f.write(str(Net(mode='test',root_path=root_path_test,batch_size=10)))#创建 train.prototxt
 