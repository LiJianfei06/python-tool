# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:00:13 2018
cifar 10 的ShuffleNet
@author: Administrator
"""

import write_prototxt







def ShuffleNet_Unit0(network,last_name='',block_name='conv2_',block_n=3,num_out0=16,num_out1=16,downsampling=False,group=3,down_method='pooling',use_global_stats='False'):
    
    input_name=last_name

      
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1
        
     
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+'1',bottom_name=last_name,top_name=block_name+'1',num_output=num_out0,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'1'+"_bn",name_scale=block_name+'1'+"_scale",
                          bottom_name=block_name+'1',top_name=block_name+'1',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+'1'+"_ReLU",bottom_name=block_name+'1',top_name=block_name+'1')
        
        
        #ConvolutionDepthwise
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+'2',bottom_name=last_name,top_name=block_name+'2',num_output=num_out0,
                                                             bias_term=False,pad=1,kernel_size=3,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'2'+'_bn',name_scale=block_name+'2'+"_scale",
                          bottom_name=block_name+'2',top_name=block_name+'2',use_global_stats=use_global_stats)
        #network,last_name=write_prototxt.ReLU(network,name=block_name+'2'+"_ReLU",bottom_name=block_name+'2',top_name=block_name+'2')
    
    
        network,last_name=write_prototxt.Convolution(network,name=block_name+'3',bottom_name=last_name,top_name=block_name+'3',num_output=num_out1,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant',group=group)
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'3'+"_bn",name_scale=block_name+'3'+"_scale",
                          bottom_name=block_name+'3',top_name=block_name+'3',use_global_stats=use_global_stats)
        #network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+'0'+"ReLU/dw",bottom_name=block_name+'0'+'/sep',top_name=block_name+'0'+'/sep')
    
        
        network,last_name = write_prototxt.Concat(network,name=block_name+'_concat',bottom_name1=input_name,bottom_name2=last_name,top_name=block_name+'_concat')
        network,last_name=write_prototxt.ReLU(network,name=block_name+'_concat_ReLU',bottom_name=block_name+'_concat',top_name=block_name+'_concat')
    return network,last_name







def ShuffleNet_Unit(network,last_name='',block_name='conv2_',block_n=3,num_out0=16,num_out1=16,downsampling=False,group=3,down_method='pooling',use_global_stats='False'):
    
    input_name=last_name

      
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1
        
     
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+'1',bottom_name=last_name,top_name=block_name+'1',num_output=num_out0,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant',group=group)
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'1'+"_bn",name_scale=block_name+'1'+"_scale",
                          bottom_name=block_name+'1',top_name=block_name+'1',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+'1'+"_ReLU",bottom_name=block_name+'1',top_name=block_name+'1')
        
        
        
 
        network,last_name=write_prototxt.ShuffleChannel(network,name=block_name+"_shuffle",bottom_name=last_name,top_name=block_name+"_shuffle",group=group)
       
        #ConvolutionDepthwise
        
        network,last_name=write_prototxt.ConvolutionDepthwise(network,name=block_name+'2',bottom_name=last_name,top_name=block_name+'2',num_output=num_out0,
                                                             bias_term=False,pad=1,kernel_size=3,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'2'+'_bn',name_scale=block_name+'2'+"_scale",
                          bottom_name=block_name+'2',top_name=block_name+'2',use_global_stats=use_global_stats)
        #network,last_name=write_prototxt.ReLU(network,name=block_name+'2'+"_ReLU",bottom_name=block_name+'2',top_name=block_name+'2')
    
    
        network,last_name=write_prototxt.Convolution(network,name=block_name+'3',bottom_name=last_name,top_name=block_name+'3',num_output=num_out1,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant',group=group)
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'3'+"_bn",name_scale=block_name+'3'+"_scale",
                          bottom_name=block_name+'3',top_name=block_name+'3',use_global_stats=use_global_stats)
        #network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+'0'+"ReLU/dw",bottom_name=block_name+'0'+'/sep',top_name=block_name+'0'+'/sep')
    
        if(downsampling==True):
            network,input_name=write_prototxt.Pooling(network,name=block_name+"_match",bottom_name=input_name,top_name=block_name+"_match",pool='AVE',kernel_size=3,stride=2)
            network,last_name = write_prototxt.Concat(network,name=block_name+'_concat',bottom_name1=input_name,bottom_name2=last_name,top_name=block_name+'_concat')
            network,last_name=write_prototxt.ReLU(network,name=block_name+'_concat_ReLU',bottom_name=block_name+'_concat',top_name=block_name+'_concat')
        else:
           network,last_name = write_prototxt.Eltwise(network,name=block_name+'_elewise',bottom_name1=input_name,bottom_name2=last_name,top_name=block_name+'_elewise',operation='SUM')
           network,last_name=write_prototxt.ReLU(network,name=block_name+'elewise_ReLU',bottom_name=block_name+'_elewise',top_name=block_name+'_elewise')
    return network,last_name



def Net( mode='train',root_path='',batch_size=32):


    network='name:"ShuffleNet"'+'\n'
    
    if mode=='train':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=True,crop_size=28,batch_size=batch_size,backend="LMDB",shuffle=True,datasets_path=root_path)
        use_global_stats=False
    elif mode=='test':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=False,crop_size=28,batch_size=batch_size,backend="LMDB",shuffle=False,datasets_path=root_path)
        use_global_stats=True
    
    
    network,last_name=write_prototxt.Convolution(network,name="conv1",bottom_name=last_name,top_name='conv1',num_output=15,
                        bias_term=False,pad=1,kernel_size=3,stride=1,weight_type='msra',bias_type='constant')
    network,last_name=write_prototxt.BatchNorm(network,name_bn="conv1_bn",name_scale="conv1__scale",
                          bottom_name="conv1",top_name="conv1",use_global_stats=use_global_stats)
    network,last_name=write_prototxt.ReLU(network,name="conv1_ReLU",bottom_name='conv1',top_name='conv1')
         
    
    
    
    network,last_name=ShuffleNet_Unit0(network,last_name=last_name,block_name='resx1_conv',
                                  block_n=1,num_out0=24,num_out1=45,downsampling=False,group=3,use_global_stats=use_global_stats)


    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx2_conv',
                                  block_n=1,num_out0=30,num_out1=60,downsampling=False,group=3,use_global_stats=use_global_stats)
    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx3_conv',
                                  block_n=1,num_out0=30,num_out1=60,downsampling=False,group=3,use_global_stats=use_global_stats)
    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx4_conv',
                                  block_n=1,num_out0=30,num_out1=60,downsampling=False,group=3,use_global_stats=use_global_stats)


    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx5_conv',
                                  block_n=1,num_out0=30,num_out1=60,downsampling=True,group=3,use_global_stats=use_global_stats)


    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx6_conv',
                                  block_n=1,num_out0=48,num_out1=120,downsampling=False,group=3,use_global_stats=use_global_stats)
    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx7_conv',
                                  block_n=1,num_out0=48,num_out1=120,downsampling=False,group=3,use_global_stats=use_global_stats)

    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx10_conv',
                                  block_n=1,num_out0=48,num_out1=120,downsampling=True,group=3,use_global_stats=use_global_stats)


    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx11_conv',
                                  block_n=1,num_out0=60,num_out1=240,downsampling=False,group=3,use_global_stats=use_global_stats)
    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx12_conv',
                                  block_n=1,num_out0=60,num_out1=240,downsampling=False,group=3,use_global_stats=use_global_stats)
    network,last_name=ShuffleNet_Unit(network,last_name=last_name,block_name='resx13_conv',
                                  block_n=1,num_out0=60,num_out1=240,downsampling=False,group=3,use_global_stats=use_global_stats)


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
    root_path_train="./examples/ljftest_cifar10_ShuffleNet/train_lmdb"
    root_path_test="./examples/ljftest_cifar10_ShuffleNet/test_lmdb"
    

     
    with open("train.prototxt", 'w') as f:
        f.write(str(Net(mode='train',root_path=root_path_train,batch_size=8)))#创建 train.prototxt
    with open("test.prototxt", 'w') as f:
        f.write(str(Net(mode='test',root_path=root_path_test,batch_size=10)))#创建 train.prototxt
 