# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:39:48 2018

@author: Administrator
"""



import write_prototxt

def ResBlock(network,last_name='',block_name='conv2_',block_n=3,num_out=16,downsampling=False,down_method='pooling',use_global_stats='False',dropout_ratio=0.5,first_layer=False):
    
    input_name=last_name

        
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1
        
        
        
        
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+str(i)+'_0',bottom_name=last_name,top_name=block_name+str(i)+'_0',num_output=num_out,
                            bias_term=True,pad=1,kernel_size=3,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+str(i)+"_bn"+'0',name_scale=block_name+str(i)+"_scale"+'0',
                          bottom_name=block_name+str(i)+'_0',top_name=block_name+str(i)+'_0',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+"_ReLU"+'0',bottom_name=block_name+str(i)+'_0',top_name=block_name+str(i)+'_0')
    
    
        network,last_name=write_prototxt.Dropout(network,name=block_name+"Drop"+str(i),bottom_name1=block_name+str(i)+'_0',top_name=block_name+str(i)+'_0',dropout_ratio=dropout_ratio)
        
        network,last_name=write_prototxt.Convolution(network,name=block_name+str(i)+'_1',bottom_name=last_name,top_name=block_name+str(i)+'_1',num_output=num_out,
                            bias_term=True,pad=1,kernel_size=3,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+str(i)+"bn"+'1',name_scale=block_name+str(i)+"_scale"+'1',
                          bottom_name=block_name+str(i)+'_1',top_name=block_name+str(i)+'_1',use_global_stats=use_global_stats)
        #network,last_name=ReLU(network,name=block_name+str(i)+"ReLU"+'_1',bottom_name=block_name+str(i)+'_1',top_name=block_name+str(i)+'_1')
        origin_name=last_name
        
        network,last_name=write_prototxt.Pooling(network,name=block_name+str(i)+"_Pooling",bottom_name=last_name,top_name=block_name+str(i)+'_Pooling',pool='AVE',global_pooling=True)
        
        num_output_temp=num_out/16
        if num_output_temp<16:num_output_temp=16
        num_output_temp*=10 # 需修改
        network,last_name=write_prototxt.Convolution(network,name=block_name+str(i)+'_2',bottom_name=last_name,top_name=block_name+str(i)+'_2',num_output=num_output_temp,
                            bias_term=True,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+"_ReLU"+'2',bottom_name=block_name+str(i)+'_2',top_name=block_name+str(i)+'_2')

        network,last_name=write_prototxt.Convolution(network,name=block_name+str(i)+'_3',bottom_name=last_name,top_name=block_name+str(i)+'_3',num_output=num_out,
                            bias_term=True,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.Sigmoid(network,name=block_name+str(i)+"_Prob"+'3',bottom_name=block_name+str(i)+'_3',top_name=block_name+str(i)+'_3')
        
        SE_name=last_name    
    
    
        if i==1 and (downsampling==True or first_layer==True):
            
            input_name_temp=last_name
            
            if down_method=='conv':   
                network,last_name=write_prototxt.Convolution(network,name=block_name+str(i)+'_down',bottom_name=input_name,top_name=block_name+str(i)+'_down',num_output=num_out,
                                bias_term=True,pad=0,kernel_size=1,stride=first_stride,weight_type='msra',bias_type='constant')
                network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+str(i)+"_bn"+'_down',name_scale=block_name+str(i)+"_scale"+'_down',
                                  bottom_name=block_name+str(i)+'_down',top_name=block_name+str(i)+'_down',use_global_stats=use_global_stats)
                #network,last_name=ReLU(network,name=block_name+str(i)+"_ReLU"+'_down',bottom_name=block_name+str(i)+'_down',top_name=block_name+str(i)+'_down')
                
            else:
                network,last_name=write_prototxt.Pooling(network,name=block_name+str(i)+"_Pooling"+'_down',bottom_name=input_name,top_name=block_name+str(i)+"_Pooling"+'_down',pool='MAX',kernel_size=2,stride=2)
            
            input_name=last_name
        
#        network,last_name=write_prototxt.Eltwise(network,name=block_name+"Eltwise_"+str(i),bottom_name1=input_name,
#                                  bottom_name2=last_name,top_name=block_name+"Eltwise_"+str(i),operation='SUM')
        network,last_name=write_prototxt.Axpy(network,name=block_name+"Axpy_"+str(i),bottom_name1=SE_name,
                                  bottom_name2=origin_name,bottom_name3=input_name,top_name=block_name+"Axpy_"+str(i))
        network,last_name=write_prototxt.ReLU(network,name=block_name+str(i)+"ReLU"+'_1',bottom_name=last_name,top_name=last_name)
        
        input_name=last_name
    return network,last_name



def WRN_cifar( mode='train',root_path='',batch_size=32,block_n=3,block_k=1):


    network='name:"WRN-28"'+'\n'
    
    if mode=='train':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=True,scale=0.00390625,crop_size=32,batch_size=batch_size,backend="LMDB",shuffle=True,datasets_path=root_path)
        use_global_stats=False
        dropout_ratio=0.5
    elif mode=='test':
        network,last_name=write_prototxt.data(network,name="Data1",mirror=False,scale=0.00390625,crop_size=32,batch_size=batch_size,backend="LMDB",shuffle=False,datasets_path=root_path)
        use_global_stats=True
        dropout_ratio=1.0
    
    
    network,last_name=write_prototxt.Convolution(network,name="conv1",bottom_name=last_name,top_name='conv1',num_output=16,
                        bias_term=True,pad=1,kernel_size=3,stride=1,weight_type='msra',bias_type='constant')
    network,last_name=write_prototxt.BatchNorm(network,name_bn="conv1/bn",name_scale="conv1/scale",
                          bottom_name="conv1",top_name="conv1",use_global_stats=use_global_stats)
    network,last_name=write_prototxt.ReLU(network,name="conv1/ReLU",bottom_name='conv1',top_name='conv1')
        
    
    
    #network=Pooling(network,name="Pooling1",bottom_name='conv1',top_name='Pooling1',pool='MAX')
    
    network,last_name=ResBlock(network,last_name=last_name,block_name='conv2_',block_n=block_n,num_out=16*block_k,downsampling=False,down_method='conv',use_global_stats=use_global_stats,dropout_ratio=dropout_ratio,first_layer=True)
    network,last_name=ResBlock(network,last_name=last_name,block_name='conv3_',block_n=block_n,num_out=32*block_k,downsampling=True,down_method='conv',use_global_stats=use_global_stats,dropout_ratio=dropout_ratio,first_layer=False)
    network,last_name=ResBlock(network,last_name=last_name,block_name='conv4_',block_n=block_n,num_out=64*block_k,downsampling=True,down_method='conv',use_global_stats=use_global_stats,dropout_ratio=dropout_ratio,first_layer=False)
    #network,last_name=ResBlock(network,last_name=last_name,block_name='conv5_',block_n=block_n,num_out=64,downsampling=True,down_method='conv',use_global_stats=use_global_stats,dropout_ratio=dropout_ratio)
    
    network,last_name=write_prototxt.Pooling(network,name="Pooling1",bottom_name=last_name,top_name='Pooling1',pool='AVE',global_pooling=True)
   
    
  
    network,last_name=write_prototxt.InnerProduct(network,name="fc1",bottom_name=last_name,top_name='fc1',num_output=10,weight_type='msra',bias_type='constant')
    #if mode=='train':
    network,last_name=write_prototxt.SoftmaxWithLoss(network,name="Softmax1",bottom_name1='fc1',bottom_name2='label',top_name='Softmax1')
    if mode=='test':
        network,last_name=write_prototxt.Accuracy(network,name="prob",bottom_name1='fc1',bottom_name2='label',top_name='prob')
#    
#   
    print network
    
    
    return network





if __name__ == '__main__':
    root_path_train="/home/lijianfei/datasets/cifar10_40/train_lmdb"
    root_path_test="/home/lijianfei/datasets/cifar10_40/test_lmdb"
    

     
    with open("train_SE-WRN_28.prototxt", 'w') as f:
        f.write(str(WRN_cifar(mode='train',root_path=root_path_train,batch_size=128,block_n=4,block_k=10)))#创建 train.prototxt
    with open("test_SE-WRN_28.prototxt", 'w') as f:
        f.write(str(WRN_cifar(mode='test',root_path=root_path_test,batch_size=10,block_n=4,block_k=10)))#创建 train.prototxt
 