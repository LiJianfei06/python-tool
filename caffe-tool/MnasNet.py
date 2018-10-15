# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:31:23 2018

@author: lijianfei
"""


import write_prototxt

def MnasNet_block_bd11(network,last_name='',block_name='conv2_',block_n=3,num_out0=16,num_out1=16,num_out2=16,downsampling=False,down_method='pooling',use_global_stats='False'):
    
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

def MnasNet_block_bdce(network,last_name='',block_name='conv2_',block_n=3,kernel_size=3,num_out1=16,num_out2=16,downsampling=False,down_method='pooling',use_global_stats='False'):

    input_name=last_name

        
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1  



        network,last_name=write_prototxt.Convolution(network,name=block_name+'/sep1',bottom_name=last_name,top_name=block_name+'/sep1',num_output=num_out1,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+"/sep1/bn",name_scale=block_name+"/sep1/scale",
                          bottom_name=block_name+'/sep1',top_name=block_name+'/sep1',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+"/sep1/ReLU",bottom_name=block_name+'/sep1',top_name=block_name+'/sep1')

            
        #ConvolutionDepthwise
        network,last_name=write_prototxt.ConvolutionDepthwise(network,name=block_name+'/dw',bottom_name=last_name,top_name=block_name+'/dw',num_output=num_out1,
                                                             bias_term=False,pad=int(kernel_size/2),kernel_size=kernel_size,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'/dw/bn',name_scale=block_name+"/dw/scale",
                          bottom_name=block_name+'/dw',top_name=block_name+'/dw',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+"/dw/ReLU",bottom_name=block_name+'/dw',top_name=block_name+'/dw')



        network,last_name=write_prototxt.Convolution(network,name=block_name+'/sep2',bottom_name=last_name,top_name=block_name+'/sep2',num_output=num_out2,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+"/sep2/bn",name_scale=block_name+"/sep2/scale",
                          bottom_name=block_name+'/sep2',top_name=block_name+'/sep2',use_global_stats=use_global_stats)
        if block_name.split("_")[-1]=="1":
            network,last_name=write_prototxt.ReLU(network,name=block_name+"/sep2/ReLU",bottom_name=block_name+'/sep2',top_name=block_name+'/sep2')


        if block_name.split("_")[-1]!="1":
            network,last_name=write_prototxt.Eltwise(network,name=block_name+"/Eltwise"+str(i),bottom_name1=input_name,
                                      bottom_name2=last_name,top_name=block_name+"/Eltwise"+str(i),operation='SUM')
            network,last_name=write_prototxt.ReLU(network,name=block_name+"/Eltwise/ReLU",bottom_name=last_name,top_name=last_name)
     
    
        return network,last_name


def MnasNet_block_first(network,last_name='',block_name='conv2_',block_n=3,num_out1=16,num_out2=16,downsampling=False,down_method='pooling',use_global_stats='False'):

    input_name=last_name

        
    for i in range(1,block_n+1,1):
        if i==1 and downsampling==True:
            first_stride=2
        else:
            first_stride=1  
            
        #ConvolutionDepthwise
        network,last_name=write_prototxt.ConvolutionDepthwise(network,name=block_name+'/dw',bottom_name=last_name,top_name=block_name+'/dw',num_output=num_out1,
                                                             bias_term=False,pad=1,kernel_size=3,stride=first_stride,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+'/dw/bn',name_scale=block_name+"/dw/scale",
                          bottom_name=block_name+'/dw',top_name=block_name+'/dw',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+"/dw/ReLU",bottom_name=block_name+'/dw',top_name=block_name+'/dw')



        network,last_name=write_prototxt.Convolution(network,name=block_name+'/sep',bottom_name=last_name,top_name=block_name+'/sep',num_output=num_out2,
                            bias_term=False,pad=0,kernel_size=1,stride=1,weight_type='msra',bias_type='constant')
        network,last_name=write_prototxt.BatchNorm(network,name_bn=block_name+"/sep/bn",name_scale=block_name+"/sep/scale",
                          bottom_name=block_name+'/sep',top_name=block_name+'/sep',use_global_stats=use_global_stats)
        network,last_name=write_prototxt.ReLU(network,name=block_name+"/sep/ReLU",bottom_name=block_name+'/sep',top_name=block_name+'/sep')





        return network,last_name


def Net( mode='train',root_path='',batch_size=32):


    network='name:"MnasNet"'+'\n'
    
    if mode=='train':
        network,last_name=write_prototxt.data(network,name="data",mirror=True,scale=0.00390625,crop_size=224,batch_size=batch_size,backend="LMDB",shuffle=True,datasets_path=root_path)
        use_global_stats=False
    elif mode=='test':
        network,last_name=write_prototxt.data(network,name="data",mirror=False,scale=0.00390625,crop_size=224,batch_size=batch_size,backend="LMDB",shuffle=False,datasets_path=root_path)
        use_global_stats=True
    elif mode=='deploy':
        network='input: "data"'+'\n'
        network=network+'input_dim: 1'+'\n'
        network=network+'input_dim: 3'+'\n'
        network=network+'input_dim: 224'+'\n'
        network=network+'input_dim: 224'+'\n'
        
        last_name="data"
        use_global_stats=True
    
    network,last_name=write_prototxt.Convolution(network,name="conv1",bottom_name=last_name,top_name='conv1',num_output=32,
                        bias_term=False,pad=1,kernel_size=3,stride=2,weight_type='msra',bias_type='constant')
    network,last_name=write_prototxt.BatchNorm(network,name_bn="conv1/bn",name_scale="conv1/scale",
                          bottom_name="conv1",top_name="conv1",use_global_stats=use_global_stats)
    network,last_name=write_prototxt.ReLU(network,name="conv1/ReLU",bottom_name='conv1',top_name='conv1')
         

    network,last_name=MnasNet_block_first(network,last_name=last_name,block_name='conv2_1',
                                  block_n=1,num_out1=32,num_out2=16,downsampling=False,use_global_stats=use_global_stats)
    
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv3_1',
                                  block_n=1,kernel_size=3,num_out1=48,num_out2=24,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv3_2',
                                  block_n=1,kernel_size=3,num_out1=72,num_out2=24,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv3_3',
                                  block_n=1,kernel_size=3,num_out1=72,num_out2=24,downsampling=False,use_global_stats=use_global_stats)


    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv4_1',
                                  block_n=1,kernel_size=5,num_out1=72,num_out2=40,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv4_2',
                                  block_n=1,kernel_size=5,num_out1=120,num_out2=40,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv4_3',
                                  block_n=1,kernel_size=5,num_out1=120,num_out2=40,downsampling=False,use_global_stats=use_global_stats)


    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv5_1',
                                  block_n=1,kernel_size=5,num_out1=240,num_out2=80,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv5_2',
                                  block_n=1,kernel_size=5,num_out1=480,num_out2=80,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv5_3',
                                  block_n=1,kernel_size=5,num_out1=480,num_out2=80,downsampling=False,use_global_stats=use_global_stats)


    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv6_1',
                                  block_n=1,kernel_size=3,num_out1=480,num_out2=96,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv6_2',
                                  block_n=1,kernel_size=3,num_out1=576,num_out2=96,downsampling=False,use_global_stats=use_global_stats)




    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv7_1',
                                  block_n=1,kernel_size=5,num_out1=576,num_out2=192,downsampling=True,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv7_2',
                                  block_n=1,kernel_size=5,num_out1=1152,num_out2=192,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv7_3',
                                  block_n=1,kernel_size=5,num_out1=1152,num_out2=192,downsampling=False,use_global_stats=use_global_stats)
    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv7_4',
                                  block_n=1,kernel_size=5,num_out1=1152,num_out2=192,downsampling=False,use_global_stats=use_global_stats)



    network,last_name=MnasNet_block_bdce(network,last_name=last_name,block_name='conv8_1',
                                  block_n=1,kernel_size=3,num_out1=1152,num_out2=320,downsampling=False,use_global_stats=use_global_stats)



    network,last_name=write_prototxt.Pooling(network,name="Pooling1",bottom_name=last_name,top_name='Pooling1',pool='AVE',global_pooling=True)


    network,last_name=write_prototxt.InnerProduct(network,name="fc1",bottom_name=last_name,top_name='fc1',num_output=2,weight_type='msra',bias_type='constant')
    #if mode=='train':
    network,last_name=write_prototxt.SoftmaxWithLoss(network,name="Softmax1",bottom_name1='fc1',bottom_name2='label',top_name='Softmax1')
    if mode=='test':
        network,last_name=write_prototxt.Accuracy(network,name="prob",bottom_name1='fc1',bottom_name2='label',top_name='prob')
    #if mode=='deploy':
        #network,last_name=write_prototxt.Softmax(network,name="prob",bottom_name='fc1',top_name='prob')

#    print network
    
    
    return network





if __name__ == '__main__':
    root_path_train="/home/lijianfei/datasets/kaggle_cat_dog/train_lmdb"
    root_path_test="/home/lijianfei/datasets/kaggle_cat_dog/test_lmdb"
    

     
    with open("train_MnasNet.prototxt", 'w') as f:
        f.write(str(Net(mode='train',root_path=root_path_train,batch_size=8)))#创建 train.prototxt
    with open("test_MnasNet.prototxt", 'w') as f:
        f.write(str(Net(mode='test',root_path=root_path_test,batch_size=10)))#创建 train.prototxt
    with open("deploy_MnasNet.prototxt", 'w') as f:
        f.write(str(Net(mode='deploy')))#创建 train.prototxt
