# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:26:19 2018

@author: Administrator
"""

import numpy as np     
import struct    
import matplotlib.pyplot as plt     
from PIL import Image,ImageFont,ImageDraw
import cv2
import scipy.misc

filename = 'C://Users//Administrator//Desktop//Conditional-Gans//data//mnist//train-images-idx3-ubyte'    
#filename = 'C:/Users/haoming/Desktop/train-images-idx3-ubyte' 
filename1 = 'C://Users//Administrator//Desktop//Conditional-Gans//data//mnist//train-labels-idx1-ubyte'

binfile = open(filename,'rb')#以二进制方式打开    
lbinfile = open(filename1,'rb')
buf  = binfile.read()    
lbuf = lbinfile.read()   

index = 0
lind  = 0
magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)#读取4个32 int    
print (magic,' ',numImages,' ',numRows,' ',numColums  )  
index += struct.calcsize('>IIII')    

lmagic, numl = struct.unpack_from('>II',lbuf,lind)
print 'label'
print (lmagic,' ', numl)
lind += struct.calcsize('>II')

outputLabel='C://Users//Administrator//Desktop//Conditional-Gans//data//mnist//labels.txt'
fw=open(outputLabel,"w+")

outputImgDir='C://Users//Administrator//Desktop//Conditional-Gans//data//mnist//Dataset_img/'

for i in range(numl):
    im = struct.unpack_from('>784B',buf,index)
    index += struct.calcsize('>784B' )
    im = np.array(im)

    #np.transpose(im) 
    #print im.shape

    im = im.reshape(28,28)
    imgdir=outputImgDir+str(i)+'.jpg'
    scipy.misc.imsave(imgdir, im)

##########3
    #tlabel=np.array((struct.unpack_from('>1B',lbuf,lind)))[0]
    tlabel=np.array((struct.unpack_from('>1B',lbuf,lind)))[0]
    #print tlabel
    fw.write(str(tlabel)+"\n")
    lind+=struct.calcsize('>1B')

fw.close()
binfile.close()
lbinfile.close()

