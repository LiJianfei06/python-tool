# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:28:03 2018

@author: Administrator
"""

from numpy import *
import numpy as np
import operator
import os
from os import listdir
import shutil
import sys
import re
import random
import xml
import cv2
from PIL import Image
from scipy.misc import imsave
from pylab import *
from xml.etree.ElementTree import ElementTree,Element
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring  
from xml.dom.minidom import parseString 


#=======替换一个文件夹下所有文件的指定内容=====正则表达==============
#---xml_place---文件夹路径--------------------
def redeal_a_xml(filename):                  
    tree = ET.parse(filename)     #打开xml文档 
    root = tree.getroot()

    for size in root.findall('size'): #找到root节点下的size节点 
        width=(size.find('width').text)   #子节点下节点width的值 
        height=(size.find('height').text)   #子节点下节点height的值 
        #print width, height

    boxs=[]
    for object in root.findall('object'): #找到root节点下的所有object节点 
        #name = object.find('name').text   #子节点下节点name的值 
        #print name
        bndbox = object.find('bndbox')      #子节点下属性bndbox的值 
        boxs.append(object.find('name').text)
        boxs.append(bndbox.find('xmin').text)
        boxs.append(bndbox.find('ymin').text)
        boxs.append(bndbox.find('xmax').text)
        boxs.append(bndbox.find('ymax').text)
    #print boxs
    return boxs

    
def extract_image(xml_place,img_place,save_img_place):
    if os.path.exists(save_img_place)==True:
        shutil.rmtree(save_img_place)    
    if os.path.exists(save_img_place)==False:
        os.makedirs(save_img_place)
    cnt=0
    #print img_place
    for dirpath, dirnames, filenames in os.walk(xml_place):
        print "Directory:%s"%dirpath
        for filename in filenames:
            #print dirpath+filename
            cnt+=1
    
            boxs=redeal_a_xml(xml_place+filename)
            image=Image.open(img_place+filename.split('.')[0]+'.JPEG')
            #print boxs
            for i in range(len(boxs)/5):
                #max_WH=max(int(boxs[4+i*5])-int(boxs[2+i*5]),int(boxs[3+i*5])-int(boxs[1+i*5]))
            
#        
#    #    box = (int(boxs[1])-(max_WH-(int(boxs[3])-int(boxs[1])))/2,
#    #           int(boxs[2])-(max_WH-(int(boxs[4])-int(boxs[2])))/2,
#    #           int(boxs[3])+(max_WH-(int(boxs[3])-int(boxs[1])))/2,
#    #           int(boxs[4])+(max_WH-(int(boxs[4])-int(boxs[2])))/2)
                box=(int(boxs[1+i*5]),int(boxs[2+i*5]),int(boxs[3+i*5]),int(boxs[4+i*5]))
                region = image.crop(box)
                #print shape(region)
                out = region.resize((256,256))
                imsave(save_img_place+boxs[0+i*5]+'_%05d'%cnt+'.jpg',out)
#        imshow(out)
#        show()        
    print cnt
        
    
if __name__ == '__main__':
    #           
    xml_place=u"I:/数据集/ImageNet/ILSVRC2012/ILSVRC2012_bbox_train_v2/n03014705/"
    img_place=u"I:/数据集/ImageNet/ILSVRC2012/ILSVRC2012_img_train/n03014705/"
    save_img_place="E:/datasets/ImageNet_simple/n03014705/"
    
    extract_image(xml_place,img_place,save_img_place)
    
    
    
    