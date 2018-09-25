# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 13:28:10 2018
1.extract_train and test image of mnist

@author: Administrator
"""

import os
import sys
import csv
import numpy as np
from PIL import Image
from PIL import ImageFilter,ImageEnhance


"""
#=======extract_train_image of mnist===================
sample: extract_train_image_mnist(filename,save_path,"train_img")    
"""
def extract_train_image_mnist(filename,save_path,prefixion):
    cnt=0    
    csvfile = open(filename)
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        cnt+=1
        if(cnt>1):
           # print len(row)
            label=row[0]
            data=np.array(row[1:], dtype = int)
            #print data.shape
            data.resize(28,28)
            #print data.shape
            
            pil_im2 = Image.fromarray((data))
            im = pil_im2.convert('RGB')
            im = im.convert('L')
            #pil_im2.show()
            im.save(os.path.join(save_path,"%s_"%(label)+prefixion+"_%05d.jpg"%(cnt-2)))
            #sys.exit()
            if (cnt-1)%100==0 :
                sys.stdout.write("\r process %d files"%(cnt-1))
                sys.stdout.flush()
    csvfile.close()#关闭文件

"""
#=======extract_test_image of mnist===================
sample: extract_test_image(filename,save_path,"test_img")
"""
def extract_test_image_mnist(filename,save_path,prefixion):
    cnt=0    
    csvfile = open(filename)
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        cnt+=1
        if(cnt>1):
            #print len(row)
            data=np.array(row, dtype = int)
            data.resize(28,28)
            #print data.shape
            pil_im2 = Image.fromarray((data))
            im = pil_im2.convert('RGB')
            im = im.convert('L')
            #pil_im2.show()
            im.save(os.path.join(save_path,prefixion+"_%05d.jpg"%(cnt-2)))
            #sys.exit()
            if (cnt-1)%100==0 :
                sys.stdout.write("\r process %d files"%(cnt-1))
                sys.stdout.flush()
    csvfile.close()#关闭文件


if __name__ == '__main__':
    pass
#    filename=r"F:\datasets\kaggle mnist\train.csv"
#    save_path=r"F:\datasets\kaggle mnist\train"
#    extract_train_image_mnist(filename,save_path,"train_img")    
    
    filename=r"F:\datasets\kaggle mnist\test.csv"
    save_path=r"F:\datasets\kaggle mnist\test"
    extract_test_image_mnist(filename,save_path,"test_img")