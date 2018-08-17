# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:31:23 2018

1.Batch rename files
2.Batch resize files
3.Batch expansion files

@author: lijianfei
"""

from numpy import *
import operator
import os
from os import listdir
import os.path
import random
import matplotlib.pyplot as plt
import re
import sys
from PIL import Image
from PIL import ImageFilter,ImageEnhance



"""
#=======Batch rename files===================
sample: Rename_Image(r"C:\Users\Administrator\Desktop\image","img")
"""
def Rename_Image(str_place,prefixion):
    print str_place
    cnt=0
    for dirpath, dirnames, filenames in os.walk(str_place):
        #print "Directory:%s"%dirpath
        #random.shuffle(filenames)
        for filename in filenames:
            #print filename
            os.rename(os.path.join(str_place,filename),os.path.join(str_place,prefixion+'_%05d'%(cnt)+'.jpg'))
            cnt+=1
            if cnt%100==0 or filename==filenames[-1]:
                sys.stdout.write("\r process %d files"%cnt)
                sys.stdout.flush()
            
            
            
"""
#=======Batch resize files===================
sample: Resize_Image(r"C:\Users\Administrator\Desktop\image","img",256,256)
"""
def Resize_Image(str_place,prefixion,new_w,new_h):
    print str_place
    cnt=0
    for dirpath, dirnames, filenames in os.walk(str_place):
        for filename in filenames:
            im = Image.open(os.path.join(str_place,filename))
            im = im.convert('RGB')
            im_resize = im.resize((new_w, new_h))
            im_resize.save(os.path.join(str_place,filename))
            cnt+=1
            if cnt%100==0 or filename==filenames[-1]:
                sys.stdout.write("\r process %d files"%cnt)
                sys.stdout.flush()
                
         
            
            
"""
#=======Batch expansion files===================
sample: Expansion_Image(r"C:\Users\Administrator\Desktop\image","img",40,40)
"""
def Expansion_Image(str_place,prefixion,new_w,new_h):
    print str_place
    cnt=0
    for dirpath, dirnames, filenames in os.walk(str_place):
        for filename in filenames:
            im = Image.open(os.path.join(str_place,filename))
            im = im.convert('RGB')
            n_im= Image.new("RGB", (new_w, new_h),"grey")
            n_im.paste(im, ((new_w-im.size[0])/2, (new_h-im.size[1])/2, 
                            (new_w-im.size[0])/2+im.size[0], (new_h-im.size[1])/2+im.size[1]))
            
            n_im.save(os.path.join(str_place,filename))
            cnt+=1
            if cnt%100==0 or filename==filenames[-1]:
                sys.stdout.write("\r process %d files"%cnt)
                sys.stdout.flush()                
                
if __name__ == '__main__':
    
    #Rename_Image(r"C:\Users\Administrator\Desktop\image","img")
    #Resize_Image(r"C:\Users\Administrator\Desktop\image","img",32,32)
    Expansion_Image(r"C:\Users\Administrator\Desktop\image","img",40,40)



