#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, cv2
import matplotlib.pyplot as plt
import numpy as np

CAPTCHA_FOLDER = "captcha/"
PROCESSED_FOLDER = "processed/"


# In[2]:


def preprocessing(from_filename, to_filename):
    if not os.path.isfile(from_filename):
        return
    img = cv2.imread(from_filename)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
    
    kernel = np.ones((4,4), np.uint8) 
    erosion = cv2.erode(denoised, kernel, iterations=1)
    burred = cv2.GaussianBlur(erosion, (5, 5), 0)
    
    edged = cv2.Canny(burred, 30, 150)
    dilation = cv2.dilate(edged, kernel, iterations=1) 

    cv2.imwrite(to_filename, dilation)
    return


# In[3]:


i = 5000

# #ignore existing image
while True:
    i += 1
    filename = PROCESSED_FOLDER + str(i) + '.jpg'
    if not os.path.isfile(filename):
        i -= 1
        break

print("start to process image from index: " + str(i))

while True:
    i += 1
    filename = str(i) + ".jpg"
    if not os.path.isfile(CAPTCHA_FOLDER + filename):
        print("filename not exists: " + (CAPTCHA_FOLDER + filename))
        break
    
    from_filename = CAPTCHA_FOLDER + filename
    to_filename = PROCESSED_FOLDER + filename
    preprocessing(from_filename, to_filename)
   
    print("i: " + str(i))

print("completed")


# In[ ]:




