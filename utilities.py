#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, cv2, csv
import numpy as np


# In[ ]:


def one_hot_encoding(text, allowedChars):
    label_list = []
    for c in text:
        onehot = [0] * len(allowedChars)
        onehot[allowedChars.index(c)] = 1
        label_list.append(onehot)
    return label_list


# In[ ]:


def one_hot_decoding(prediction, allowedChars):
    text = ''
    for predict in prediction:
        value = np.argmax(predict[0])
        text += allowedChars[value]
    return text


# In[ ]:


def read_train_data(filename, size):
    train_data = []
    if os.path.isdir(filename):
        train_data = np.stack([np.array(cv2.imread(filename + str(index) + ".jpg"))/255.0 for index in range(1, size + 1)])
    return train_data


# In[ ]:


def read_label_data(filename, allowedChars, num_dic):
    train_label = []
    traincsv = open(filename, 'r', encoding = 'utf8')
    
    read_label =  [one_hot_encoding(row[0], allowedChars) for row in csv.reader(traincsv)]
    train_label = [[] for _ in range(num_dic)]
    
    for arr in read_label:
        for index in range(num_dic):
            train_label[index].append(arr[index])
    train_label = [arr for arr in np.asarray(train_label)]
    return train_label


# In[ ]:


import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def create_cnn_model(width, height, allowedChars, num_digit):
    print('Creating CNN model...')
    tensor_in = Input((height, width, 3))

    tensor_out = tensor_in
    tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
    tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
    tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = BatchNormalization(axis=1)(tensor_out)
    tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
    tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
    tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
    tensor_out = BatchNormalization(axis=1)(tensor_out)
    tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)

    tensor_out = Flatten()(tensor_out)
    tensor_out = Dropout(0.5)(tensor_out)

    tensor_out = [Dense(len(allowedChars), name='digit' + str(i), activation='softmax')(tensor_out) for i in range(1, num_digit + 1)]
    
    model = Model(inputs=tensor_in, outputs=tensor_out)
    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    print(model.summary())
    
    return model


# In[ ]:




