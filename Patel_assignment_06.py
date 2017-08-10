# Patel, Nabilahmed
# 1001-234-817
# 2016-12-04
# Assignment_06

import theano
from theano import tensor as T
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation
from keras.models import model_from_json
import numpy as np
import scipy.misc
from scipy import linalg as la
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
from os import listdir
from os.path import isfile, join
import h5py

def modelling(task_no,train_input,test_input,hidden_nodes,number_epoch):

    n_hidden = hidden_nodes
    nb_classes = 784
    optimizer = 'RMSprop'     
    # optimizer = SGD(lr=0.001)
    loss = 'mean_squared_error'
    metrics = ['accuracy']
    batch_size = 128
    nb_epoch = number_epoch


    if task_no in ['4','5']:

        #laoding model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    else:

        #initializing model
        # input_img = Input(shape=(784,))
        # hidden_output = Dense(n_hidden, activation='relu')(input_img)
        # final_output = Dense(nb_classes, activation='linear')(hidden_output)
        # model = Model(input=input_img, output=final_output)
        # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model = Sequential()
        model.add(Dense(n_hidden, input_shape=(784,), activation='relu')) 
        model.add(Dense(nb_classes, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    

    #In all tasks as the target and input is the same we are passing input at both the places

    if task_no == '1':

        #task1
        history = model.fit(train_input,train_input, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(test_input, test_input), shuffle=True)
        return history

    elif task_no == '2':
        
        #task2
        history = model.fit(train_input,train_input, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(test_input, test_input), shuffle=True)
        train_score = model.evaluate(train_input, train_input, verbose=1)
        test_score = model.evaluate(test_input, test_input, verbose=1)
        return [train_score[0],test_score[0]]

    elif task_no == '3':
        
        #task3
        history = model.fit(train_input,train_input, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True)
        #saving mdoel to json file
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
        weights, biases = model.layers[0].get_weights()
        return weights

    elif task_no in ['4','5']:
        
        #task4
        output = loaded_model.predict(test_input)
        return output


def display_loss(epoch,xlabel,ylabel,train_score,test_score):

    #dsiplaying loss   

    fig1=plt.figure('Loss')
    ax1=fig1.add_subplot(111)
    ax1.plot(epoch, train_score, 'b-', label='train')
    ax1.plot(epoch, test_score, 'g-', label='validation')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.legend()
    plt.show()
 


def display_weights(weights):

    #dsiplaying weights
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    fig.suptitle("Weights", fontsize=16)
    for i in range(100):
        row, column = divmod(i, 10)
        axes[row, column].imshow(weights[:, i].reshape(28, 28), cmap=plt.cm.gray)
        axes[row, column].axis('off') # get rid of tick marks/labels
    plt.show()


def display_images(_input,output):
 
    #displaying input and output image
    fig1, axes1 = plt.subplots(10, 10, figsize=(12, 12))
    fig1.suptitle("Input", fontsize=16)
    fig2, axes2 = plt.subplots(10, 10, figsize=(12, 12))
    fig2.suptitle("Output", fontsize=16)
    for i in range(100):
        row, column = divmod(i, 10)
        axes1[row, column].imshow(_input[i,:].reshape(28, 28), cmap=plt.cm.gray)
        axes1[row, column].axis('off') # get rid of tick marks/labels
        axes2[row, column].imshow(output[i,:].reshape(28, 28), cmap=plt.cm.gray)
        axes2[row, column].axis('off') # get rid of tick marks/labels
    plt.show()

def display_eign_vecs(_input,output):
    #displaying input and output image
    fig1, axes1 = plt.subplots(10, 10, figsize=(12, 12))
    fig1.suptitle("First 100 Eigen vectors of inputs", fontsize=16)
    fig2, axes2 = plt.subplots(10, 10, figsize=(12, 12))
    fig2.suptitle("First 100 Eigen vectors of outputs", fontsize=16)
    for i in range(100):
        row, column = divmod(i, 10)
        axes1[row, column].imshow(_input[:,i].reshape(28, 28), cmap=plt.cm.gray)
        axes1[row, column].axis('off') # get rid of tick marks/labels
        axes2[row, column].imshow(output[:,i].reshape(28, 28), cmap=plt.cm.gray)
        axes2[row, column].axis('off') # get rid of tick marks/labels
    plt.show()

def find_PCA(data):
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    cov_mat = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since cov_mat is symmetric, 
    # the performance gain is substantial
    eign_vals, eign_vecs = la.eigh(cov_mat)
    # sort eigenvalue in decreasing order
    idx = np.argsort(eign_vals)[::-1]
    eign_vecs = eign_vecs[:,idx]
    # sort eigenvectors according to same index
    eign_vals = eign_vals[idx]
    # select the first 100 eigenvectors 
    eign_vecs = eign_vecs[:, :100]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    print(eign_vecs.shape)
    return eign_vecs


def reading_input(path):

    #reading and making input
   
    i = 1
    files = []

    for filename in os.listdir(path):
        if isfile(join(path,filename)):
            files.append(filename)
            files = np.array(files)
            #shuffling files
            np.random.shuffle(files)
            files  = files.tolist()                                   
                                   
    for filename in files:
        img = scipy.misc.imread(join(path,filename)).astype(np.float32)  #read image and convert to float
        img = img.reshape(1,-1)  #reshape to column vector 
        
        if i == 1:
            P = np.array(img)        
            i = i + 1
        else:   
            #adding each image to final input           
            P = np.concatenate((P,img),axis=0)
    
    P = np.divide(P,255)
    print(P.shape)
    return P



#starting...

ds1_input = np.array([])
ds3_input = np.array([])
weights = []


while(True):

    epoch = []
    number_of_nodes = []
    train_score = []
    test_score = []  

    #taking task_no from user via
    #command line arguments
    task_no = raw_input('please input the task no(1 to 5) enter any other number to quit: ')

    #reading training and testing data
   

    if task_no in ['1','2','3','5'] and ds1_input.size == 0:
        ds1_input = reading_input("set1_20k/train/")
        ds2_input = reading_input("set2_2k/")

    elif task_no == '4' and ds3_input.size == 0:
        ds3_input = reading_input("set3_100/")


    #task1
    if task_no == '1':
        history = modelling('1',ds1_input,ds2_input,100,50)
        epoch = [i for i in range(0,50)]
        train_score = history.history['loss']
        test_score = history.history['val_loss']
        display_loss(epoch,"Number of epoch","Mean Squared Error",train_score,test_score)

    #task2
    elif task_no == '2':
        score = modelling('2',ds1_input,ds2_input,20,50)
        train_score.append(score[0])
        test_score.append(score[1])
        score = modelling('2',ds1_input,ds2_input,40,50)
        train_score.append(score[0])
        test_score.append(score[1])
        score = modelling('2',ds1_input,ds2_input,60,50)
        train_score.append(score[0])
        test_score.append(score[1])
        score = modelling('2',ds1_input,ds2_input,80,50)
        train_score.append(score[0])
        test_score.append(score[1])
        score = modelling('2',ds1_input,ds2_input,100,50)
        train_score.append(score[0])
        test_score.append(score[1])
        epoch = [20,40,60,80,100]
        display_loss(epoch,"Number of nodes in hidden layer","Mean Squared Error",train_score,test_score)

    #task3
    elif task_no == '3':
        weights = modelling('3',ds1_input,ds2_input,100,100)
        display_weights(weights)

    #task4
    elif task_no == '4':
        if len(weights) != 0:
            output = modelling('4',ds1_input,ds3_input,100,100)
            display_images(ds3_input,output)
        else:
            print('first complete 3rd task!')

    #task5
    elif task_no == '5':
        if len(weights) != 0:
            output = modelling('5',ds1_input,ds2_input,100,100)
            ds2_PCA = find_PCA(ds2_input)
            output_PCA = find_PCA(output)
            display_eign_vecs(ds2_PCA,output_PCA)
        else:
            print('first complete 3rd task!')

    else:
        sys.exit()
    
