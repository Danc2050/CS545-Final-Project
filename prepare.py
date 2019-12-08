'''
Preprocess data, 
Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np
from os import path
from sklearn.model_selection import train_test_split


def normalize_data(data):
  ''' normalize data for training
      method: ( data - mean ) / std
      in: np array of training inputs
      out: np array of normalized inputs 
  '''
  means = np.mean(data, axis=0) # mean, col-wise
  stds = np.std(data, axis=0) # std
  stds[stds == 0.0] = 0.00001 # avoid dividing by '0'
  dim = np.shape(means)[0]
  ret = data - np.transpose(means).reshape(1, dim)
  ret = ret / np.transpose(stds).reshape(1, dim)
  return ret


def prepare_data(data, test_size=0.25, normalize=True):
  ''' preprocess data for training
      in: np array with all raw dataset
      out: inputs, labels as np arrays
  '''
  idx_label = np.shape(data)[1] - 1 # last column
  labels = data[:, [idx_label]] #Get labels column
  labels = labels - np.amin(labels) # remove offset, so start from '0'
  n_class = int(np.amax(labels) + 1) # assume continous class numbers
  data = np.delete(data, idx_label, axis=1) #Remove labels from data
  if normalize: data = normalize_data(data)
  data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size)
  return (data_train, data_test, labels_train, labels_test, n_class)

  
if __name__=="__main__":
  data = np.genfromtxt(path.join('data', 'data.csv'), delimiter=',')
  idx_label = np.shape(data)[1] - 1 # last column
  (train_set, test_set, n_class) = prepare_data(data, idx_label, 3, 1, 1)
