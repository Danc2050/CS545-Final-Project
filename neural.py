'''
Our neural net implementation on the credit-card data.
Details: 1 Hidden layer (due to UAT), Dimensionality reduction.
Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np

def normalize_data(data):
  ''' normalize data for training
      method: ( data - mean ) / std
      in: np array of training inputs
      out: np array of normalized inputs 
  '''
  means = np.mean(data, axis=0) # mean, col-wise
  stds = np.std(data, axis=0) # std
  dim = np.shape(means)[0]
  ret = data - np.transpose(means).reshape(1, dim)
  ret = ret / np.transpose(stds).reshape(1, dim)
  return ret

def prepare_data(data, idx_label):
  ''' preprocess data for training
      in: np array with all raw dataset
      out: inputs, labels as np arrays
  '''
  labels = data[:, [idx_label]]
  raw_examples = np.delete(data, idx_label, axis=1)
  examples = normalize_data(raw_examples)
  return (labels, examples)

def split_indexes(n_train, n_validate, n_test, row_total):
  ''' calculate where to split
  '''
  i0 = n_train / (n_train + n_validate + n_test) * row_total
  i1 = (n_train + n_validate) / (n_train + n_validate + n_test) * row_total
  return (int(i0), int(i1))


def main():
  # read in data
  raw_data = np.genfromtxt('data\data.csv', delimiter=',')
  # preprocessing, normalization, etc.
  (labels, examples) = prepare_data(raw_data, np.shape(raw_data)[1] - 1)
  # split
  (i0, i1) = split_indexes(3, 1, 1, np.shape(examples)[0])
  train_set = (examples[:i0], labels[0:i0])
  valid_set = (examples[i0:i1], labels[i0:i1])
  test_set = (examples[i1:], labels[i1:])

  # train the model

  pass



  
if __name__=="__main__":
  main()
