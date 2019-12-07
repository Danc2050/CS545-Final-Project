'''
Our neural net implementation on the credit-card data.
Details: 1 Hidden layer (due to UAT), Dimensionality reduction.
Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np
import os
from prepare import prepare_data
from MLP import mlp_test as mt

def main():
  # read in data
  data = np.genfromtxt(os.path.normpath('data/dataset_even.csv'), delimiter=',')
  idx_label = np.shape(data)[1] - 1 # last column
  (train_set, valid_set, test_set, n_class) = prepare_data(data, idx_label, 3, 0, 1)
  mt.sweep_test(train_set, valid_set, test_set, n_class)

if __name__=="__main__":
  main()
