'''
Our neural net implementation on the credit-card data.
Details: 1 Hidden layer (due to UAT), Dimensionality reduction.
Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np
from os import path
from prepare import prepare_data
from MLP import mlp_test as mt



def main():
  # read in data
  data = np.genfromtxt(path.join('data', 'data.csv'), delimiter=',')
  (data_train, data_test, labels_train, labels_test, n_class) = prepare_data(data)
  data_train = (data_train, labels_train)
  data_test = (data_test, labels_test)

  mt.sweep_test(data_train, data_test, n_class)

if __name__=="__main__":
  main()
