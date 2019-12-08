'''
Our neural net implementation on the credit-card data.
Details: 1 Hidden layer (due to UAT), Dimensionality reduction.
Authors: Daniel Connelly, Dalton Bohning, Ebele Esimai, Dawei Zhang, Daniel Lee
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np
import prepare
from MLP import mlp_test as mt



def main():
  (data, data_train, data_test, labels_train, labels_test, n_class) = prepare.getPreparedData()
  data_train = (data_train, labels_train)
  data_test = (data_test, labels_test)

  mt.sweep_test(data_train, data_test, n_class)

if __name__=="__main__":
  main()
