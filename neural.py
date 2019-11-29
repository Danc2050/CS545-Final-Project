'''
Our neural net implementation on the credit-card data.
Details: 1 Hidden layer (due to UAT), Dimensionality reduction.
Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np
from prepare import prepare_data
from MLP import mlp_test as mt

def main():
  (train_set, valid_set, test_set) = prepare_data()
  mt.sweep_test(train_set, valid_set, test_set)

if __name__=="__main__":
  main()
