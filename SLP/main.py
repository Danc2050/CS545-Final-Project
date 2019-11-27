#starting for the program2
import numpy as np
import function as fun
#accuracy to be stored
test_accuracy = []
train_accuracy = []
#initial data
attributes = 24 #labels are included
learning_rate = 0.1
#importing from csv
train = np.genfromtxt('credit_card_train.csv',delimiter=',',max_rows=25000) #max 25000
test = np.genfromtxt('credit_card_test.csv',delimiter=',',max_rows=5000) #max 5000

#separating labels
label_train = fun.add_label(train,(train.shape[0]))
label_test = fun.add_label(test,(test.shape[0]))

#preprocess data

train = fun.preprocess(train)
test = fun.preprocess(test)

#create weight matrix

weights_train = fun.create_weights_slp(attributes) #24 one for the bias and only one output

#confusion matrix creation

conf_matrix = fun.confusion_matrix_create(2)

#epoch train
weights_train = fun.epoch_train_slp(train,label_train, learning_rate,weights_train, conf_matrix,label_test,test,200,train_accuracy,test_accuracy)

#export file

with open("train_accuracy.txt",'w') as e:
	e.write(str(train_accuracy))
	e.write("\n")
	e.close()

with open("test_accuracy.txt",'w') as f:
	f.write(str(test_accuracy))
	f.write("\n")
	f.close()
