import numpy as np

def add_label(data,number_of_inputs):
	"""
	takes the first column which are the labels
	data: data set
	number_of_inputs: the number of rows required usually data.shape[0]
	"""
	label = np.zeros(number_of_inputs,dtype=float)
	label = data[:,0]
	return label;

def preprocess(data):
	"""
	Makes first column bias of 1
	data: data set
	"""
	array_data = np.array(data,dtype=float)
	array_data[:,0] = 1
	return array_data;

def create_weights_slp(rows):
	"""
	generates initial weights from -0.5 to 0.5
	rows: number of rows which is the number of attributes + 1
	"""
	weights = np.random.rand(rows) # creates decimal values between 0 and 1
	weights -= 0.5 #shift value down by 0.5 to establish range
	weights /= 10 #shrink weights to smaller values

	return weights;

def confusion_matrix_create(number_of_outputs):
	"""
	creates a square confusion_matrix
	number_of_outputs: types of outputs for the classification
	"""

	conf_matrix = np.zeros((number_of_outputs,number_of_outputs),dtype=float)
	return conf_matrix;

def slp_weight_update(data_input,weights,label_input,learning_rate):
	"""
	computes the output from using the input and weights by using a dot product
	data_input: A row from the data set
	weights: The weight matrix for the current layer
	label_input: The label for the row
	learning_rate: The learning rate from gradient descent formula
	"""

	output = np.dot(data_input,np.transpose(weights)) #transpose to allow for dot product
	output = 1 if output > 0 else 0

	#If the prediction is incorrect then update all of the weights
	if (label_input != output):
		weights = weights - (learning_rate * (output - label_input) * data_input)
	return weights

def epoch_train_slp(data_input,label_input, learning_rate,weights, confusion_matrix,label_test,test_input, epoch_input, train_accuracy, test_accuracy):
	"""
	Runs an epoch given the data set and randomizes training set each time
	"""
	#randomize function
	training_pointer = np.arange(label_input.shape[0])#creates an ordered array to point to the training set
	testing_pointer = np.arange(label_test.shape[0])#creates an ordered array to to point to the testing set
	for epoch in range(epoch_input):
		np.random.shuffle(training_pointer)# randomize the array
		for x in range(label_input.shape[0]):
			weights = slp_weight_update(data_input[(training_pointer[x]),:],weights,label_input[(training_pointer[x])],learning_rate)
		#throw back in the train data
		np.random.shuffle(training_pointer) # randomize
		for xt in range(label_input.shape[0]):
			test_slp(weights,confusion_matrix, label_input[(training_pointer[xt])], data_input[(training_pointer[xt]),:])
		train_accuracy.append(print_accuracy_slp(confusion_matrix))
		#throw in the test data
		np.random.shuffle(testing_pointer) # randomize
	#test 
		for y in range(label_test.shape[0]):
			test_slp(weights,confusion_matrix, label_test[(testing_pointer[y])], test_input[(testing_pointer[y]),:])
		test_accuracy.append(print_accuracy_slp(confusion_matrix))

	return weights;

def test_slp(weights,confusion_matrix,label_tests,input_tests):
	"""
	Does test runs after each epocha
	weights: The paused weight dataset
	confusion_matrix: This is used calculate the accuracy
	input_tests: test data
	label_tests: labels for the test data
	"""
	output = np.dot(input_tests, weights.T) #transpose to allow for dot product
	output = 1 if output > 0 else 0
	confusion_matrix[int(label_tests)][output] += 1

def print_accuracy_slp(confusion_matrix):
	"""
	Prints the accuracy of the set utilizing the confusion matrix and resets the matrix
	"""
	accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[1][0] + confusion_matrix[0][1])

	confusion_matrix.fill(0) # resets matrix

	return accuracy;


def train(maxTrainRows=25000, maxTestRows=5000):
	#starting for the program2
	#accuracy to be stored
	test_accuracy = []
	train_accuracy = []
	#initial data
	attributes = 24 #labels are included
	learning_rate = 0.1
	#importing from csv
	train = np.genfromtxt('SLP\\credit_card_train.csv',delimiter=',',max_rows=maxTrainRows)
	test = np.genfromtxt('SLP\\credit_card_test.csv',delimiter=',',max_rows=maxTestRows)

	#separating labels
	label_train = add_label(train,(train.shape[0]))
	label_test = add_label(test,(test.shape[0]))

	#preprocess data

	train = preprocess(train)
	test = preprocess(test)

	#create weight matrix

	weights_train = create_weights_slp(attributes) #24 one for the bias and only one output

	#confusion matrix creation

	conf_matrix = confusion_matrix_create(2)

	#epoch train
	weights_train = epoch_train_slp(train,label_train, learning_rate,weights_train, conf_matrix,label_test,test,200,train_accuracy,test_accuracy)


	#Print the results to the screen in a flat format
	print("Training accuracy: ", end="")
	print(*train_accuracy, sep=",")
	print("Test accuracy: ", end="")
	print(*test_accuracy, sep=",")

	"""
	#export file
	with open("train_accuracy.txt",'w') as e:
		e.write(str(train_accuracy))
		e.write("\n")
		e.close()

	with open("test_accuracy.txt",'w') as f:
		f.write(str(test_accuracy))
		f.write("\n")
		f.close()
	"""

