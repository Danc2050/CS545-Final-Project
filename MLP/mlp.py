'''
Single layer multi perceptron learning model.
Authors: Daniel Connelly, Dalton Bohning, Ebele Esimai, Dawei Zhang, Daniel Lee
Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import numpy as np

class mlp:
    def __init__(self, feature_cnt: int, node_cnt: int, out_cnt: int):
        # use random number from -0.05 to 0.05 to init weights numbers
        # weights1 for input to hidden layer, weights2 for hidden to output
        self.weights1 = np.random.rand(feature_cnt + 1, node_cnt) * 0.1 - 0.05
        self.weights2 = np.random.rand(node_cnt + 1, out_cnt) * 0.1 - 0.05

    def activate(self, arr: np.array):
        return 1 / (1 + np.exp(-arr)) # sigmoid function

    def forward(self, inputs: np.array, batch: int):
        middles = self.activate(np.dot(inputs, self.weights1))
        middles = np.concatenate((middles, np.ones((batch, 1))), axis=1) 
        outputs = self.activate(np.dot(middles, self.weights2))
        return (middles, outputs)

    def train(self, inputs: np.array, labels: np.array, eta: float, mom: float, batch: int):
        count = np.shape(inputs)[0]
        assert(count == np.shape(labels)[0] and count % batch == 0)

        inputs = np.concatenate((inputs, np.ones((count,1))), axis=1) # append bias value '1's

        last_w1 = np.zeros(np.shape(self.weights1)) # delta weights1
        last_w2 = np.zeros(np.shape(self.weights2)) # delta weights2
        
        for i in range(0, count, batch):
            # grab data based on batch size
            ins_batch = inputs[i:i+batch]
            lbs_batch = labels[i:i+batch]

            # forward phase
            (middles, outputs) = self.forward(ins_batch, batch)

            # backward phase
            delta_o = outputs*(1.0-outputs)*(lbs_batch-outputs)
            delta_h = middles*(1.0-middles)*(np.dot(delta_o, np.transpose(self.weights2)))

            # update weights
            last_w2 = eta*(np.dot(np.transpose(middles), delta_o)) + mom*last_w2
            self.weights2 += last_w2
            last_w1 = eta*(np.dot(np.transpose(ins_batch), delta_h[:,:-1])) + mom*last_w1
            self.weights1 += last_w1
    
    def test(self, inputs: np.array, labels: np.array, out_cnt: int):
        count = np.shape(inputs)[0]
        assert(count == np.shape(labels)[0])
        confusion = np.zeros((out_cnt, out_cnt))      

        inputs = np.concatenate((inputs, np.ones((count,1))), axis=1) # append bias value '1's
        outputs = self.forward(inputs, count)[1]

        outputs = np.argmax(outputs, axis=1)
        labels = np.argmax(labels, axis=1)

        # calculate accuracy and confusion matrix
        n_pass = 0
        for i in range(count):
            if outputs[i] == labels[i]:
                n_pass += 1
            confusion[labels[i], outputs[i]] += 1
        accuracy = n_pass / count
            
        return (accuracy, confusion)