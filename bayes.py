'''
Training Naive Bayes Classifier for our final project CS545 taught by Anthony Rhodes
Daniel Connelly, Dalton Boehnig, Ebele Esimai, Dawei Zhang, Daniel Lee
Fall 2019
'''
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

def prior_prob(good, bad):
    '''
    Compute every positive and negative (1 and 0) in our data set. Since our categories are binary, we find out our prior probability by the following formula: class / total sum of classes
    '''
    positive = negative = total = 0

    for i in range(np.shape(bad)[0]):
        if bad[i][24] == 1.0: positive += 1; total +=1;

    for i in range(np.shape(good)[0]):
        if good[i][24] == 0.0: negative += 1; total +=1;

    return np.divide(positive, total, dtype=np.float128), np.divide(negative, total, dtype=np.float128)

def prob_model(features): return np.mean(features, axis=0, dtype=np.float128), np.std(features, axis=0, dtype=np.float128)

def nb(mean, std_dev, feature):
    '''one liner naive bayes for each feature in the feature set.
       A mean is the mean of that class. A std_dev is the std_dev of that class. A feature is x_i.
       Using np functions are important as they allow for greater precision (128)
       Note: The dtype=np.float128 is necessary, but not to the extent I have done so (i.e., not every function needs it).
    '''
    return np.multiply(np.divide(1,((np.multiply(np.sqrt(np.multiply(2,np.pi, dtype=np.float128), dtype=np.float128),std_dev, dtype=np.float128))), dtype=np.float128),np.exp(-(((np.power(feature-mean, 2, dtype=np.float128)))/(np.multiply(2, (np.power(std_dev,2,dtype=np.float128))))), dtype=np.float128), dtype=np.float128)

def class_NB(mean, stddev, test, prior):
    class_type = [] # could be a bad credit person or good credit person. This function is generic, so we leave the name of this generic
    i,j = test.shape
    # for each example, compute the NB of the likelihood that it is POS (bad credit person)
    for a in range(0,i):
        feature = test[a][:-1] # we must not calculate the later column (the indicator of bad credit or good credit)
        summation = 0
        for b in range(0,j-1): # '     '
            nb_result = nb(mean[b], stddev[b], feature[b])
            if nb_result == 0:
                nb_result = 1
            summation += np.log10(nb_result) + np.log10(prior)
        class_type.append(summation)
    return class_type

def conf_matrix(test, class_choice):
    target_list = [];
    for i in range(np.shape(test)[0]):
        target_list.append(test[i][24])

    conf_matrix = confusion_matrix(target_list,class_choice) # similar to program #1
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tn = conf_matrix[1][1]

    print(conf_matrix)
    print("Tp, fp, fn, tn: ",tp,fp,fn,tn)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}".format((tp+tn)/15000 * 100, tp / (tp/fp),tp/(tp+fn)))

def main():
    # part 1 -- reading in data
    # non-randomized version

    # randomized version
    bad = []; good = []; test = [];
    examples = np.genfromtxt("data/data.csv", delimiter=',')
    np.random.shuffle(examples)
    for line in examples:
        if len(test) != 15000: # There is an odditiy here... if I place it last, I only get good credit owners...
            test.append(line)
        elif line[24] == 0.0 and len(good) != 7500: good.append(line)
        elif line[24] == 1.0 and len(bad) != 7500: bad.append(line)

    # conversion of arrays to numpy arrays
    bad = np.asarray(bad, dtype=np.float128)
    good = np.asarray(good, dtype=np.float128)
    test = np.asarray(test, dtype=np.float128)

    '''
    #The below is for testing purposes only
    test_good = []; test_bad = [];
    print(np.shape(bad))
    print(np.shape(good))
    print(np.shape(test))
    import time
    for line in test:
        print(line[24])
        if line[24] == 0.0: test_good.append(line)
        elif line[24] == 1.0: test_bad.append(line)

    test_bad = np.asarray(test_bad, dtype=np.float128)
    test_good = np.asarray(test_good, dtype=np.float128)

    print(np.shape(test_bad))
    print(np.shape(test_good))
    test_bad, test_good = prior_prob(test_good, test_bad)
    print(test_bad, test_good)
    sys.exit()
    #end testing purpose
    '''

    #[] part 2 -- create probabilistic model
    min_std_dev = 0.01

    bad_prior, good_prior = prior_prob(good,bad) # note: last column does not affect calculations
    print("Prior of the bad credit score: {}. Prior of the good: {}.".format(bad_prior, good_prior))

    #*- mean and std dev (of train set) for each class (bad credit,good credit)
    bad_credit_mean, bad_stddev = prob_model(bad)
    good_credit_mean, good_stddev = prob_model(good)

    for i in range(len(bad_stddev)):
        if bad_stddev[i] == 0:
           #print(bad_stddev[i])
           bad_stddev[i] = min_std_dev # Set any 0 std dev to min_std_dev to avoid divide by zero error
    # good credit
    for i in range(len(good_stddev)):
        if good_stddev[i] == 0:
           #print(good_stddev[i])
           good_stddev[i] = min_std_dev # Set any 0 std dev to min_std_dev to avoid divide by zero error

    #[] part 3 -- Run NB on test data -- gives P(x_i | class) for each class it is given
    class_bad = class_NB(bad_credit_mean, bad_stddev, test, bad_prior)
    class_good = class_NB(good_credit_mean, good_stddev, test, good_prior)

    #* find argmax of decisions (bad credit or good credit) for each class of test set. One loop due to even # of data
    class_choice = []
    for i in range(0, len(class_bad)):
        if class_bad[i] > class_good[i]: class_choice.append(1)
        if class_bad[i] <= class_good[i]: class_choice.append(0)

    #* Results
    conf_matrix(test, class_choice)

if __name__=="__main__":
    main()
