'''
Training Naive Bayes Classifier for our final project CS545 taught by Anthony Rhodes
Daniel Connelly, Dalton Boehnig, Ebele Esimai, Dawei Zhang, Daniel Lee
Fall 2019
'''
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
from time import sleep

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

    # TODO -- is this function giving a wrong result?
    #print(confusion_matrix(target_list,class_choice)) # similar to program #1
    conf_matrix = confusion_matrix(target_list,class_choice) # similar to program #1

    tp = conf_matrix[1][1]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]
    tn = conf_matrix[0][0]
    conf_matrix = np.array([[tp, fp], [fn, tn]]) # so our TP is actually defaulting

    print(conf_matrix)
    print("Tp, fp, fn, tn: ",tp,fp,fn,tn)
    print("Accuracy: {}\nPrecision: {}\nRecall: {}".format((tp+tn)/7500 * 100, tp / (tp+fp),tp/(tp+fn)))

def main():
    # Set to true to read in credit data.
    optional = True

    # part 1 -- reading in data
    import preprocess
    examples, bad, good, test = preprocess.preproc()

    # to print out shapes of our data to verify ratios
    print("Test(5841 Good and 1659 Bad): " + str(np.shape(test)))
    print("Train Good: " + str(np.shape(good)))
    print("Train Bad: " + str(np.shape(bad)))

    #[] part 2 -- create probabilistic model
    min_std_dev = 0.01

    bad_prior, good_prior = prior_prob(good,bad) # note: last column does not affect calculations
    print("Prior of the bad credit score: {}. Prior of the good: {}.".format(bad_prior, good_prior))

    #*- mean and std dev (of train set) for each class (bad credit,good credit)
    bad_credit_mean, bad_stddev = prob_model(bad)
    good_credit_mean, good_stddev = prob_model(good)

    bad_stddev = np.clip(bad_stddev, min_std_dev, None) # enforces minimum std_devs to be > 0.
    good_stddev = np.clip(good_stddev, min_std_dev, None) # '                                 '

    # OPTIONAL STEP 1 -- Load in our own data
    if optional:
        daniel = np.genfromtxt("data/Daniel_Connelly_row.txt", delimiter=',')
        examples = np.append(examples, daniel)

    #[] part 3 -- Run NB on test data -- gives P(x_i | class) for each class it is given
    class_bad = class_NB(bad_credit_mean, bad_stddev, test, bad_prior)
    class_good = class_NB(good_credit_mean, good_stddev, test, good_prior)

    # OPTIONAL STEP 2 -- result data for Daniel
    if optional:
        if class_bad[7499] > class_good[7499]: print ("Daniel is a credit delinquent.")
        else: print("Daniel is not a credit delinquent.")

    #* find argmax of decisions (bad credit or good credit) for each class of test set. One loop due to even # of data
    class_choice = []
    for i in range(0, len(class_bad)):
        if class_bad[i] > class_good[i]: class_choice.append(1.0)
        if class_bad[i] <= class_good[i]: class_choice.append(0.0)

    #* Results
    conf_matrix(test, class_choice)

if __name__=="__main__":
    main()
