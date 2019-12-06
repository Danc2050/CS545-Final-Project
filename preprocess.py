import numpy as np
def preproc():
    '''
    # ensuring proper ratios
      # Since we have a small amount of test and train, I decided to make a 3:1 ratio of good + bad train data and good + bad test data. This makes the entire train and test data 3:1.
        # Good data ratio
            # If we use 3/4 * 23364 = 17523
            # 1/4 * 23364 = 5841
            # This is 3(train):1(test) of the good creditors data
        # Bad data ratio
            # 3/4 * 6636 = 4977
            # 1/4 * 6636 = 1659
            # This is 3(train):1(test) of the bad creditors data,
    '''
    bad = []; good = []; test = [];
    data = np.genfromtxt("data/data.csv", delimiter=',') # There are 6636 "bad" credit examples, 23364 "good" examples
    np.random.shuffle(data)

    for line in data:
        if len(good) < 17523 and line[24] == 0.0: good.append(line)
        elif len(bad) < 4977 and line[24] == 1.0: bad.append(line) # 3000 is close to 3:1/train:test
        else: test.append(line)

    # conversion of arrays to numpy arrays
    bad = np.asarray(bad, dtype=np.float128)
    good = np.asarray(good, dtype=np.float128)
    test = np.asarray(test, dtype=np.float128)

    return data, bad, good, test
