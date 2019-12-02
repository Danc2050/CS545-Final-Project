#!python

'''
  Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
  Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''
import sys
import getopt
import subprocess
import multiprocessing as mp
import re
from os import path
import numpy as np

try:
    import sklearn
except ImportError as error:
  p = subprocess.run('pip install sklearn', shell=True)
  import sklearn
try:
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
except ImportError as error:
  p = subprocess.run('pip install matplotlib', shell=True)
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm


assert sys.version_info >= (3, 6) #Pyhon 3 required

quiet = False
serial = False


def main(argv):
  global quiet
  global serial
  versions = []
  def usage():
    print(f'Usage: {argv[0]} [--quiet] [--serial] [-v <bayes | slp | mlp>]')
    exit(2)

  try: 
    opts, args = getopt.getopt(argv[1:], ":v:qs", ['quiet', 'serial'])
  except getopt.GetoptError: 
    usage()

  #Reject extraneous arguments
  if len(args) > 0: usage()

  for opt, arg in opts:
    if opt in ('-v', '-version'):
      if arg not in ('bayes', 'slp', 'mlp'):
        usage()
      versions.append(arg)
    if opt in ('-q', '--quiet'):  quiet = True
    if opt in ('-s', '--serial'): serial = True

  versions = list(set(versions)) #Make distinct
  if versions == []: versions = ['bayes', 'slp', 'mlp'] #Run them all

  if serial:
    if 'bayes' in versions: bayes()
    if 'slp' in versions:   slp()
    if 'mlp' in versions:   mlp()
  else:
    processes = []
    if 'bayes' in versions: processes.append(mp.Process(target=bayes))
    if 'slp' in versions:   processes.append(mp.Process(target=slp))
    if 'mlp' in versions:   processes.append(mp.Process(target=mlp))
    for p in processes: p.start()
    for p in processes: p.join()


def bayes():
  outfilename = path.join('output', 'bayes.txt')
  output(f'Executing bayes (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    redirectOutputToFile(outfile)
    import bayes
    bayes.main()
    restoreStandardOutput()
    
def slp():
  outfilename = path.join('output', 'slp.txt')
  pngfilename = path.join('output', 'slp.png')
  output(f'Executing slp (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    redirectOutputToFile(outfile)
    from SLP import slp
    slp.train()
    restoreStandardOutput()
  with open(outfilename, 'r') as infile:
    accuracy_train = []
    accuracy_test = []
    for line in infile:
      match = re.search('Training accuracy: (.*)', line)
      if match: 
        accuracy_train = np.array(match.group(1).split(','), dtype=np.float)
        continue
      match = re.search('Test accuracy: (.*)', line)
      if match: 
        accuracy_test = np.array(match.group(1).split(','), dtype=np.float)

    if accuracy_train.shape[0] == 0 or accuracy_test.shape[0] == 0:
      print("ERROR: Could not find training and test accuracy")
      return
    plot(accuracy_train, accuracy_test, pngfilename)


def mlp():
  outfilename = path.join('output', 'mlp.txt')
  pngfilename = path.join('output', 'mlp.png')
  output(f'Executing mlp (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    redirectOutputToFile(outfile)
    import neural
    neural.main()
    restoreStandardOutput()
  with open(outfilename, 'r') as infile:
    accuracy_train = []
    accuracy_test = []
    for line in infile:
      match = re.search('accuracy: \(train (.*), test (.*)\)', line)
      if match:
        accuracy_train.append(match.group(1))
        accuracy_test.append(match.group(2))
    accuracy_train = np.array(accuracy_train, dtype=np.float)
    accuracy_test = np.array(accuracy_test, dtype=np.float)

    if accuracy_train.shape[0] == 0 or accuracy_test.shape[0] == 0:
      print("ERROR: Could not find training and test accuracy")
      return
    plot(accuracy_train, accuracy_test, pngfilename)


#Plot the accuracies with pyplot
def plot(accuracy_train, accuracy_test, pngfilename):
  colors = iter(cm.rainbow(np.linspace(0, 1, 2))) #2 colors
  plt.plot(range(accuracy_train.shape[0]), accuracy_train, color=next(colors), label="train={:.2%}".format(accuracy_train[-1]))
  plt.plot(range(accuracy_test.shape[0]), accuracy_test, color=next(colors), label="test={:.2%}".format(accuracy_test[-1]))
  plt.legend(loc='lower right')
  plt.ylim(np.amin([accuracy_test, accuracy_train])-0.1, np.amax([accuracy_test, accuracy_train])+0.1)
  plt.savefig(pngfilename)
  plt.clf()
  output(f'Plot in {pngfilename}')


'''
  Redirects standard output and errors to the file supplied
'''
def redirectOutputToFile(f):
  sys.stdout = sys.stderr = f

'''
  Restores output to the standard (screen) output in case it was redirected.
'''
def restoreStandardOutput():
  sys.stdout = sys.__stdout__
  sys.stderr = sys.__stderr__

'''
  Allows output to be easily turned on or off.
'''
def output(s, end='\n'):
  if not quiet: print(s, end=end, flush=True)


if __name__=="__main__":
  main(sys.argv)
