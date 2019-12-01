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

  installModules()

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
  outfilename = 'output/bayes.txt'
  output(f'Executing bayes (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    sys.stdout = sys.stderr = outfile
    import bayes
    bayes.main()
    
def slp():
  outfilename = 'output/slp.txt'
  output(f'Executing slp (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    sys.stdout = sys.stderr = outfile
    print('TODO')

def mlp():
  outfilename = 'output/mlp.txt'
  output(f'Executing mlp (output in {outfilename})')
  with open(outfilename, 'w+') as outfile:
    sys.stdout = sys.stderr = outfile
    import neural
    neural.main()
    sys.stdout = sys.__stdout__
    sys.stdrr = sys.__stderr__
  with open(outfilename, 'r') as infile:
    accuracy_train = []
    accuracy_test = []
    for line in infile:
      match = re.search('accuracy: \(train (.*), test (.*)\)', line)
      if match:
        accuracy_train.append(match.group(1))
        accuracy_test.append(match.group(2))
    #TODO: plot with pylab


'''
  Makes sure we have the necessary modules installed.
'''
def installModules():
  output('Checking for required modules... ', end='')

  try:
    import sklearn
  except ImportError as error:
    print('sklearn not found. Installing...')
    p = subprocess.run('pip install sklearn', shell=True)
    
  output('Done')


'''
  Allows output to be easily turned on or off.
'''
def output(s, end='\n'):
  if not quiet: print(s, end=end, flush=True)


if __name__=="__main__":
  main(sys.argv)
