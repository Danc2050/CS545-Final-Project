#!/usr/bin/env python3

'''
  Authors: Dalton, Ebele, Dawei, Daniel Lee, Daniel Connelly
  Class: CS545 - Fall 2019 | Professor Anthony Rhodes
'''

import sys
import getopt
import subprocess


assert sys.version_info >= (3, 6) #Pyhon 3 required

quiet = False

def main(argv):
  global quiet
  versions = []
  def usage():
    print(f'Usage: {argv[0]} [--quiet] [-v <bayes | slp | mlp>]')
    exit(2)

  try: 
    opts, args = getopt.getopt(argv[1:], ":v:q", ['quiet'])
  except getopt.GetoptError: 
    usage()

  #Reject extraneous arguments
  if len(args) > 0: usage()

  for opt, arg in opts:
    if opt in ('-v', '-version'):
      if arg not in ('bayes', 'slp', 'mlp'):
        usage()
      versions.append(arg)
    if opt in ('-q', '--quiet'):
      quiet = True

  versions = list(set(versions)) #Make distinct
  if versions == []: versions = ['bayes', 'slp', 'mlp'] #Run them all

  installModules()
  if 'bayes' in versions: bayes()
  if 'slp' in versions: slp()
  if 'mlp' in versions: mlp()


def bayes():
  output('\n\nExecuting bayes...')
  import bayes
  bayes.main()

def slp():
  output('\n\nExecuting slp...')
  print('TODO')

def mlp():
  output('\n\nExecuting mlp...')
  import neural
  neural.main()


def installModules():
  output('Checking for required modules... ', end='')

  try:
    import sklearn
  except ImportError as error:
    print('sklearn not found. Installing...')
    p = subprocess.run('pip3 install sklearn', shell=True)
    
  output('Done')


def output(s, end='\n'):
  if not quiet: print(s, end=end, flush=True)


if __name__=="__main__":
  main(sys.argv)
