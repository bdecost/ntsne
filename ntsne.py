""" ntsne.py
    numpy wrapper for bh_tsne (https://github.com/lvdmaaten/bhtsne)
    Brian DeCost bdecost@andrew.cmu.edu
"""
import os
import re
import shutil
import struct
import subprocess
import numpy as np

# vdM's bh_tsne reads and writes from/to these hardcoded paths
# in the directory from which bh_tsne is run
DATAFILE = 'data.dat'
RESULTFILE = 'result.dat'
TSNESOURCE = 'https://github.com/lvdmaaten/bhtsne'

# check to see if bh_tsne in on system PATH or in the CWD;
# if not, clone and build it in ~/.ntsne if necessary
augmented_path = os.getenv('PATH') + ':{}'.format(os.getcwd())
TSNE = shutil.which('bh_tsne', path=augmented_path)
if TSNE is None:
    TSNEDIR = os.path.expanduser('~/.ntsne')
    TSNE = os.path.join(TSNEDIR, 'bh_tsne')

# set default bh_tsne parameters
THETA = 0.5
PERPLEXITY = 30
MAP_DIMS = 2
MAX_ITER = 1000
SEED = None
    
def build_bhtsne():
    """ clone and build lvdmaaten's bhtsne """
    subprocess.call(['git', 'clone', TSNESOURCE, TSNEDIR])
    subprocess.call(['g++', 'sptree.cpp', 'tsne.cpp', '-o', 'bh_tsne', '-O2'], cwd=TSNEDIR)
    return

def write_tsne_input(X, theta=THETA, perplexity=PERPLEXITY, map_dims=MAP_DIMS, max_iter=MAX_ITER, seed=SEED):
    """ serialize 2D data matrix (numpy array) with t-SNE options to vdM's binary input format """
    with open(DATAFILE, 'wb') as f:
        n, d = X.shape
        f.write(struct.pack('=i', n))   # number of instances
        f.write(struct.pack('=i', d))   # initial dimensionality
        f.write(struct.pack('=d', theta))
        f.write(struct.pack('=d', perplexity))
        f.write(struct.pack('=i', map_dims))
        f.write(struct.pack('=i', max_iter))
        f.write(X.tobytes()) 

        if seed is not None:
            f.write(struct.pack('=i', map_dims))
            
def read_tsne_results():
    """ load t-SNE results from vdM's binary results file format """
    with open(RESULTFILE, 'rb') as f:
        n, = struct.unpack('=i', f.read(4))  # number of instances
        md, = struct.unpack('=i', f.read(4)) # map dimensionality        
        sz = struct.calcsize('=d')
        x_tsne = [struct.unpack_from('=d', f.read(), sz*offset)
                  for offset in range(n*md)]
        
    return np.array(x_tsne).reshape((n,md))

def tsne(X, theta=THETA, perplexity=PERPLEXITY, map_dims=MAP_DIMS, max_iter=MAX_ITER, seed=SEED):
    """ simple wrapper function for applying bh_tsne to the data matrix X """
    write_tsne_input(X, perplexity=perplexity, theta=theta)
    subprocess.call(TSNE)
    return read_tsne_results()

def tsne_error(results):
    """ find the error string for each iteration; get the min (likely the last iteration...) """
    errorstrings = re.findall('error is \d+\.\d+', results.decode())
    # lexicographic ordering is equivalent to splitting each sting and converting to float...
    error = min(errorstrings).split()[-1]
    return float(error)

def best_tsne(X, theta=THETA, perplexity=PERPLEXITY, map_dims=MAP_DIMS, max_iter=MAX_ITER, seed=SEED, n_iter=10):
    """ run bh_tsne {n_iter} times and return results with lowest KL divergence """
    write_tsne_input(X, theta=theta, perplexity=perplexity,
                     map_dims=map_dims, max_iter=max_iter, seed=seed)
    lowest_error = 1e9
    x_tsne = None
    for iteration in range(n_iter):
        results = subprocess.check_output(TSNE)
        error = tsne_error(results)
        print('error is {}'.format(error))
        if error < lowest_error:
            lowest_error = error
            x_tsne = read_tsne_results()
    return x_tsne

if not os.path.isfile(TSNE):
    print('bh_tsne not found; cloning from {}'.format(TSNESOURCE))
    build_bhtsne()
