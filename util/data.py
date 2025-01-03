"""
Utilities for data handling and IO
"""

import dill as pickle
from datetime import datetime, timezone, timedelta
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.stats import sigma_clip
from canislib.misc import print_bold
import traceback

def datetime_now():
    now = datetime.now(tz = timezone(timedelta(hours=-7)))
    return now.strftime('%a%b%d_%H:%M')

def fitsload(fn, hdu=1, verbose=True):
    data = fits.open(fn)[hdu].data
    if verbose:
        print(f'🌌: Loaded {type(data)} from {fn} hdu {hdu}')
    return data

def psave(data, fn, verbose=True):
    try:
        with open(fn, 'wb') as f:
            pickle.dump(data, f)
            if verbose:
                print(f'💽 Pickled {type(data)} to {fn}')
    except TypeError: # TypeError: cannot pickle '_abc._abc_data' object
        traceback.print_exc()
        print('Falling back to regular pickle instead of dill')
        pickle_save(data, fn, verbose=True)

def pload(fn, verbose=True):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        if verbose:
            print(f'💽 Loaded {type(data)} from {fn}')
        return data

def cat(fn, silent=False):
    """Print contents of a file"""
    with open(fn, 'r') as fin:
        txt = fin.read()
        if not silent:
            print(txt)
        return txt

def textload(fn):
    with open(fn, 'r') as fin:
        return fin.read()

def textsave(text, fn):
    """ cat > fn"""
    with open(fn, 'w') as fout:
        fout.write(text)


def pickle_save(data, fn, verbose=True):
    import pickle as pickle_not_dill
    with open(fn, 'wb') as f:
        pickle_not_dill.dump(data, f)
        if verbose:
            print(f'💽 Successfully pickled {type(data)} to {fn}')

def nsave(data, fn):
    with open(fn, 'wb') as f:
        np.save(f, data)

def nload(fn):
    with open(fn, 'rb') as f:
        return np.load(f)