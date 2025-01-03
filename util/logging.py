import matplotlib as mpl
from natsort import natsorted
import numpy as np
from glob import glob
import os, sys
import socket

class colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def print_hline(n=None):
    char = u'\u2500' if not socket.gethostname() == 'uzay.mit.edu' else '-'
    try:
        if n is None:
            n = 50
            #n = os.get_terminal_size().columns
        print(char * n)
    except OSError:
        print(char * 100)

def fstr_ratio(num, den, digits=1):
    """Returne string formatted with ratio to percentage"""
    return f'{num}/{den}≈{round(num/den*100, digits)}%'

def print_bold(text, color=None):
    if color not in ['red']:
        print(f'{colors.BOLD}{text}{colors.END}')
    if color =='red':
        print(f'{colors.BOLD}{colors.RED}{text}{colors.END}')