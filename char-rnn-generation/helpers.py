# https://github.com/spro/practical-pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

# wow, another unicode-related package
def read_file(filename):
    """
    read a file in unicode, return the content, and length by char
    """
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    """
    well well well, no batch is so nice!, but with torchtext, things'll be nicer
    """
    tensor = torch.zeros(len(string)).long() # type casting! as idx, LongTensor is default
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor) # shape [T]

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:d}m {:.1f}s'.format(m, s)

