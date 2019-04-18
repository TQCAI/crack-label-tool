import os
from scipy.misc import imsave

finish_file='/finish_list.txt'

def dump_finish_record(dir,file):
    with open(dir+finish_file,'a+') as f:
        f.write(file+'\n')

def load_finish(dir):
    if not os.path.exists(dir + finish_file): return []
    with open(dir + finish_file, 'r') as f:
        return f.read().split('\n')

if __name__ == '__main__':
    pass