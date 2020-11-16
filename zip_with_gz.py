import gzip
import shutil

path_from = 'ex2.py'
path_to = 'ex2.py.gz'

with open(path_from, 'rb') as f_in:
    with gzip.open(path_to, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)