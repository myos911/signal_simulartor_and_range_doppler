# write a script to compare the pk file contents from files in 2 folders
# and output the differences to a file

import pickle as pk
import numpy as np

file1 = './original_data/azimuth_ifft.pk'
file2 = './retrace_data/azimuth_ifft.pk'


# open the files
with open(file1, 'rb') as f1:
    pk1 = pk.load(f1)

with open(file2, 'rb') as f2:
    pk2 = pk.load(f2)
# print type
# print('pk1: ', type(pk1))
# print('pk2: ', type(pk2))
# print
print('pk1: ', pk1)
print('pk2: ', pk2)

# print diff
print('pk1 - pk2: ', np.abs(pk1 - pk2))

# print max diff
print('max diff: ', np.max(np.abs(pk1 - pk2)))

# print relative diff
print('max relative diff: ', np.max(np.abs(pk1 - pk2)/np.abs(pk1)))
