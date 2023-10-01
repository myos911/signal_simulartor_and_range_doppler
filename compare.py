# write a script to compare the pk file contents from files in 2 folders
# and output the differences to a file

import pickle as pk
import numpy as np

file1 = './original_data/step4_filter_matrix.pk'
file2 = './retrace_data/step4_filter_matrix.pk'


# open the files
with open(file1, 'rb') as f1:
    pk1 = pk.load(f1)

with open(file2, 'rb') as f2:
    pk2 = pk.load(f2)
# print type
# print('pk1: ', type(pk1))
# print('pk2: ', type(pk2))
print('pk1: ', pk1)
print('pk2: ', pk2)

diff = np.abs(pk1 - pk2)

# print diff
print('pk1 - pk2: ', diff)

# print max diff
print('max diff: ', np.max(diff))

# print count of values where diff >0.01
print('count of values where diff > 0.0000001: ', np.sum(diff > 0.000001))

# print length
print('length: ', len(pk1))

# print relative diff
print('max relative diff: ', np.max(diff/np.abs(pk1)))
