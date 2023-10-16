# write a script to compare the pk file contents from files in 2 folders
# and output the differences to a file

import pickle as pk
import numpy as np

file1 = './original_data/azimuth_ifft.pk'
file2 = './optimised_data/azimuth_ifft.pk'


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
print('max absolute diff: ', np.max(diff))

# print count of values where diff >0.01
print('count of values where diff > 0.0000001: ', np.sum(diff > 1E-6))

# print length
print('length: ', len(pk1))

# print relative diff
print('max relative diff: ', np.max(diff/(np.abs(pk1)+1E-16)))

def mean_absolute_error(pk1, pk2):
    return np.mean(np.abs(pk1 - pk2))

def root_mean_squared_error(pk1, pk2):
    return np.sqrt(np.mean(np.square(pk1 - pk2)))


def phase_correlation(pk1, pk2):
    cross_power_spectrum = np.conj(pk1) * pk2
    phase_correlation_map = np.fft.ifft2(cross_power_spectrum / (np.abs(pk1) * np.abs(pk2) + 1E-16))
    return np.max(phase_correlation_map)


mae = mean_absolute_error(pk1, pk2)
rmse = root_mean_squared_error(pk1, pk2)
print('mae: ', mae)
print('rmse: ', rmse)
# print('phase_corr', phase_correlation(pk÷1, pk2))