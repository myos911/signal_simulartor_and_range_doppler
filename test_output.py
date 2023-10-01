# ----- Imports ------

import pickle as pk
import numpy as np

import unittest
import subprocess



# ----- Util ------


def load_dump(file_path: str):
    with open(file_path, 'rb') as data_dump:
        return pk.load(data_dump)

# Test Cases

class TestRangeDoppler(unittest.TestCase):

    @classmethod
    def setUp(self):
        
        #  Setup the original data
        self.original_step1_matched_filter = load_dump('./original_data/matched_filter.pk')
        self.original_step2_azimuth_fft = load_dump('./original_data/azimuth_fft.pk')
        self.original_step3_rcmc = load_dump('./original_data/rcmc.pk')
        self.original_step4_azmiuth_filter = load_dump('./original_data/azimuth_filter4.pk')
        self.original_step5_azmiuth_ifft = load_dump('./original_data/azimuth_ifft.pk')
        print('Running rangeDoppler for dumps... Please wait...')
        result = subprocess.run(["python3.8", "scriptedsympyr.py", "--dump-data"], capture_output=True, text=True)
        print('Success: ', result.returncode == 0 )

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')
          
if __name__ == '__main__':
  unittest.main()