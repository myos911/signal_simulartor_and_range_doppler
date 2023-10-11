# ----- Imports ------

import pickle as pk
import numpy as np

import unittest
import subprocess
import argparse



# ----- Util ------


def load_dump(file_path: str):
    with open(file_path, 'rb') as data_dump:
        return pk.load(data_dump)

# Test Cases

class TestRangeDoppler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #  Setup the original data
        cls.original_step1_matched_filter = load_dump('./original_data/matched_filter.pk')
        cls.original_step2_azimuth_fft = load_dump('./original_data/azimuth_fft.pk')
        cls.original_step3_rcmc = load_dump('./original_data/rcmc.pk')
        cls.original_step4_azmiuth_filter = load_dump('./original_data/azimuth_filter4.pk')
        cls.original_step5_azmiuth_ifft = load_dump('./original_data/azimuth_ifft.pk')
        
        cls.tolerance = 1E-6 # Set the tolerance threshold (e.g., 1E-6)
        
        is_running_algorithm = True
        if is_running_algorithm:
            print('Running algorithm... Please wait...')
            result = subprocess.run(["python3.8", "scriptedsympyr.py", "--dump-data"], capture_output=True, text=True)
            print('SUCCESSFUL RUN?: ', result.returncode == 0)
        else:
            print('Algorithm execution is disabled. Using most recent retrace_data dump')

    def test_step1_matched_filter(self):

        # Load the dump
        retrace_step1_matched_filter = load_dump('./retrace_data/matched_filter.pk')

        absolute_difference = np.abs(retrace_step1_matched_filter - self.original_step1_matched_filter)

        # Check if all elements in the relative difference array are within the tolerance
        self.assertEqual(retrace_step1_matched_filter.shape, self.original_step1_matched_filter.shape)
        self.assertTrue(np.all(absolute_difference <= self.tolerance))
    
    def test_step2_azimuth_fft(self):

        # Load the dump
        retrace_step2_azimuth_fft = load_dump('./retrace_data/azimuth_fft.pk')
        absolute_difference = np.abs(retrace_step2_azimuth_fft - self.original_step2_azimuth_fft)
        # Check if all elements in the relative difference array are within the tolerance
        self.assertEqual(retrace_step2_azimuth_fft.shape, self.original_step2_azimuth_fft.shape)
        self.assertTrue(np.all(absolute_difference <= self.tolerance))

    def test_step3_rcmc(self):

        # Load the dump
        retrace_step3_rcmc = load_dump('./retrace_data/rcmc.pk')
        absolute_difference = np.abs(retrace_step3_rcmc - self.original_step3_rcmc)
        # Check if all elements in the relative difference array are within the tolerance
        self.assertEqual(retrace_step3_rcmc.shape, self.original_step3_rcmc.shape)
        self.assertTrue(np.all(absolute_difference <= self.tolerance))
    
    def test_step4_azimuth_filter(self):

        # Load the dump
        retrace_step4_azmiuth_filter = load_dump('./retrace_data/azimuth_filter4.pk')
        absolute_difference = np.abs(retrace_step4_azmiuth_filter - self.original_step4_azmiuth_filter)
        # Check if all elements in the relative difference array are within the tolerance
        self.assertEqual(retrace_step4_azmiuth_filter.shape, self.original_step4_azmiuth_filter.shape)
        self.assertTrue(np.all(absolute_difference <= self.tolerance))

    def test_step5_azmiuth_ifft(self):
        # Load the dump
        retrace_step5_azmiuth_ifft = load_dump('./retrace_data/azimuth_ifft.pk')
        absolute_difference = np.abs(retrace_step5_azmiuth_ifft - self.original_step5_azmiuth_ifft)
        # Check if all elements in the relative difference array are within the tolerance
        self.assertEqual(retrace_step5_azmiuth_ifft.shape, self.original_step5_azmiuth_ifft.shape)
        self.assertTrue(np.all(absolute_difference <= self.tolerance))
          
if __name__ == '__main__':
    unittest.main()