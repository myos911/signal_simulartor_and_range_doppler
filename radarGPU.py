"""
Authors: Matan Yosef and Jia Tee

This file represents the GPU version of the radar.py file. It uses CuPy rather than NumPy for array
operations, and makes transfers as required.
"""

#  ____________________________imports_____________________________
import numpy as np
from linearFM import Chirp
from chirpGPU import ChirpGPU
from geometryRadar import RadarGeometry
from geometryRadarGPU import RadarGeometryGPU
from pprint import pprint
from antenna import Antenna
from radar import Radar
#  ____________________________utilities __________________________


# from antenna import Antenna
# ___________________________________________ Classes ______________________________________________

class RadarGPU:
    def __init__(self, radar_cpu: Radar, path='pippo'):
        ## Geometric description of radar platform


        # First, make a copy of the geometry to the GPU
        self.geometry = RadarGeometryGPU(radar_cpu.geometry)


        self.fc = radar_cpu.fc
     

        self.prf =radar_cpu.prf
        # transmitted impulse
        self.pulse = ChirpGPU(radar_cpu.pulse)
        # impulse setup
        self.antenna = radar_cpu.antenna

    
        # default initialization values print out
        # pprint('default values')
        # pprint(self.__dict__)
        # pprint('geometry: ')
        # pprint(self.geometry.__dict__)
        # pprint('pulse: ')
        # pprint(self.pulse.__dict__)

    def transmitted_signal(self, t):
        """
        compute the transmitted signal
        :param t: time axis
        :return: impulse train
        """
        return self.pulse.baseband_chirp_train_fast(t,self.prf)

    def set_prf(self, prf: float):
        self.prf = prf

    def set_carrier_frequency(self, fc):
        self.fc = fc
        self.pulse.fc = self.fc


# _____________________________Test________________________________________
if __name__ == '__main__':
    radar = Radar()