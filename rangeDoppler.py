import matplotlib.pyplot as plt
from numba import prange, jit, cuda, prange, complex64, float64
import numba
import math
import cmath
from numpy import fft
import pickle as pk
import tracemalloc
import cupy as cp

from channel import Channel
from dataMatrix import Data
from channelGPU import ChannelGPU
from dataMatrixGPU import DataGPU
from interpolator_v2 import sphere_interp
from radar import Radar
import numpy as np
from utils import meshSph2cart, mesh_lcs_to_gcs, mesh_doppler_to_azimuth, mesh_azimuth_range_to_ground_gcs, \
    mesh_gcs_to_lcs, meshCart2sph


########################################## JIT FUNCTIONS ##############################
# create filter matrix
# @jit(nopython=True, parallel=True)
def az_filter_matrix_gpu(filter_matrix, t_range_axis, speed, lamb_c, B, prf, doppler_centroid, linearfmapprox):
    if len(filter_matrix[:, 0]) % 2 == 0:
        # this is only valid if we have an even number of samples
        doppler_axis = cp.linspace(-prf / 2,
                                   (1 - 1 / len(filter_matrix[:, 0])) * prf / 2,
                                   len(filter_matrix[:, 0]))
    else:
        # this is for the case we have a odd number of samples (always)
        doppler_axis = cp.linspace(-prf / 2,
                                   prf / 2,
                                   len(filter_matrix[:, 0]))

    # this tells us which replica of the spectrum we are looking at
    foffset = cp.ceil(doppler_centroid / prf) * prf

    # print("foffset in filter matrix", foffset)
    # this centers the axis on the correct replica
    doppler_axis += foffset
    # this wraps around the freq axis to make sure we are filtering the power from the correct replica even if the
    # doppler centroid is not perfectly centered in the axis
    doppler_axis_out = cp.where(doppler_axis > doppler_centroid + prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] -= prf
        doppler_axis_out = cp.where(doppler_axis < doppler_centroid - prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] += prf
    # wrapped around range axis and fast time
    c = 299792458  # m/s
    f_range_axis = t_range_axis % (c / (2 * prf))
    # for every possible range bin
    tot_bins = len(filter_matrix[0, :])
    for rr in range(tot_bins):
        # the range is
        rng = t_range_axis[rr]
        # false range
        frng = f_range_axis[rr]
        Ka = - 2 / lamb_c * speed ** 2 / rng
        if linearfmapprox == True:
            # uncomment one of the following:
            # -> filter with range bin phase removal
            # G = 2/B * np.sqrt(2 * np.abs(Ka)) * np.exp(- 1j * np.pi / 4) * np.exp(1j * 4 * np.pi * rng / lamb_c) \
            # * np.exp(1j * np.pi * doppler_axis **2 / Ka)
            # -> filter with amplitude normalization
            # G = np.sqrt(np.abs(Ka)) / B * np.exp(-1j * np.pi / 4) * \
            #    np.exp(1j * np.pi * doppler_axis ** 2 / Ka)
            # -> filter without amplitude normalization
            G = cp.exp(-1j * cp.pi / 4) \
                * cp.exp(1j * cp.pi * doppler_axis ** 2 / Ka) \
                * cp.exp(1j * + 4 * cp.pi * frng * doppler_axis / c)
        else:
            # filter that uses non linear FM POSP approximation ( with range bin phase removal)
            # -> filter with actual posp amplitude normalization)
            # norm = (4 * speed ** 2 - doppler_axis ** 2 * lamb_c ** 2) ** (3 / 4) / np.sqrt(4 * lamb_c * speed * rng)
            # G = norm * np.sqrt(np.abs(Ka)) * np.exp(1j * (4 * np.pi * rng / lamb_c *
            #                                        np.sqrt(1 - lamb_c ** 2 * doppler_axis ** 2 / (4 * speed ** 2))
            #                                        + np.pi / 4))
            # -> filter with some amplitude normalization)
            # G = np.sqrt(np.abs(Ka)) * np.exp(1j * (4 * np.pi * rng / lamb_c *
            #                                        np.sqrt(1 - lamb_c ** 2 * doppler_axis ** 2 / (4 * speed ** 2))
            #                                        + np.pi / 4))
            # -> filter without amplitude normalization ( with range bin phase removal)
            G = cp.exp(1j * (4 * cp.pi * rng / lamb_c *                                         # range bin center phase
                             cp.sqrt(1 - lamb_c ** 2 * doppler_axis ** 2 / (4 * speed ** 2))    # doppler shift phase removal
                             - cp.pi / 4                                                        # constant phase
                             - 4 * cp.pi * frng * doppler_axis / c))                            # residual inter-pulse doppler shift removal
        # that's a column of the matrix
        filter_matrix[:, rr] = G
    return doppler_axis, filter_matrix

@cuda.jit
def az_filter_matrix_kernel(filter_matrix, doppler_axis, c, t_range_axis, f_range_axis, lamb_c, speed, linearfmapprox):
    rr = cuda.grid(1)
    if rr < filter_matrix.shape[1]:
        rng = t_range_axis[rr]
        frng = f_range_axis[rr]
        Ka = - 2 / lamb_c * speed ** 2 / rng
        for dd in range(filter_matrix.shape[0]):
            doppler_val = float64(doppler_axis[dd])
            pi = float64(cmath.pi)
            if linearfmapprox:
                G = cmath.exp(-1j * pi / 4) \
                    * cmath.exp(1j * pi * doppler_val ** 2 / Ka) \
                    * cmath.exp(1j * 4 * pi * frng * doppler_val / c)
            else:
                G = cmath.exp(1j * float64(4 * pi * rng / lamb_c *
                                    math.sqrt(1 - lamb_c ** 2 * doppler_val ** 2 / (4 * speed ** 2))
                                    - pi / 4
                                    - 4 * pi * frng * doppler_val / c))
            filter_matrix[dd, rr] = G

def az_filter_matrix_cpu(filter_matrix, t_range_axis, speed, lamb_c, B, prf, doppler_centroid, linearfmapprox):
    if len(filter_matrix[:, 0]) % 2 == 0:
        # this is only valid if we have an even number of samples
        doppler_axis = np.linspace(-prf / 2,
                                   (1 - 1 / len(filter_matrix[:, 0])) * prf / 2,
                                   len(filter_matrix[:, 0]))
    else:
        # this is for the case we have a odd number of samples (always)
        doppler_axis = np.linspace(-prf / 2,
                                   prf / 2,
                                   len(filter_matrix[:, 0]))

    # this tells us which replica of the spectrum we are looking at
    foffset = np.ceil(doppler_centroid / prf) * prf

    # print("foffset in filter matrix", foffset)
    # this centers the axis on the correct replica
    doppler_axis += foffset
    # this wraps around the freq axis to make sure we are filtering the power from the correct replica even if the
    # doppler centroid is not perfectly centered in the axis
    doppler_axis_out = np.where(doppler_axis > doppler_centroid + prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] -= prf
        doppler_axis_out = np.where(doppler_axis < doppler_centroid - prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] += prf
    # wrapped around range axis and fast time
    c = 299792458  # m/s
    f_range_axis = t_range_axis % (c / (2 * prf))
    # for every possible range bin
    tot_bins = len(filter_matrix[0, :])
    for rr in range(tot_bins):
        # the range is
        rng = t_range_axis[rr]
        # false range
        frng = f_range_axis[rr]
        Ka = - 2 / lamb_c * speed ** 2 / rng
        if linearfmapprox == True:
            G = np.exp(-1j * np.pi / 4) \
                * np.exp(1j * np.pi * doppler_axis ** 2 / Ka) \
                * np.exp(1j * + 4 * np.pi * frng * doppler_axis / c)
        else:
            G = np.exp(1j * (4 * np.pi * rng / lamb_c *                                         # range bin center phase
                             np.sqrt(1 - lamb_c ** 2 * doppler_axis ** 2 / (4 * speed ** 2))    # doppler shift phase removal
                             - np.pi / 4                                                        # constant phase
                             - 4 * np.pi * frng * doppler_axis / c))                            # residual inter-pulse doppler shift removal
        # that's a column of the matrix
        filter_matrix[:, rr] = G
    return doppler_axis, filter_matrix

# frequency range wak including the doppler error in the impulse compression peak point
@jit(nopython=True)
def r_of_f_r_dopl(f, rc, v, lam_c, c, K):
    vts = (f ** 2 * rc ** 2) / (4 * v ** 2 / lam_c ** 2 - f ** 2)
    r = np.sqrt(rc ** 2 + vts) - f * c / (2 * K)
    return r


# range cell migration compensation
@jit(nopython=True, parallel=True)
def rcmc(range_doppler_matrix, range_doppler_matrix_rcmc, t_range_axis, Fs, prf, v_sat, cc, lamb_c, rate,
         doppler_centroid):
    doppler_axis = np.linspace(-prf / 2,
                               (1 - 1 / len(range_doppler_matrix[:, 0])) * prf / 2,
                               len(range_doppler_matrix[:, 0]))
    foffset = np.ceil(doppler_centroid / prf) * prf
    doppler_axis += foffset
    # wrapped around freq axis
    doppler_axis_out = np.where(doppler_axis > doppler_centroid + prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] -= prf
        doppler_axis_out = np.where(doppler_axis < doppler_centroid - prf / 2)
    if len(doppler_axis_out) != 0:
        doppler_axis[doppler_axis_out] += prf

    # for every possible range bin
    tot_bins = len(range_doppler_matrix[0, :])
    for rr in (prange(tot_bins)):
        # print('line ', rr, ' of ', tot_bins)
        # range reference
        r_bin = t_range_axis[rr]
        # for every line of the range doppler image
        for ll in range(len(range_doppler_matrix[:, 0])):
            # the range migration is
            rm = r_of_f_r_dopl(doppler_axis[ll], r_bin, v_sat, lamb_c, cc, rate)
            # the associated delay in fast time is
            tm = (2 * rm / cc) % (1 / prf)
            # the closest fast time bin is
            t_idx = np.ceil(tm * Fs)
            # the interpolation will be performed using n bins
            n_interp = 128
            point_r = 0
            point_i = 0
            # performing the interpolation
            for n in range(int(-n_interp / 2), int(n_interp / 2)):  # this cannot be run in parallel
                # the fast time bin to include in the interpolation is
                t_n = int((t_idx + n) % tot_bins)
                # the sum argument for the interpolation is then:
                point_r += np.real(range_doppler_matrix[ll, t_n]) * np.sin(np.pi * Fs * (tm - t_n / Fs)) / (
                        np.pi * Fs * (tm - t_n / Fs))
                point_i += np.imag(range_doppler_matrix[ll, t_n]) * np.sin(np.pi * Fs * (tm - t_n / Fs)) / (
                        np.pi * Fs * (tm - t_n / Fs))
            range_doppler_matrix_rcmc[ll, rr] = point_r + 1j * point_i
    return range_doppler_matrix_rcmc

@cuda.jit
def rcmc_cuda(range_doppler_matrix, range_doppler_matrix_rcmc, t_range_axis, Fs, prf, v_sat, cc, lamb_c, rate, doppler_axis):
    # Get the thread index
    rr, ll = cuda.grid(2)
    # Get the total number of range bins and lines
    tot_bins = range_doppler_matrix.shape[1]
    tot_lines = range_doppler_matrix.shape[0]

    # Check if the thread index is within the range
    if rr < tot_bins and ll < tot_lines:
        # range reference
        r_bin = t_range_axis[rr]
    
        # the range migration is
        vts = (doppler_axis[ll] ** 2 * r_bin ** 2) / (4 * v_sat ** 2 / lamb_c ** 2 - doppler_axis[ll] ** 2)
        rm = math.sqrt(r_bin ** 2 + vts) - doppler_axis[ll] * cc / (2 * rate)
        # the associated delay in fast time is
        tm = (2 * rm / cc) % (1 / prf)
        # the closest fast time bin is
        t_idx = math.ceil(tm * Fs)
        # the interpolation will be performed using n bins
        n_interp = 128
        point_r = 0
        point_i = 0
        # performing the interpolation
        for n in range(int(-n_interp / 2), int(n_interp / 2)):
            # the fast time bin to include in the interpolation is
            t_n = int((t_idx + n) % tot_bins)
            # the sum argument for the interpolation is then:
            pi_arg = math.pi * Fs * (tm - t_n / Fs)
            sin_arg = math.sin(pi_arg) / (pi_arg)
            point_r += range_doppler_matrix[ll, t_n].real * sin_arg
            point_i += range_doppler_matrix[ll, t_n].imag * sin_arg
        range_doppler_matrix_rcmc[ll, rr] = point_r + 1j * point_i

@cuda.jit
def rcmc_cuda_optimizedY(range_doppler_matrix, range_doppler_matrix_rcmc, t_range_axis, Fs, prf, v_sat, cc, lamb_c, rate, doppler_axis):
    rr, ll = cuda.grid(2)
    tot_bins = range_doppler_matrix.shape[1]
    tot_lines = range_doppler_matrix.shape[0]
    
    if rr < tot_bins and ll < tot_lines:
         # range reference
        r_bin = t_range_axis[rr]
    
        # the range migration is
        vts = (doppler_axis[ll] ** 2 * r_bin ** 2) / (4 * v_sat ** 2 / lamb_c ** 2 - doppler_axis[ll] ** 2)
        rm = math.sqrt(r_bin ** 2 + vts) - doppler_axis[ll] * cc / (2 * rate)
        # the associated delay in fast time is
        tm = (2 * rm / cc) % (1 / prf)
        # the closest fast time bin is
        t_idx = int(math.ceil(tm * Fs))
        # the interpolation will be performed using n bins
        n_interp = 64
        point_r = 0
        point_i = 0

        
        # Allocate shared memory for range_doppler_matrix slices
        shared_matrix = cuda.shared.array(shape=(32, 32), dtype=complex64)
        
        for n in range(int(-n_interp / 2), int(n_interp / 2)):
            t_n = int((t_idx + n) % tot_bins)
            if n < 32:
                shared_matrix[ll % 32, t_n % 32] = range_doppler_matrix[ll, t_n]
            cuda.syncthreads()

            shared_value = shared_matrix[ll % 32, t_n % 32]

            sin_arg = math.sin(math.pi * Fs * (tm - t_n / Fs)) / (math.pi * Fs * (tm - t_n / Fs))
            point_r += shared_value.real * sin_arg
            point_i += shared_value.imag * sin_arg
            cuda.syncthreads()
        
        range_doppler_matrix_rcmc[ll, rr] = complex64(point_r + 1j * point_i)

########################################### CLASSES ###################################
class RangeDopplerCompressor:
    def __init__(self, channel: ChannelGPU, data: DataGPU, is_dumping: bool):
        # radar class used for the raw data simulation
        self.radar = channel.radar
        self.c = channel.c

        # data class
        self.data = data

        # sampling frequency
        self.Fs = data.Fs

        # dopplercentroid calculated assuming the antenna pattern symmetric
        gamma = channel.radar.geometry.forward_squint_angle
   
        self.doppler_centroid = 2 * self.radar.geometry.abs_v * \
                                self.radar.fc * cp.sin(gamma) / self.c

        # find the beam center range
        beta = self.radar.geometry.side_looking_angle
        height = self.radar.geometry.S_0[2]
        self.r_0 = (height / cp.cos(beta))

        # true fast time axis and enveloped time axis
        self.time_ax, self.true_fast_time_axis = self.set_true_time_ax()

        # creating empty range-doppler filter matrix
        self.azimuth_filter_matrix = None  # self.create_azimuth_filter_matrix()

        self.is_dumping = is_dumping # GLOBAL TOGGLE FROM scriptedsympyr

    # SETTERS

    def set_true_time_ax(self):
        """
        the true delay associated to a target on graund at closest approach point
        :return:  time_ax (pri-enveloped time), true_time_axis
        """
        self.time_ax = cp.linspace(0,
                                   (1 / self.radar.prf) - (1 / self.Fs),
                                   int(cp.round(self.Fs / self.radar.prf)))
        # delay at the first range bin
        min_delay = cp.floor((self.r_0 * 2 / self.c) *
                             self.radar.prf) / self.radar.prf
        self.true_fast_time_axis = self.time_ax + min_delay
        return self.time_ax, self.true_fast_time_axis

    def set_dopplercentroid(self, doppler_centroid):
        """
        override the doppler centroid value (eg in case of squinted antenna pattern)
        :param doppler_centroid: doppler centroid frequency
        :return:
        """
        self.doppler_centroid = doppler_centroid
    

    def create_azimuth_filter_matrix(self, linearFMapproximation=False):
        """
        creates the range-doppler domain filter response (no windowing, phase correction only)
        :param: linearFMapproximation Use the linear FM approx for the azimuth filter or the non-linear POSP one
        :return: filter matrix
        """

        # GPU KERNEL Version
        # self.azimuth_filter_matrix = 1j * cp.zeros((self.data.rows_num, self.data.columns_num))

        # rows = self.azimuth_filter_matrix.shape[0]
        # cols = self.azimuth_filter_matrix.shape[1]

        # prf = self.radar.prf

        # if rows % 2 == 0:
        #     # this is only valid if we have an even number of samples
        #     doppler_axis = cp.linspace(-prf / 2,
        #                             (1 - 1 / rows) * prf / 2,
        #                             rows)
        # else:
        #     # this is for the case we have a odd number of samples (always)
        #     doppler_axis = cp.linspace(-prf / 2,
        #                             prf / 2,
        #                             rows)

        # foffset = cp.ceil(self.doppler_centroid / prf) * prf
        # doppler_axis += foffset
        # doppler_axis_out = cp.where(doppler_axis > self.doppler_centroid + prf / 2)
        # if len(doppler_axis_out) != 0:
        #     doppler_axis[doppler_axis_out] -= prf
        #     doppler_axis_out = cp.where(doppler_axis < self.doppler_centroid - prf / 2)
        # if len(doppler_axis_out) != 0:
        #     doppler_axis[doppler_axis_out] += prf
    
        # c = 299792458  # m/s
        # t_range_axis = self.get_true_range_axis()
        # f_range_axis = t_range_axis % (c / (2 * prf))

        # lamb_c = self.c / self.radar.fc
        # speed = self.radar.geometry.abs_v

        # # Define grid and block dimensions
        # threads_per_block = (256,)  # 1D grid, so only one dimension
        # blocks_per_grid_x = int(cp.ceil(self.azimuth_filter_matrix.shape[1] / threads_per_block[0]))
        # blocks_per_grid = (blocks_per_grid_x,)
        # az_filter_matrix_kernel[blocks_per_grid, threads_per_block](
        #     self.azimuth_filter_matrix,
        #     doppler_axis,
        #     float64(c),
        #     t_range_axis,
        #     f_range_axis,
        #     float64(lamb_c), 
        #     float64(speed),
        #     linearFMapproximation, 
        # )
        # cuda.synchronize()
        # return doppler_axis, self.azimuth_filter_matrix


        # GPU CUPY Version
        # self.azimuth_filter_matrix = 1j * cp.zeros((self.data.rows_num, self.data.columns_num))
        # self.doppler_axis, self.azimuth_filter_matrix = az_filter_matrix_gpu(
        #     self.azimuth_filter_matrix,
        #     self.get_true_range_axis(),
        #     self.radar.geometry.abs_v,
        #     self.c / self.radar.fc,
        #     self.radar.pulse.get_bandwidth(),
        #     self.radar.prf,
        #     self.doppler_centroid,
        #     linearFMapproximation
        # )
        # return self.doppler_axis, self.azimuth_filter_matrix

        # CPU VERSION
        self.azimuth_filter_matrix = 1j * np.zeros((self.data.rows_num, self.data.columns_num))

        tra_cpu = cp.asnumpy(self.get_true_range_axis())

        dc_cpu = cp.asnumpy(self.doppler_centroid)

        self.doppler_axis, self.azimuth_filter_matrix = az_filter_matrix_cpu(
            self.azimuth_filter_matrix,
            tra_cpu,
            self.radar.geometry.abs_v,
            self.c / self.radar.fc,
            self.radar.pulse.get_bandwidth(),
            self.radar.prf,
            dc_cpu,
            linearFMapproximation
        )

        return cp.asarray(self.doppler_axis), cp.asarray(self.azimuth_filter_matrix)

    def create_azimuth_equalization_matrix(self, range_axis, doppler_axis):
        """
        projects the antenna pattern on ground and uses interpolation and posp approximation to
        find the approximated antenna pattern weighthing matrix in range - doppler domain

        :return: the filter matrix
        """
        # define a range doppler grid
        R, D = cp.meshgrid(range_axis, doppler_axis)
        # R as it is is the enveloped range, we need the true range so:
        midswath_range = self.radar.geometry.get_broadside_on_ground()
        midswath_range = self.radar.geometry.get_lcs_of_point(midswath_range, cp.zeros(1))
        midswath_range = cp.linalg.norm(midswath_range)
        pri_offset = int(self.radar.prf * midswath_range * 2 / self.c)
        range_offset = pri_offset * self.c / (2 * self.radar.prf)
        R += range_offset
        # transform to range azimuth points
        # using posp time-frequency locking relation for slow time:
        R, A = mesh_doppler_to_azimuth(R, D, self.c / self.radar.fc, self.radar.geometry.abs_v)
        # transform to ground gcs points
        X, Y = mesh_azimuth_range_to_ground_gcs(R, A, self.radar.geometry.velocity, self.radar.geometry.S_0)
        # transform to lcs points
        X, Y, Z = mesh_gcs_to_lcs(X, Y, cp.zeros_like(X), self.radar.geometry.Bc2s, self.radar.geometry.S_0)
        # transform to spherical
        R1, T, P = meshCart2sph(X, Y, Z)
        P = cp.where(P < 0, cp.pi * 2 + P, P)
        # find pattern over grid need a spherical interpolator method for meshgrids
        self.r_d_gain = 1j * cp.zeros_like(cp.ravel(T))
        self.r_d_gain = sphere_interp(cp.ravel(T),
                                      cp.ravel(P),
                                      self.radar.antenna.theta_ax,
                                      self.radar.antenna.phi_ax,
                                      self.radar.antenna.gain_matrix,
                                      self.r_d_gain)
        self.r_d_gain = self.r_d_gain.reshape(T.shape) \
                        * cp.sqrt(4 * self.c / self.radar.fc * self.radar.geometry.abs_v * R) / \
                        ((4 * self.radar.geometry.abs_v ** 2 - D ** 2 * self.c ** 2 / self.radar.fc ** 2) ** (3 / 4))

        return self.r_d_gain

    # GETTERS

    def get_true_range_axis(self):
        return self.true_fast_time_axis * self.c / 2

    def get_azimuth_axis(self):
        return self.data.get_azimuth_axis(self.radar.geometry.abs_v)

    def get_doppler_axis(self, doppler_centroid):
        """
        get the doppler axis given a doppler centroid enveloped around doppler centroid
        note class Data has a similar method, but that one is not enveloped to PRF, this particular
        implementation is used in the generation of azimuth filter
        :param doppler_centroid: doppler centroid of the image, zero for side looking radars
        :return: doppler axis, same length of the rows number
        """
        if self.data.rows_num % 2 == 0:
            # this is only valid if we have an even number of samples
            doppler_axis = cp.linspace(-self.radar.prf / 2,
                                       (1 - 1 / self.data.rows_num) * self.radar.prf / 2,
                                       self.data.rows_num)
        else:
            # this is for the case we have a odd number of samples (always)
            doppler_axis = cp.linspace(-self.radar.prf / 2,
                                       self.radar.prf / 2,
                                       self.data.rows_num)

        foffset = cp.ceil(doppler_centroid / self.radar.prf) * self.radar.prf
        # print("foffset in filter matrix", foffset)
        doppler_axis += foffset
        # wrapped around freq axis
        doppler_axis_out = cp.where(doppler_axis > doppler_centroid + self.radar.prf / 2)
        if len(doppler_axis_out) != 0:
            doppler_axis[doppler_axis_out] -= self.radar.prf
            doppler_axis_out = cp.where(doppler_axis < doppler_centroid - self.radar.prf / 2)
        if len(doppler_axis_out) != 0:
            doppler_axis[doppler_axis_out] += self.radar.prf

        return doppler_axis

    # ALGORITHM COMPONENTS

    def azimuth_fft(self, range_compressed_matrix):
        out = 1j * cp.zeros_like(range_compressed_matrix)
        for ii in range(len(range_compressed_matrix[0, :])):
            out[:, ii] = cp.fft.fftshift(cp.fft.fft(cp.fft.ifftshift(range_compressed_matrix[:, ii])))
        return out / self.radar.prf

    def azimuth_ifft(self, doppler_range_filtered_matrix):
        out = 1j * cp.zeros_like(doppler_range_filtered_matrix)
        for ii in range(len(doppler_range_filtered_matrix[0, :])):
            out[:, ii] = cp.fft.fftshift(cp.fft.ifft(cp.fft.ifftshift(doppler_range_filtered_matrix[:, ii])))
        return out * self.radar.prf

    def rcmc(self, doppler_range_compressed_matrix):
        """
        performs the rcmc step in the range-doppler domain
        :param doppler_range_compressed_matrix: matrix containing range compressed data in the azimuth frequency domain
        :return: rcmc'ed martrix
        """


        # # CUDA IMPLEMENTATION
        # Set the CUDA kernel configuration
        threads_per_block = (32, 32)
        blocks_per_grid_x = int(cp.ceil(doppler_range_compressed_matrix.shape[1] / threads_per_block[0]))
        blocks_per_grid_y = int(cp.ceil(doppler_range_compressed_matrix.shape[0] / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        doppler_axis = cp.linspace(-self.radar.prf / 2,
                                    (1 - 1 / len(doppler_range_compressed_matrix[:, 0])) * self.radar.prf / 2,
                                    len(doppler_range_compressed_matrix[:, 0]))
        foffset = cp.ceil(self.doppler_centroid / self.radar.prf) * self.radar.prf
        doppler_axis += foffset
        # wrapped around freq axis
        doppler_axis_out = cp.where(doppler_axis > self.doppler_centroid + self.radar.prf / 2)
        if len(doppler_axis_out) != 0:
            doppler_axis[doppler_axis_out] -= self.radar.prf
            doppler_axis_out = cp.where(doppler_axis < self.doppler_centroid - self.radar.prf / 2)
        if len(doppler_axis_out) != 0:
            doppler_axis[doppler_axis_out] += self.radar.prf
        matrix_rcmc = 1j * cp.zeros((self.data.rows_num, self.data.columns_num))

        rcmc_cuda[blocks_per_grid, threads_per_block](doppler_range_compressed_matrix,
                           matrix_rcmc,
                           self.get_true_range_axis(),
                           self.Fs,
                           self.radar.prf,
                           self.radar.geometry.abs_v,
                           self.c,
                           self.c / self.radar.fc,
                           self.radar.pulse.rate,
                           doppler_axis)

        cuda.synchronize()
        return matrix_rcmc
        # # NON CUDA IMPLEMENTATION
        
        # doppler_range_compressed_matrix_cpu = cp.asnumpy(doppler_range_compressed_matrix)
        # tra_cpu = cp.asnumpy(self.get_true_range_axis())
        # doppler_centroid_cpu = cp.asnumpy(self.doppler_centroid)

        # matrix_rcmc = 1j * np.zeros((self.data.rows_num, self.data.columns_num))
        # matrix_rcmc = rcmc(doppler_range_compressed_matrix_cpu,
        #                    matrix_rcmc,
        #                    tra_cpu,
        #                    self.Fs,
        #                    self.radar.prf,
        #                    self.radar.geometry.abs_v,
        #                    self.c,
        #                    self.c / self.radar.fc,
        #                    self.radar.pulse.rate,
        #                    doppler_centroid_cpu)
        # return cp.asarray(matrix_rcmc)

    def pattern_equalization(self, range_doppler_reconstructed_matrix):
        # estimate the antenna gain projected on ground
        self.r_d_gain = self.create_azimuth_equalization_matrix(self.data.get_range_axis(), self.doppler_axis)
        # divide the range_doppler_reconstructed_matrix by such normalized pattern and return
        return range_doppler_reconstructed_matrix * self.radar.antenna.broadside_gain / self.r_d_gain

    # AUTOMATED PROCESSING

    def azimuth_compression_test(self, range_compressed_matrix):
        """
        performs all the steps
        :param range_compressed_matrix: time domain matrix to be filtered in azimuth
        :return: reconstructed figure
        """
        # 1 azimuth fft
        doppler_range_compressed_matrix = self.azimuth_fft(range_compressed_matrix)
        # 2 rcmc
        doppler_range_compressed_matrix_rcmc = self.rcmc(doppler_range_compressed_matrix)
        # 3 azimuth filtering
        print("creating azimuth filter matrix")
        dax, self.azimuth_filter_matrix = self.create_azimuth_filter_matrix()
        doppler_range_image_matrix = doppler_range_compressed_matrix_rcmc * self.azimuth_filter_matrix
        # 4 ifft
        outimage = self.azimuth_ifft(doppler_range_image_matrix)
        # return reconstructed image
        return outimage

    def dump_cupy_array_as_numpy(self, file_path: str, cupy_array: cp.ndarray):
        if (self.is_dumping): 
            print('DUMPING DATA - ', file_path)
            with open(file_path, 'wb') as handle:
                pk.dump(cp.asnumpy(cupy_array), handle)
                handle.close()

    def azimuth_compression(self, doppler_bandwidth=0, patternequ=True):
        """
        performs all the steps and records intermediate results in the data class
        gets the input data from the data class
        :param doppler_bandwidth: optional parameter, specify the doppler bandwidth you want to use
        :param patternequ: optional, if true enables antenna pattern compensation in range-doppler domain
        :return: reconstructed figure
        """

        # memory tracing
        size, peak = tracemalloc.get_traced_memory()
        print("sie: ", size, ", peak: ", peak)

        # 1 azimuth fft
        print('1/4 performing azimuth fft')
        doppler_range_compressed_matrix = self.azimuth_fft(self.data.data_range_matrix)
        
        # RETRACE STEP 2
        self.dump_cupy_array_as_numpy('./retrace_data/azimuth_fft.pk', doppler_range_compressed_matrix)

        # dump raw data and free memory
        self.data.dump_rx_data()
        self.data.dump_range_compressed_matrix()
        # perform fft
        self.data.set_doppler_range_compressed_matrix(doppler_range_compressed_matrix)

        # memory tracing
        size, peak = tracemalloc.get_traced_memory()
        print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")

        # 2 rcmc
        print('2/4 performing range cell migration correction')
        doppler_range_compressed_matrix_rcmc = self.rcmc(doppler_range_compressed_matrix)

        # RETRACE STEP 3
        self.dump_cupy_array_as_numpy('./retrace_data/rcmc.pk', doppler_range_compressed_matrix_rcmc)

        self.data.set_doppler_range_compressed_matrix_rcmc(doppler_range_compressed_matrix_rcmc)
        # dump data and free memory
        self.data.dump_doppler_range_compressed_matrix()
        del doppler_range_compressed_matrix

        # memory tracing
        size, peak = tracemalloc.get_traced_memory()
        print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")

        # 3 azimuth filtering
        print('3/4 performing azimuth filtering')
        print(" -creating azimuth filter matrix")
        self.doppler_axis, self.azimuth_filter_matrix = self.create_azimuth_filter_matrix(linearFMapproximation=False)
        
        #output_asarray = cp.asnumpy(doppler_range_compressed_matrix_rcmc)
        doppler_range_image_matrix =  doppler_range_compressed_matrix_rcmc * self.azimuth_filter_matrix

        # 3.1 pattern equalization
        if patternequ:
            print("patternequ ")
            doppler_range_image_matrix = self.pattern_equalization(doppler_range_image_matrix)

        # 3.2 doppler windowing
        if doppler_bandwidth != 0:
            print("runnign doppler windowing ")

            # matrix line window
            doppler_window = cp.where(cp.abs(self.doppler_axis - self.doppler_centroid) <= doppler_bandwidth / 2, 1, 0)
            # apply the window
            doppler_range_image_matrix = doppler_range_image_matrix * doppler_window[:, cp.newaxis]
        
        # RETRACE STEP 4
        self.dump_cupy_array_as_numpy('./retrace_data/azimuth_filter4.pk', doppler_range_image_matrix)
        
        # dump and free memory
        self.data.dump_doppler_range_compressed_matrix_rcmc()
        del doppler_range_compressed_matrix_rcmc
        self.azimuth_filter_matrix = None
        self.data.set_range_doppler_reconstructed_image(doppler_range_image_matrix)

        # memory tracing
        size, peak = tracemalloc.get_traced_memory()
        print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")

        # 4 ifft
        print('4/4 performing azimuth ifft')
        outimage = self.azimuth_ifft(doppler_range_image_matrix)
        
        # RETRACE STEP 5
        self.dump_cupy_array_as_numpy('./retrace_data/azimuth_ifft.pk', outimage)
        
        # dump free and set memory
        self.data.set_reconstructed_image(outimage)
        self.data.dump_range_doppler_reconstructed_image()
        del doppler_range_image_matrix
        self.data.dump_reconstructed_image()
        # return reconstructed image
        print('Done')

        # memory tracing
        size, peak = tracemalloc.get_traced_memory()
        print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")

        return outimage


######################################## SELF TEST #####################################

def main():
    # memory profiling
    # memory tracing
    tracemalloc.start()
    print("initial memory occupation:")
    size, peak = tracemalloc.get_traced_memory()
    print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")

    # toggle to choose wether to output or not an image at the end of this test
    IMAGEOUT = False

    print('unpickling')
    with open('./Simulation_Data/channel_dump.pk', 'rb') as handle:
        channel = pk.load(handle)
        handle.close()
    with open('./Simulation_Data/data_dump.pk', 'rb') as handle:
        data = pk.load(handle)
        if data.data_matrix.any() == None:
            data.load_rx_data()
            data.load_range_compressed_matrix()
        handle.close()

    # create a range Doppler instance
    rangedop = RangeDopplerCompressor(channel, data)

    outimage = rangedop.azimuth_compression()

    # %%
    az_ax = rangedop.get_azimuth_axis()
    r_ax = rangedop.get_true_range_axis()

    # %% picke back data
    print("pickling")
    with open('./Simulation_Data/channel_dump.pk', 'wb') as handle:
        pk.dump(channel, handle)
        handle.close()
    with open('./Simulation_Data/data_dump.pk', 'wb') as handle:
        pk.dump(data, handle)
        handle.close()

    # %%
    if (IMAGEOUT):
        fig, ax = plt.subplots(1)
        c = ax.pcolormesh(r_ax, az_ax, (np.abs(outimage)),
                          shading='auto', cmap=plt.get_cmap('jet'))

        fig.colorbar(c)
        # ax.set_xlim((401e3, 412e3))
        ax.set_xlabel("range [m]")
        ax.set_ylabel("azimuth [m]")
        plt.show()

    # memory tracing
    print("final memory occupation:")
    size, peak = tracemalloc.get_traced_memory()
    print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")
    tracemalloc.stop()
    return str(("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB"))


if __name__ == '__main__':
    main()
    input("press return to close")
