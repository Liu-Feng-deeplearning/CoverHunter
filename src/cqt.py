#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/7/7 3:22 PM
# software: PyCharm

import logging
import os
import time

import librosa
import numpy as np
import scipy


def shorter(feat, mean_size):
  if mean_size == 1:
    return feat
  cqt = feat.T
  height, length = cqt.shape
  new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
  for i in range(int(length / mean_size)):
    new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
  return new_cqt.T


def compute_cqt_with_librosa(signal=None, sr=16000, filename=None):
  if filename is not None:
    signal, _ = librosa.load(path=filename, sr=sr)
  cqt = np.abs(librosa.cqt(signal, sr=sr))
  cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
  return cqt_db


class PyCqt:
  """just wrapper for cqt extractor,

  References:https://github.com/zafarrafii/Zaf-Python/blob/master/zaf.py
  """

  def __init__(self, sample_rate, hop_size, octave_resolution=12, min_freq=32,
               max_freq=None):
    self._hop_size = hop_size
    self._sample_rate = sample_rate
    if not max_freq:
      max_freq = sample_rate // 2
    self._kernel = self._compute_cqt_kernel(sample_rate, octave_resolution,
                                            min_freq, max_freq)
    logging.info("cqt kernal: {}".format(np.shape(self._kernel)))
    return

  @staticmethod
  def _compute_cqt_kernel(
      sampling_frequency, octave_resolution, minimum_frequency,
      maximum_frequency
  ):
    """
    Compute the constant-Q transform (CQT) kernel.
    Inputs:
        sampling_frequency: sampling frequency in Hz
        octave_resolution: number of frequency channels per octave
        minimum_frequency: minimum frequency in Hz
        maximum_frequency: maximum frequency in Hz
    Output:
        cqt_kernel: CQT kernel (sparse) (number_frequencies, fft_length)

    """

    # Compute the constant ratio of frequency to resolution (= fk/(fk+1-fk))
    quality_factor = 1 / (pow(2, 1 / octave_resolution) - 1)

    # Compute the number of frequency channels for the CQT
    number_frequencies = round(
      octave_resolution * np.log2(maximum_frequency / minimum_frequency)
    )

    # Compute the window length for the FFT
    # (= longest window for the minimum frequency)
    fft_length = int(
      pow(
        2, np.ceil(
          np.log2(quality_factor * sampling_frequency / minimum_frequency))
      )
    )

    # Initialize the (complex) CQT kernel
    cqt_kernel = np.zeros((number_frequencies, fft_length), dtype=complex)

    # Loop over the frequency channels
    for i in range(number_frequencies):
      # Derive the frequency value in Hz
      frequency_value = minimum_frequency * pow(2, i / octave_resolution)

      # Compute the window length in samples
      # (nearest odd value to center the temporal kernel on 0)
      window_length = (
          2 * round(
        quality_factor * sampling_frequency / frequency_value / 2) + 1
      )

      # Compute the temporal kernel for the current frequency(odd and symmetric)
      temporal_kernel = (np.hamming(window_length)
                         * np.exp(2 * np.pi * 1j * quality_factor
                                  * np.arange(-(window_length - 1) / 2,
                                              (
                                                  window_length - 1) / 2 + 1) / window_length
                                  ) / window_length)

      # Derive the pad width to center the temporal kernels
      pad_width = int((fft_length - window_length + 1) / 2)

      # Save the current temporal kernel at the center
      # (the zero-padded temporal kernels are not perfectly symmetric
      # anymore because of the even length here)
      cqt_kernel[i, pad_width: pad_width + window_length] = temporal_kernel

    # Derive the spectral kernels by taking the FFT of the temporal kernels
    # (the spectral kernels are almost real because the temporal kernels are almost symmetric)
    cqt_kernel = np.fft.fft(cqt_kernel, axis=1)

    # Make the CQT kernel sparser by zeroing magnitudes below a threshold
    cqt_kernel[np.absolute(cqt_kernel) < 0.01] = 0

    # Make the CQT kernel sparse by saving it as a compressed sparse row matrix
    cqt_kernel = scipy.sparse.csr_matrix(cqt_kernel)

    # Get the final CQT kernel by using Parseval's theorem
    cqt_kernel = np.conjugate(cqt_kernel) / fft_length
    return cqt_kernel

  @staticmethod
  def _compute_cqt_spec(audio_signal, sampling_frequency, time_resolution,
                        cqt_kernel):
    """
    Compute the constant-Q transform (CQT) spectrogram using a CQT kernel.
    Inputs:
        audio_signal: audio signal (number_samples,)
        sampling_frequency: sampling frequency in Hz
        time_resolution: number of time frames per second
        cqt_kernel: CQT kernel (number_frequencies, fft_length)
    Output:
        cqt_spectrogram: CQT spectrogram (number_frequencies, number_times)
    """

    # Derive the number of time samples per time frame
    step_length = round(sampling_frequency / time_resolution)

    # Compute the number of time frames
    number_times = int(np.floor(len(audio_signal) / step_length))

    # Get th number of frequency channels and the FFT length
    number_frequencies, fft_length = np.shape(cqt_kernel)

    # Zero-pad the signal to center the CQT
    audio_signal = np.pad(
      audio_signal,
      (
        int(np.ceil((fft_length - step_length) / 2)),
        int(np.floor((fft_length - step_length) / 2)),
      ),
      "constant",
      constant_values=(0, 0),
    )

    cqt_spectrogram = np.zeros((number_frequencies, number_times))
    i = 0
    for j in range(number_times):
      # Compute the magnitude CQT using the kernel
      cqt_spectrogram[:, j] = np.absolute(
        cqt_kernel * np.fft.fft(audio_signal[i: i + fft_length])
      )
      i = i + step_length
    return cqt_spectrogram

  def compute_cqt(self, signal_float=None, feat_dim_first=True):
    y = signal_float
    time_resolution = int(1 / self._hop_size)
    cqt_spectrogram = self._compute_cqt_spec(y, self._sample_rate,
                                             time_resolution, self._kernel)
    cqt_spectrogram = cqt_spectrogram + 1e-9
    ref_value = np.max(cqt_spectrogram)
    cqt_spectrogram = 20 * np.log10(cqt_spectrogram) - 20 * np.log10(ref_value)
    if not feat_dim_first:
      cqt_spectrogram = cqt_spectrogram.T
    return cqt_spectrogram

