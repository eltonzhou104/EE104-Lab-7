import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

############################################################
# Generate the signal
############################################################

# Seed the random number generator
np.random.seed(1234)

time_step = 0.05
period = 3

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(8 * np.pi / period * time_vec) + 0.6 * np.random.randn(time_vec.size)) + (np.sin(105 * np.pi / period * time_vec) + 0.3 * np.random.randn(time_vec.size)) +(np.sin(550 * np.pi / period * time_vec)+ 0.2 * np.random.randn(time_vec.size))


plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
sig_fft = fftpack.fft(sig)

############################################################
# Compute and plot the power
############################################################


# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)**2

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('power')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:9], power[:9])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection

############################################################
# Remove all the high frequencies
############################################################
#
# We now remove all the high frequencies and transform back from
# frequencies to signal.

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time(seconds)')
plt.ylabel('Amplitude')

plt.legend(loc='best')

filtered_sig_fft = fftpack.fft(filtered_sig)
filtered_power = np.abs(filtered_sig_fft)**2
filtered_sig = fftpack.ifft(high_freq_fft)


sig_fft1 = fftpack.fft(filtered_sig)
power1 = np.abs(sig_fft1)**2

sample_freq1 = fftpack.fftfreq(filtered_sig.size, d=time_step)

plt.figure(figsize=(6, 5))
plt.plot(sample_freq1, power1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('power')
############################################################
#
# **Note** This is actually a bad way of creating a filter: such brutal
# cut-off in frequency space does not control distorsion on the signal.

#This is to do the once twice filtfilt lowpass filtered signals
from scipy import signal
t=time_vec
#3rd order lowpass butterworth
b, a = signal.butter(5, 0.1)

#Apply filter to signal
zi = signal.lfilter_zi(b, a)
z,_=signal.lfilter(b, a, sig, zi=zi*sig[0])

b, a = signal.butter(5, 0.1)
#Apply to filter again
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

z3, _ = signal.lfilter(b, a, z, zi=zi*z[0])

z4, _ = signal.lfilter(b, a, z, zi=zi*z[0])
y=signal.filtfilt(b, a, sig)

# Display noisy signal, lowpass filter once twice and filtfilt signal
plt.figure
plt.plot(t, sig, 'purple', alpha=0.75)
plt.plot(t, z, 'cyan', t, z2, 'r', t, y, 'k')
plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice', 'third filter'), loc='best')
plt.grid(True)
plt.show()
