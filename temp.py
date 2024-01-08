from scipy.signal import butter, lfilter, iirnotch, filtfilt
from scipy.signal import find_peaks

import numpy as np
import wfdb
import matplotlib.pyplot as plt
import os

#C:/Users/KULLANICI/.spyder-py3/01000/01000_hr.hea
# Directory containing your .hea files
signal_file = 'C:\\USERS\\KULLANICI\\.spyder-py3\\01000\\01000_hr'  # Update this to your actual directory path


# Use wfdb.rdrecord() to read the signal file
record = wfdb.rdrecord(signal_file)

# Access the signal data and metadata
signal_data = record.p_signal  # NumPy array containing signal data
signal_metadata = record.__dict__  # Dictionary containing metadata

# You can access various metadata properties like sampling frequency, units, etc.
fs = record.fs  # Sampling frequency

# Visualize the signal
plt.figure(figsize=(12, 4))
plt.plot(signal_data)
plt.title('ECG')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def notch_filter(freq, Q, fs):
    nyq = 0.5 * fs
    normal_cutoff = freq / nyq
    b, a = iirnotch(normal_cutoff, Q)
    return b, a

ecg_signal = record.p_signal

# Preprocess each lead
preprocessed_ecg = np.zeros_like(ecg_signal)
for i in range(ecg_signal.shape[1]):
    # Remove baseline wander with high-pass filter
    b, a = butter_highpass(0.5, fs)
    ecg_lead = filtfilt(b, a, ecg_signal[:, i])
    
    # Remove power line interference with notch filter
    b, a = notch_filter(50, 30, fs)  # Set frequency to 60 for regions with 60 Hz power line frequency
    ecg_lead = filtfilt(b, a, ecg_lead)
    
    # Remove high-frequency noise with low-pass filter
    b, a = butter_lowpass(40, fs)
    ecg_lead = filtfilt(b, a, ecg_lead)
    
    # Normalize the signal to have zero mean and unit variance
    ecg_lead = (ecg_lead - np.mean(ecg_lead)) / np.std(ecg_lead)
    
    preprocessed_ecg[:, i] = ecg_lead

# Plot the preprocessed signal
plt.figure(figsize=(12, 4))
for i in range(preprocessed_ecg.shape[1]):
    plt.plot(preprocessed_ecg[:, i] + 10*i)  # Offset each lead for clarity
plt.title('Preprocessed ECG')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


ecg_lead = ecg_signal[:, 0]

# Detect R-peaks
# The height parameter should be tuned according to the amplitude of the QRS complex
# The distance parameter should be set according to the expected minimum distance between peaks (based on heart rate)
r_peaks, _ = find_peaks(ecg_lead, height=np.mean(ecg_lead), distance=150)  # distance in samples

# Plotting the results
plt.figure(figsize=(15, 5))
plt.plot(ecg_lead, label='ECG Lead')
plt.plot(r_peaks, ecg_lead[r_peaks], 'x', label='R-peaks')
plt.title('R-peak Detection')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


heartbeat_segments = []

# Define the window to look around the R-peak
# This will depend on the sampling rate and desired window size
# For example, to include 200 ms before and 400 ms after the R-peak
window_before = int(0.2 * fs)
window_after = int(0.4 * fs)

for i in range(len(r_peaks)):
    # Start of the segment (make sure it's not negative)
    start = max(r_peaks[i] - window_before, 0)
    # End of the segment (make sure it doesn't go past the signal length)
    end = min(r_peaks[i] + window_after, len(ecg_signal))
    
    # Extract the segment
    segment = ecg_signal[start:end, :]
    heartbeat_segments.append(segment)

# Plot an example segment
example_segment = heartbeat_segments[2]  # Change the index to view different segments
plt.plot(example_segment)
plt.title('Individual Heartbeat Segment')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()



# Assume 'ecg_signal' is a 1D NumPy array containing the preprocessed ECG signal for a single lead
# Assume 'r_peaks' is a list or array containing the indices of detected R peaks

# Initialize arrays to hold the indices of the S wave ends and T wave starts
s_indices = np.zeros_like(r_peaks, dtype=int)
t_indices = np.zeros_like(r_peaks, dtype=int)

# Define a search window (in samples) for the S wave end and T wave start
s_window = int(0.05 * fs)  # For example, 50ms after R peak
t_window = int(0.2 * fs)   # For example, 200ms after R peak, but this can vary

for i, r_peak in enumerate(r_peaks):
    # Search for S wave end by finding the minimum after the R peak
    s_search_end = min(r_peak + s_window, len(ecg_signal))
    s_wave_end_index = np.argmin(ecg_signal[r_peak:s_search_end]) + r_peak
    s_indices[i] = s_wave_end_index
    
    # Search for T wave start by finding the point where the signal starts to rise significantly after the S wave
    t_search_end = min(s_wave_end_index + t_window, len(ecg_signal))
    # This is a placeholder for the logic to find the start of the T wave
    # You could look for a change in slope or a threshold crossing, for example
    t_wave_start_index = s_wave_end_index + np.argmax(ecg_signal[s_wave_end_index:t_search_end] - np.mean(ecg_signal[s_wave_end_index:t_search_end])) 
    t_indices[i] = t_wave_start_index

# Now you have s_indices and t_indices which you can use for ST-segment analysis



baseline_level = -0.1  # Assume we've measured the baseline at -0.1 mV

# Parameters
