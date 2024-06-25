import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Path to the directory you want to scan
folder_path = '/home/sm32289/Cat_Pose/AutoEncoder/Model2/Data/2/Normal'

# List to hold the names of folders
folder_normal = []
folder_abnormal = []

# Iterate over the entries in the specified directory
for entry in os.scandir(folder_path):
    if entry.is_dir():
        folder_normal.append(entry.name)
        
folder_path = '/home/sm32289/Cat_Pose/AutoEncoder/Model2/Data/2/Abnormal'

# Iterate over the entries in the specified directory
for entry in os.scandir(folder_path):
    if entry.is_dir():
        folder_abnormal.append(entry.name)

# print(folder_abnormal)


try:
    path = '/home/sm32289/Cat_Pose/AutoEncoder/Model2/Data/2/Normal/' + folder_normal[0] + '/' + 'dist.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file)
    # Use the data
    # print(data)
except FileNotFoundError:
    print("The file was not found.")
except pickle.UnpicklingError:
    print("There was an error unpickling the file.")

# number of dimensions
# num_dimensions = np.array(data).ndim
# print("Number of dimensions:", num_dimensions)

# print(data[:,:,4,13])

fft_data = data[:,:,4,13]
# print(fft_data)


# Assuming 'data' is your original multi-dimensional array
# Replace 'data' with your actual data array




# Concatenate the slices to create a continuous time series
continuous_time_series = data[:, :, 4, 13].flatten()

# Perform Fourier Transform on the continuous time series
fft_result = np.fft.fft(continuous_time_series)

# Frequency axis for plotting (considering sampling rate as 1)
freq = np.fft.fftfreq(len(continuous_time_series), d=1)

# Plotting the Fourier Transform result
plt.figure(figsize=(12, 6))
plt.plot(freq, np.abs(fft_result))
plt.title('Fourier Transform of the Continuous Time Series Data')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('/home/sm32289/Cat_Pose/AutoEncoder/Model2/preprocess/normal/' + folder_normal[0] +'.png', dpi=300, bbox_inches='tight')

plt.show()

# for i in range(0, len(folder_abnormal)):
#     try:
#         path = '/home/sm32289/Cat_Pose/AutoEncoder/Model2/Data/2/Abnormal/' + folder_abnormal[i] + '/' + 'dist.pkl'
#         with open(path, 'rb') as file:
#             data = pickle.load(file)
#         # Use the data
#         # print(data)
#     except FileNotFoundError:
#         print("The file was not found.")
#     except pickle.UnpicklingError:
#         print("There was an error unpickling the file.")
#     # fft_data = data[:,:,4,13]
#     # Concatenate the slices to create a continuous time series
#     continuous_time_series = data[:, :, 4, 13].flatten()

#     # Perform Fourier Transform on the continuous time series
#     fft_result = np.fft.fft(continuous_time_series)

#     # Frequency axis for plotting (considering sampling rate as 1)
#     freq = np.fft.fftfreq(len(continuous_time_series), d=1)

#     # Plotting the Fourier Transform result
#     plt.figure(figsize=(12, 6))
#     plt.plot(freq, np.abs(fft_result))
#     plt.title('Fourier Transform of the Continuous Time Series Data')
#     plt.xlabel('Frequency')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.savefig('/home/sm32289/Cat_Pose/AutoEncoder/Model2/preprocess/abnormal/' + folder_abnormal[i] +'.png', dpi=300, bbox_inches='tight')
#     print(str(i) + ':' +folder_abnormal[i] + ' visualized')
#     plt.close()


# try:
#     path = '/home/sm32289/Cat_Pose/AutoEncoder/Model2/Data/2/Abnormal/' + folder_abnormal[52] + '/' + 'dist.pkl'
#     with open(path, 'rb') as file:
#         data = pickle.load(file)
#     # Use the data
#     print(data)
# except FileNotFoundError:
#     print("The file was not found.")
# except pickle.UnpicklingError:
#     print("There was an error unpickling the file.")
# fft_data = data[:,:,4,13]
# Concatenate the slices to create a continuous time series
# continuous_time_series = data[:, :, 4, 13].flatten()

# # Perform Fourier Transform on the continuous time series
# fft_result = np.fft.fft(continuous_time_series)

# # Frequency axis for plotting (considering sampling rate as 1)
# freq = np.fft.fftfreq(len(continuous_time_series), d=1)

# # Plotting the Fourier Transform result
# plt.figure(figsize=(12, 6))
# plt.plot(freq, np.abs(fft_result))
# plt.title('Fourier Transform of the Continuous Time Series Data')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.savefig('/home/sm32289/Cat_Pose/AutoEncoder/Model2/preprocess/abnormal/' + folder_abnormal[52] +'.png', dpi=300, bbox_inches='tight')
print(str(52) + ':' +folder_abnormal[52] + ' visualized')
# plt.close()
