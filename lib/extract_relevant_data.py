from lib.helpers import *
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import sys
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import lib.helpers as help
import cv2
import h5py

plt.set_cmap('plasma')

date = '2023-06-16'
path = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + date + '/'
autolog = path + '/' + date + '-' + 'auto-log.txt'

##First, calibrate the energy!
##

test = open_images(2579, 2608, path, date, roix=[0, 1600])

image = np.squeeze(np.apply_over_axes(np.sum, test, [2]))

##

transformed_image = shear_image(image, -2.5)
profile = np.apply_over_axes(np.sum, transformed_image, [0])
profile = np.ravel(profile) / np.max(profile) * 1000

roix = [520, 1600]

add_ons = 0

profile[1350:] = 0

data, peaks = fit_energy_calibration_peaks(profile, 60, roix, 41)
peaksn = np.ravel(np.zeros([np.size(peaks) + add_ons, 1]))

if add_ons == 0:
    print("nothing happens")
else:
    #peaksn[-1] = int(1340)
    peaksn[-1] = int(1207)
    peaksn[:-add_ons] = peaks
    peaks = peaksn.astype(int)

plt.figure(1)
plt.clf()
plt.pcolormesh(np.log10(transformed_image))
plt.plot(profile)
plt.plot(peaks, profile[peaks], 'o')

##
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam

first_harmonic = 15
E = np.ones_like(peaks) * first_harmonic * Eq / qe + np.arange(0, np.size(peaks)) * 2 * Eq / qe
peak_mod = np.shape(transformed_image)[1] - np.flip(peaks)

p = np.polyfit(peak_mod, E, 3)

x_axis = np.arange(0, np.shape(transformed_image)[1])
scale_x_axis = np.polyval(p, x_axis)
E_axis = scale_x_axis
y_axis = np.arange(0, np.shape(transformed_image)[0])
y = y_axis

E, new_image = treat_image(image, E_axis, return_correct_E_axis=1)

plt.figure(9)
plt.pcolormesh(E, y, np.log10(new_image), cmap='plasma')
plt.xlim([15, 42])
plt.xlabel("Energy (eV")
x = np.arange(15, 35) * Eq / qe
[plt.axvline(_x, linewidth=1, color='g') for _x in x]
plt.title("This should look okay")

## Now load all pictures!
start_im = 2579
end_im = 3778

all_images = open_images(start_im, end_im, path, date, roix=[0, 1600], show_status=1)  # c = 2.2, p = 2.5 bar
## Now get all the profiles
treated_profiles = np.zeros([all_images.shape[1], all_images.shape[2]])
for i in range(0, np.shape(treated_profiles)[1]):
    print(i)
    im = all_images[:, :, i]
    treated_profiles[:, i] = np.squeeze(treat_image_profiles(im, E_axis))
##
plt.figure(1234)
test2 = treated_profiles[:,900]
plt.clf()
plt.plot(test2)

## background
#bgdate = '2023-06-15'
#bgpath = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + bgdate + '/'
#background = open_images(1978, 2037, bgpath, bgdate, roix=[0, 1600], show_status=1)
#bg = np.apply_over_axes(np.average, background, [2])
##
#treated_images = np.zeros_like(all_images)
#for i in range(0, np.shape(treated_images)[2]):
#    print(i)
#    im = all_images[:, :, i] - bg[:, :, 0]
#    treated_images[:, :, i] = np.squeeze(treat_image(im, E_axis))

##
autolog_content = open_autologfile(autolog)
relevant_content = autolog_content[start_im:end_im + 1, :9]
relevant_content = relevant_content.astype(float)

red_power = relevant_content[:, 1]
green_power = relevant_content[:, 2] * 1e-3

all_ratios = 4 * green_power / (4 * green_power + red_power)
ratio = np.unique(all_ratios)

phase = np.unique(relevant_content[:, 3])

##
phase_error = relevant_content[:, 5]
phase_error = np.reshape(phase_error, (ratio.size, phase.size))
phase_set = relevant_content[:, 3]
phase_set = np.reshape(phase_set, (ratio.size, phase.size))
phase_avg = relevant_content[:, 4]
phase_avg = np.reshape(phase_avg, (ratio.size, phase.size))
phase_dev = phase_set - phase_avg

##
data_filename = 'D:/dlab/2_color_harmonics/2023-06-16/c3_0p3.hdf5'

hf = h5py.File(data_filename, 'w')
##
hf.create_dataset('raw_images', data=all_images)
#hf.create_dataset('treated_images', data=treated_images)
#hf.create_dataset('background', data=bg)
hf.create_dataset('E', data=E)
hf.create_dataset('phase', data=phase)
hf.create_dataset('ratio', data=ratio)
hf.create_dataset('phase_error', data=phase_error)
hf.create_dataset('phase_dev', data=phase_dev)
hf.create_dataset('y', data = y)
hf.create_dataset('treated_profiles', data = treated_profiles)
##
hf.close()
