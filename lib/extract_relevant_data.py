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

date = '2023-08-24'
path = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + date + '/'
autolog = path + '/' + date + '-' + 'auto-log.txt'

##First, calibrate the energy!
##

test = open_images(18, 47, path, date, image_dim=[512, 512], roix=[0, 512])

image = np.squeeze(np.apply_over_axes(np.sum, test, [2]))

##

transformed_image = shear_image(image, -2.5)
profile = np.apply_over_axes(np.sum, transformed_image, [0])
profile = np.ravel(profile) / np.max(profile) * 1000

roix = [0, 512]

add_ons = 0

# profile[400:] = 0

data, peaks = fit_energy_calibration_peaks(profile, 70, roix, 11)

condition = (peaks > 0) & (peaks < 512)

peaks = peaks[condition]

peaksn = np.ravel(np.zeros([np.size(peaks) + add_ons, 1]))

if add_ons == 0:
    print("nothing happens")
else:
    # peaksn[-1] = int(1340)
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

first_harmonic = 17
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
start_im = 18
end_im = 317

all_images = open_images(start_im, end_im, path, date, roix=[0, 512], image_dim=[512, 512], show_status=1,
                         transform=1)  # c = 2.2, p = 2.5 bar
## background_subtracted images
bg = open_images(328, 337, path, date, roix=[0, 512], image_dim=[512, 512], show_status=1,
                         transform=1)  # c = 2.2, p = 2.5 bar
images_bg_treated = subtract_background(all_images,bg)
##
plt.figure(222)
plt.subplot(2,1,1)
plt.pcolormesh(np.log(all_images[:,:,100]))
plt.colorbar()
plt.subplot(2,1,2)
plt.pcolormesh(np.log(images_bg_treated[:,:,100]))
plt.colorbar()
##
#all_images_new = np.zeros_like(all_images)
#all_images_new[80:320, 200:450, :] = all_images[80:320, 200:450, :]
## Now get all the profiles
treated_profiles = np.zeros([all_images.shape[1], all_images.shape[2]])
for i in range(0, np.shape(treated_profiles)[1]):
    print(i)
    im = all_images[:, :, i]
    treated_profiles[:, i] = np.squeeze(treat_image_profiles(im, E_axis))
##
#plt.figure(3)
#plt.subplot(2, 1, 1)
#plt.pcolormesh(all_images_new[:, :, 505])
#plt.colorbar()
#plt.subplot(2, 1, 2)
#plt.plot(np.sum(all_images_new[:, :, 505], 0))
## test images
#t = all_images_new[:, :, -30:]
#all_stds = np.squeeze(np.apply_over_axes(np.std, t, 2))

#plt.figure(4)
#plt.pcolormesh(all_stds)
# plt.clim(10,np.max(all_stds))
#plt.colorbar()

##
#plt.figure(5)
#plt.plot(t[220, 364, :])


##
# 80:320,200:450
#def cosine_func(x, amplitude, frequency, phi, offset):
#    return amplitude * np.cos(2 * np.pi * frequency * x + phi) + offset


#for i in range(200, 450):
#    for k in range(80, 320):
#        x = np.squeeze(t[k, i])

##
plt.figure(1234)
test2 = treated_profiles[:, 120]
test3 = treated_profiles[:, 220]
test4 = treated_profiles[:, 20]

plt.clf()
plt.plot(test2)
plt.plot(test3)
plt.plot(test4)
## background
# bgdate = '2023-06-15'
# bgpath = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + bgdate + '/'
# background = open_images(1978, 2037, bgpath, bgdate, roix=[0, 1600], show_status=1)
# bg = np.apply_over_axes(np.average, background, [2])
##
# treated_images = np.zeros_like(all_images)
# for i in range(0, np.shape(treated_images)[2]):
#    print(i)
#    im = all_images[:, :, i] - bg[:, :, 0]
#    treated_images[:, :, i] = np.squeeze(treat_image(im, E_axis))

##
autolog_content = open_autologfile(autolog)
relevant_content = autolog_content[start_im:end_im + 1, :7]
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
import os

path_save = 'D:/dlab/2_color_harmonics/' + date
if os.path.exists(path_save):
    print("Path exists!")
    data_filename = 'D:/dlab/2_color_harmonics/' + date + '/test.hdf5'
else:
    os.makedirs(path_save)
    data_filename = 'D:/dlab/2_color_harmonics/' + date + '/test.hdf5'

hf = h5py.File(data_filename, 'w')
##
hf.create_dataset('raw_images', data=all_images)
# hf.create_dataset('treated_images', data=treated_images)
# hf.create_dataset('background', data=bg)
hf.create_dataset('E', data=E)
hf.create_dataset('phase', data=phase)
hf.create_dataset('ratio', data=ratio)
hf.create_dataset('phase_error', data=phase_error)
hf.create_dataset('phase_dev', data=phase_dev)
hf.create_dataset('y', data=y)
hf.create_dataset('treated_profiles', data=treated_profiles)
##
hf.close()
