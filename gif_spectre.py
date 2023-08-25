##
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import sys
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import lib.helpers as help
import cv2
import os

date = '2023-06-16'
path = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + date + '/'
autolog = path + '/' + date + '-' + 'auto-log.txt'
image_dim = [1000, 1600]


def open_images(start, end, path, date, roix, show_status=0):
    image_matrix = np.zeros([image_dim[0], np.size(np.arange(roix[0], roix[1])), end - start + 1])
    for ind in range(start, end + 1):
        file = path + date + '-' + str(int(ind)) + '.bmp'
        im_temp = np.asarray(Image.open(file))
        im_temp = im_temp[:, roix[0]:roix[1]]
        image_matrix[:, :, ind - start] = im_temp
    return image_matrix


def redistribute_image(sheared_image, E_axis):
    Jacobian_vect = h * c / (E_axis ** 2)
    Jacobian_vect_norm = Jacobian_vect / np.max(Jacobian_vect)
    Jacobian_mat = np.tile(Jacobian_vect_norm, [np.shape(sheared_image)[0], 1])
    redistributed_image = np.multiply(sheared_image, Jacobian_mat)
    return redistributed_image


def shear_image(image_old, val):
    T = np.float32([[1, val / 100, 0], [0, 1, 0]])
    size_T = (image.shape[1], image_old.shape[0])
    image_new = cv2.warpAffine(image_old, T, size_T)
    return image_new


def treat_image(image_old, energy_axis, return_correct_E_axis=0):
    sheared_image = shear_image(image_old, -2.5)
    redistributed_image = redistribute_image(sheared_image, energy_axis)
    y_axis = np.arange(0, np.shape(sheared_image)[0])
    correct_E_axis = np.arange(energy_axis[0], energy_axis[-1],
                               abs((energy_axis[0] - energy_axis[-1]) / np.shape(redistributed_image)[1]))
    interp_func = interp1d(energy_axis, np.flip(redistributed_image, 1), axis=1, kind='linear')
    image_new = interp_func(correct_E_axis)
    if return_correct_E_axis:
        return correct_E_axis, image_new
    else:
        return image_new


def fit_energy_calibration_peaks(prof, prom=2000, roi=[640, 1600], smoothing=21):
    dat = prof
    dat[0: roi[0]] = 0
    calibration = help.savitzky_golay(prof, smoothing, 3)  # window size 51, polynomial order 3
    peak, _ = find_peaks(calibration, prominence=prom)
    return calibration, peak


# Energy calibration
test = open_images(2579, 2608, path, date, roix=[0, 1600])
image = np.apply_over_axes(np.sum, test, [2])
transformed_image = shear_image(image, -2.5)
profile = np.apply_over_axes(np.sum, transformed_image, [0])
profile = np.ravel(profile)
roix = [620, 1600]
add_ons = 2
data, peaks = fit_energy_calibration_peaks(profile, 5000, roix, 21)
peaksn = np.ravel(np.zeros([np.size(peaks) + add_ons, 1]))
peaksn[:-add_ons] = peaks
peaksn[-1] = int(1340)
peaksn[-2] = int(1207)
peaks = peaksn.astype(int)
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
pixel_size = 5e-6
distance_cam_mcp = 0.3
angle_calibration = pixel_size / distance_cam_mcp
y_angle = y * angle_calibration * 1e3
y_angle -= np.max(y_angle) / 2

# Traitement
start_im = 3179
end_im = start_im + 30

all_images = open_images(start_im, end_im, path, date, roix=[0, 1600], show_status=1)

bgdate = '2023-06-15'
bgpath = 'G:/Atto/Data/LASC/dlab/dlabharmonics/' + bgdate + '/'
background = open_images(1978, 2037, bgpath, bgdate, roix=[0, 1600], show_status=1)

bg = np.apply_over_axes(np.average, background, [2])

treated_images = np.zeros_like(all_images)
for i in range(0, np.shape(treated_images)[2]):
    im = all_images[:, :, i] - bg[:, :, 0]
    treated_image = treat_image(im, E_axis)[0:1000, 0:1600]
    treated_images[:, :, i] = treated_image

treated_images = treated_images[:, :801, :]

index = range(0, np.shape(treated_images)[2])
harmonics = np.sum(treated_images[:, :, index], 0)
oui = np.linspace(-np.pi, np.pi, len(index))
harmonic_map = np.array(harmonics).T

harmonic_30 = harmonic_map
x_min = 294
x_max = 306
y_min = 0
y_max = harmonic_30.shape[0]

harmonic_30 = harmonic_30[y_min:y_max, x_min:x_max]
integration = np.sum(harmonic_30, axis=1)
integration /= np.max(integration)

phases = range(0,30)
images = []  # List to store the plot images
for phase in phases:
    harmonic_data = np.sum(treated_images[:, :, phase], axis=0)
    plt.figure(phase)
    plt.plot(harmonic_data, label=f'Harmonic for phase {phase}')
    plt.xlim((0,600))
    plt.ylim((0, 2000))
    plt.legend(loc='upper right')
    plt.savefig(f'./harmonics/harmonic_plot_{phase}.png')  # Save the plot as an image
    plt.close()  # Close the figure to free memory
    images.append(Image.open(f'./harmonics/harmonic_plot_{phase}.png'))  # Add the image to the list
# Save the list of images as a GIF
images[0].save('harmonic_plots.gif',
               save_all=True,
               append_images=images[1:],
               duration=500,
               loop=1)

print("GIF created successfully.")


