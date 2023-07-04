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

data_file_name = 'D:/dlab/2_color_harmonics/2023-06-27/c3p3.hdf5'

hfr = h5py.File(data_file_name, 'r')
print(hfr.keys())

##
raw = np.asarray(hfr.get('raw_images'))
#E = np.asarray(hfr.get('E'))
#phase = np.asarray(hfr.get('phase'))
#ratio = np.asarray(hfr.get('ratio'))
#y = np.asarray(hfr.get('y'))
#phase_dev = np.asarray(hfr.get('phase_dev'))
#phase_error = np.asarray(hfr.get('phase_error'))

h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam

##
std_total = np.apply_over_axes(np.std, raw[:,:,:],[2])
##
plt.figure(123)
plt.clf()
plt.pcolormesh(np.log10(np.squeeze(std_total)))
plt.colorbar()

##
profiles = np.squeeze(np.apply_over_axes(np.sum,raw,[0]))
##
#


from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

treated_profiles = np.zeros_like(profiles)

for i in np.arange(0,profiles.shape[1]):
    p = profiles[:,i]
    f = gaussian_filter1d(p, 3)
    peaks, _ = find_peaks(-f)
    y_bg = np.interp(np.arange(0, 1600), peaks, f[peaks])
    p_new = p-y_bg
    treated_profiles[:,i] = p_new

##


test = profiles[:,666]
test = raw[450,:,0]

plt.figure(123)
plt.clf()
plt.subplot(4,1,1)
plt.plot(-test,'k')

testf = gaussian_filter1d(test,3)

plt.plot(-testf,'r')

peaks,_ = find_peaks(-testf)
plt.scatter(peaks,-testf[peaks],color='b')

plt.subplot(4,1,2)

y_bg = np.interp(np.arange(0,1600),peaks, testf[peaks])
plt.plot(testf)
plt.scatter(peaks,testf[peaks],color='r')
plt.plot(y_bg)

plt.subplot(4,1,3)
plt.plot(testf-y_bg)

plt.subplot(4,1,4)
plt.plot(test-y_bg)




