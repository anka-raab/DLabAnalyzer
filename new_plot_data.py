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
from scipy.optimize import curve_fit

plt.set_cmap('plasma')
#This is only for having the same energy axis everywhere!!! This we don't touch
#data_file_name = 'D:/dlab/2_color_harmonics/2023-08-14/c3_long_trajectories.hdf5'
#hfr = h5py.File(data_file_name, 'r')
#E = np.asarray(hfr.get('E'))

#That is the one we change.
data_file_name = 'D:/dlab/2_color_harmonics/2023-08-24/test2.hdf5'
hfr = h5py.File(data_file_name, 'r')
treated_images = np.asarray(hfr.get('treated_images'))
E = np.asarray(hfr.get('E'))
phase = np.asarray(hfr.get('phase'))
ratio = np.asarray(hfr.get('ratio'))
y = np.asarray(hfr.get('y'))
phase_dev = np.asarray(hfr.get('phase_dev'))
phase_error = np.asarray(hfr.get('phase_error'))

h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam


#treated_profiles_cropped = treated_profiles
#treated_profiles_cropped[300:, :] = 0
#treated_profiles_cropped[:33, :] = 0
total_counts = np.apply_over_axes(np.sum, treated_images, [0,1])
total_counts = total_counts.ravel()

res_total_counts = np.reshape(total_counts, (ratio.size, phase.size))

only_red_total_counts = np.average(res_total_counts[0, :])

##
#plt.figure(222)
#plt.plot(E,treated_profiles_cropped[:,1])
#plt.plot(E,treated_profiles_cropped[:,100])
#plt.plot(E,treated_profiles_cropped[:,200])




##

plot_name = '_phase_stability'
fig, axes = plt.subplots(2,2, num=1)
fig.set_facecolor('none')
ax = axes[0,0]
ax.cla()

im = ax.pcolormesh(phase, ratio, res_total_counts / only_red_total_counts)
fig.colorbar(im, ax=ax, orientation="vertical")
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("Total Gain")
#ax.set_ylim([0,0.44])
#im.set_clim(0,1.5)



ax = axes[0,1]
ax.cla()
phase_error[0, :] = np.nan
im = ax.pcolormesh(phase, ratio, phase_error)
fig.colorbar(im, ax=ax, orientation="vertical")
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("Phase Error (rad)")
#im.set_clim(0,0.3)

ax = axes[1,0]
ax.cla()
phase_dev[0, :] = np.nan
im = ax.pcolormesh(phase, ratio, phase_dev)
cbar = fig.colorbar(im, ax=ax, orientation="vertical")
im.set_clim(-0.5, 0.5)
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("Phase Deviation (rad)")

ax = axes[1,1]
ax.axis('off')

plt.tight_layout()

plt.savefig(data_file_name[:-5]+plot_name+'.png', dpi = 300,format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='None', edgecolor='None',
        backend=None
       )

##
phase_averaged = np.reshape(treated_images, (512, 512, ratio.size, phase.size))
phase_averaged_sum = np.squeeze(np.apply_over_axes(np.sum, phase_averaged,[0]))


##
avgd = np.squeeze(np.apply_over_axes(np.average, phase_averaged_sum, [2]))
maxed = np.squeeze(np.apply_over_axes(np.nanmax, phase_averaged_sum, [2]))


to_plot_avg = avgd.T / np.max(avgd.T)
to_plot_max = maxed.T / np.max(maxed.T)
to_plot_max[0, :] = avgd.T[0,:]/ np.max(maxed.T)

comparison_value_avg = np.sum(to_plot_avg[0,:])
comparison_value_max = np.sum(to_plot_max[0,:])

plot_name = '_gain_over_all_harmonics'
fig, ax = plt.subplots(2, 2, num=2)
fig.set_facecolor('none')
ax[0, 0].set_facecolor('none')  #

im = ax[0, 0].pcolormesh(E, ratio, 20 * np.log10(to_plot_avg), cmap='plasma')
ax[0, 0].set_aspect('auto')
ax[0, 0].set_xlim(17, 50)
ax[0, 0].set_xlabel("Energy (eV)")
ax[0, 0].set_ylabel("SH Intensity Fraction")
ax[0, 0].set_title("Counts, phase avg., normalized (dB)")
im.set_clim(-40, 0)
fig.colorbar(im, ax=ax[0, 0], orientation="vertical")
ax[0, 0].set_ylim([0,0.44])

sum_over_ratios = np.apply_over_axes(np.sum, to_plot_avg, [1]) / comparison_value_avg
ax[0, 1].plot(sum_over_ratios, ratio)
ax[0, 1].set_yticklabels([])
ax[0, 1].set_yticks([])
ax[0, 1].set_xlabel("Gain")
plt.subplots_adjust(wspace=0.1, hspace=0.2)
ax[0, 1].set_title("Best fraction of SH: {:.2f}".format(ratio[np.argmax(sum_over_ratios)]))
ax[0, 1].set_ylim([0,0.44])

im = ax[1, 0].pcolormesh(E, ratio, 20 * np.log10(to_plot_max), cmap='plasma')
ax[1, 0].set_aspect('auto')
ax[1, 0].set_xlim(17, 50)
ax[1, 0].set_xlabel("Energy (eV)")
ax[1, 0].set_ylabel("SH Intensity Fraction")
ax[1, 0].set_title("Counts, best phase., normalized (dB)")
im.set_clim(-40, 0)
fig.colorbar(im, ax=ax[1, 0], orientation="vertical")
ax[1, 0].set_ylim([0,0.44])

sum_over_ratios = np.apply_over_axes(np.sum, to_plot_max, [1]) / comparison_value_max
ax[1, 1].plot(sum_over_ratios, ratio)
ax[1, 1].set_yticklabels([])
ax[1, 1].set_yticks([])
ax[1, 1].set_xlabel("Gain")
plt.subplots_adjust(wspace=0.1, hspace=0.2)
ax[1, 1].set_title("Best fraction of SH: {:.2f}".format(ratio[np.argmax(sum_over_ratios)]))
#ax[1, 1].set_ylim([0,0.44])

plt.tight_layout()

plt.savefig(data_file_name[:-5]+plot_name+'.png', dpi = 300,format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='None', edgecolor='None',
        backend=None
       )

##

plt.figure(123)
plt.subplot(2, 1, 1)
plt.clf()
plt.imshow(np.log(treated_images[:,:,100]))
#plt.xlim(0, 600)


##
harmonics = range(15, 41)
harmonics_counts = np.zeros([ratio.size, phase.size, np.size(harmonics)])
for i, h in enumerate(harmonics):
    ind = np.argmin(abs(E - h * Eq / qe))
    counts = np.apply_over_axes(np.sum, treated_images[:,ind - 5:ind + 5,:], [0,1])
    plt.plot(ind, 500, 'o', color='black')
    counts = np.ravel(counts)
    res = np.reshape(counts, (ratio.size, phase.size))
    harmonics_counts[:, :, i] = res

harmonics_counts = harmonics_counts / np.max(harmonics_counts)
harmonics_counts_log = 20 * np.log10(harmonics_counts)

a = np.ceil(np.size(harmonics) / 2)

fig, axes = plt.subplots(1, int(a), num=41)

counter_even = 1
counter_odd = 1
for i, h in enumerate(harmonics):
    print(i, h)
    if np.mod(h, 2) == 1:
        ax = axes[counter_odd - 1]
        # plt.subplot(1, int(a),counter_odd)
        counter_odd = counter_odd + 1
        im = ax.pcolormesh(phase, ratio, harmonics_counts_log[:, :, i], cmap='plasma')
        ax.set_aspect('auto')
        ax.set_title("{}".format(h))
        ax.axis('off')
        im.set_clim(-50, 0)
# fig.colorbar(im, ax=axes.ravel().tolist())
cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

fig, axes = plt.subplots(1, int(a), num=42)
for i, h in enumerate(harmonics):
    print(i, h)
    if np.mod(h, 2) == 0:
        ax = axes[counter_even - 1]
        counter_even = counter_even + 1
        im = ax.pcolormesh(phase, ratio, harmonics_counts_log[:, :, i], cmap='plasma')
        ax.set_aspect('auto')

        ax.set_title("{}".format(h))
        ax.axis('off')
        im.set_clim(-50, 0)
# fig.colorbar(im, ax=axes.ravel().tolist())
cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

##
##


# Define the cosine function to fit
def cosine_func(x, amplitude, frequency, phi, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phi) + offset


# Perform the cosine fit
## try the fit for each pixel!!!

sine_fits_px = np.zeros([ratio.size,phase.size,512,512])
sine_fit_parameters_px = np.zeros([ratio.size, 512,512, 4]) * np.nan
test = np.reshape(treated_images, (512, 512, ratio.size, phase.size))

xs = np.arange(0,512)
ys = np.arange(0,512)

for ix, x in enumerate(xs):
    print(ix)
    for iy,y in enumerate(ys):
        for ir, r in enumerate(ratio):
            profile = np.squeeze(test[ix,iy,ir,:])
            initial_guess = [(np.max(profile) - np.min(profile)) / 2, 1 / np.pi, 0, (np.max(profile) + np.min(profile))/2]  # Initial parameter guess [amplitude, frequency, phase, offset]
            bounds = ([(np.max(profile) - np.min(profile)) / 2 - 0.2*((np.max(profile) - np.min(profile)) / 2), 1 / np.pi - 0.01, -np.pi, np.min(profile)], [(np.max(profile) - np.min(profile)) / 2 + 0.2*((np.max(profile) - np.min(profile)) / 2), 1 / np.pi + 0.01, np.pi, np.max(profile)])
            try:
                fit_params, fit_covariance = curve_fit(cosine_func, phase, profile, p0=initial_guess, bounds=bounds)
                # Extract the fitted parameters
                amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
                y_fit = cosine_func(phase, amplitude_fit, frequency_fit, phase_fit, offset_fit)
                sine_fits_px[ix,iy,ir,:] = y_fit
                sine_fit_parameters_px[ix,iy,ir, :] = fit_params
            except:
                a = 1+1
                #print(h, r, "first fail")

##
plt.figure(1111)
t = np.squeeze(test[200, 100, 5, :])
plt.plot(t)

initial_guess = [(np.max(t) - np.min(t)) / 2, 1 / np.pi, 0, (np.max(t) + np.min(t))/2]  # Initial parameter guess [amplitude, frequency, phase, offset]
bounds = ([0, 1 / np.pi - 0.01, -np.pi, -0.1], [np.max(t), 1 / np.pi + 0.01, np.pi, np.max(t)])
try:
    fit_params, fit_covariance = curve_fit(cosine_func, phase, t, p0=initial_guess, bounds=bounds)
    # Extract the fitted parameters
    amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
    y_fit = cosine_func(phase, amplitude_fit, frequency_fit, phase_fit, offset_fit)
    #sine_fits_px[ix,iy,ir,:] = y_fit
    #            sine_fit_parameters_px[ix,iy,ir, :] = fit_params
except:
    a = 1+1

plt.plot(cosine_func(phase,fit_params[0],fit_params[1],fit_params[2],fit_params[3]))
##
offsets_pixel = np.squeeze(sine_fit_parameters_px[5,:,:,2])
plt.figure(111)
plt.pcolormesh(offsets_pixel)
##
sine_fits = np.zeros_like(harmonics_counts)
sine_fit_parameters = np.zeros([ratio.size, len(harmonics), 4]) * np.nan

min_harmonics_counts = np.min(harmonics_counts)
if min_harmonics_counts < 0:
    harmonics_counts = harmonics_counts + abs(min_harmonics_counts)

counter_odd = 1
for i, h in enumerate(harmonics):
    for ind, r in enumerate(ratio):
        profile = harmonics_counts[ind, :, i]

        initial_guess = [(np.max(profile) - np.min(profile)) / 2, 1 / np.pi, 0, (np.max(profile) - np.min(
            profile))]  # Initial parameter guess [amplitude, frequency, phase, offset]
        bounds = ([0, 1 / np.pi - 0.01, -np.pi, -0.1], [1, 1 / np.pi + 0.01, np.pi, 1])
        try:
            fit_params, fit_covariance = curve_fit(cosine_func, phase, profile, p0=initial_guess, bounds=bounds)
            # Extract the fitted parameters
            amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
            y_fit = cosine_func(phase, amplitude_fit, frequency_fit, phase_fit, offset_fit)
            sine_fits[ind, :, i] = y_fit
            sine_fit_parameters[ind, i, :] = fit_params
        except:
            print(h, r, "first fail")
            try:
                initial_guess = [(np.max(profile) - np.min(profile)) / 2, 1 / np.pi, 0.1,
                                 (np.max(profile) - np.min(profile))]
                fit_params, fit_covariance = curve_fit(cosine_func, phase, profile, p0=initial_guess)
                # Extract the fitted parameters
                amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
                y_fit = cosine_func(phase, amplitude_fit, frequency_fit, phase_fit, offset_fit)
                sine_fits[ind, :, i] = y_fit
                sine_fit_parameters[ind, i, :] = fit_params

            except:
                # plt.figure()
                # plt.plot(profile)
                print("second fail")
                sine_fits[ind, :, i] = np.zeros_like(phase) * np.nan

sine_fits = sine_fits / np.nanmax(sine_fits)
sine_fits_log = 20 * np.log10(sine_fits)
sine_fit_parameters[0, :, :]=np.nan

fig, axes = plt.subplots(1, int(a), num=61)

counter_even = 1
counter_odd = 1
for i, h in enumerate(harmonics):
    print(i, h)
    if np.mod(h, 2) == 1:
        ax = axes[counter_odd - 1]
        # plt.subplot(1, int(a),counter_odd)
        counter_odd = counter_odd + 1
        im = ax.pcolormesh(phase, ratio, sine_fits_log[:, :, i], cmap='plasma')
        ax.set_aspect('auto')
        ax.set_title("{}".format(h))
        ax.axis('off')
        im.set_clim(-50, 0)
cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

fig, axes = plt.subplots(1, int(a), num=62)

counter_even = 1
counter_odd = 1
for i, h in enumerate(harmonics):
    print(i, h)
    if np.mod(h, 2) == 0:
        ax = axes[counter_odd - 1]
        # plt.subplot(1, int(a),counter_odd)
        counter_odd = counter_odd + 1
        im = ax.pcolormesh(phase, ratio, sine_fits_log[:, :, i], cmap='plasma')

        ax.set_aspect('auto')
        ax.set_title("{}".format(h))
        ax.axis('off')
        im.set_clim(-50, 0)
cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')




##

# amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
plot_name='_phase_oscillation_parameters'
fig = plt.figure(70)
fig.set_facecolor('none')  # or fig.set_facecolor('w', alpha=0)

plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(abs(np.flipud(sine_fit_parameters[:, :, 0])), cmap='plasma',
           extent=[min(harmonics), max(harmonics), min(ratio), max(ratio)])
plt.gca().set_aspect('auto')
plt.colorbar()
plt.clim(0, 0.5)
plt.xlabel("Harmonic Order")
plt.ylabel("SH Intensity Fraction")
plt.title("Amplitude")
plt.ylim([0,0.44])

plt.subplot(2, 2, 2)
plt.imshow(np.flipud(sine_fit_parameters[:, :, 1]), cmap='plasma',
           extent=[min(harmonics), max(harmonics), min(ratio), max(ratio)])
plt.gca().set_aspect('auto')
plt.colorbar()
plt.clim(1 / np.pi - 0.02, 1 / np.pi + 0.02)
plt.xlabel("Harmonic Order")
plt.ylabel("SH Intensity Fraction")
plt.title("Oscillation Frequency")
plt.ylim([0,0.44])

plt.subplot(2, 2, 3)
plt.imshow(np.flipud(sine_fit_parameters[:, :, 2]), cmap='plasma',
           extent=[min(harmonics), max(harmonics), min(ratio), max(ratio)])
plt.gca().set_aspect('auto')
plt.colorbar()
plt.clim(-np.pi, np.pi)
plt.xlabel("Harmonic Order")
plt.ylabel("SH Intensity Fraction")
plt.title("Oscillation Phase Offset")
plt.ylim([0,0.44])

plt.subplot(2, 2, 4)
plt.imshow(np.flipud(sine_fit_parameters[:, :, 3]), cmap='plasma',
           extent=[min(harmonics), max(harmonics), min(ratio), max(ratio)])
plt.gca().set_aspect('auto')
plt.colorbar()
plt.clim(0, 0.5)
plt.xlabel("Harmonic Order")
plt.ylabel("SH Intensity Fraction")
plt.title("Oscillation Offset")
plt.ylim([0,0.44])

plt.tight_layout()


plt.savefig(data_file_name[:-5]+plot_name+'.png', dpi = 300,format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='None', edgecolor='None',
        backend=None
       )
##
plt.figure(71)
plt.clf()
plt.pcolormesh(harmonics,ratio,sine_fit_parameters[:, :, 0]/sine_fit_parameters[:, :, 3])
plt.xlabel("Harmonic Order")
plt.ylabel("SH Intensity Fraction")
plt.title("Amplitude/Offset")
plt.colorbar()

## harmonics_counts to gain

plot_name = '_best_ratio_and_gain_for_individual_harmonics'

from scipy.ndimage import gaussian_filter1d

harmonics_counts_gain = harmonics_counts/np.apply_over_axes(np.average, harmonics_counts[0,:,:],[0])

phase_averaged = np.squeeze(np.apply_over_axes(np.average,harmonics_counts_gain, [1]))
phase_max = np.squeeze(np.apply_over_axes(np.nanmax,harmonics_counts_gain, [1]))

phase_averaged_filter = gaussian_filter1d(phase_averaged,3,0)
phase_max_filter = gaussian_filter1d(phase_averaged_filter,3,0)

maxima_avg = np.argmax(phase_averaged_filter,0)
maxima_max = np.argmax(phase_max_filter,0)

harmonics = np.asarray(harmonics)

fig, axes = plt.subplots(2,1,num=91)
#fig.set_facecolor('none')

ax1=axes[0]
#ax1.set_facecolor('none')  #

res = np.zeros([harmonics.size,5])*np.nan


for ind,h in enumerate(np.asarray(harmonics)):
    ax1.scatter(h,ratio[maxima_avg[ind]],color = 'k')
    res[ind,0]= h
    res[ind,1] = ratio[maxima_avg[ind]]

ax1.set_ylim([0,0.44])
ax1.set_title("Data averaged over phase")
ax1.set_xlabel("Harmonic Order")
ax1.set_ylabel("Opt. SH Intensity Fraction")

ax2 = ax1.twinx()
#ax2.set_facecolor('none')  #

ax2.set_ylabel("Maximum Gain", color = 'r')
ax2.spines['right'].set_color('r')
ax2.tick_params(axis='y', colors='r')

for ind,h in enumerate(np.asarray(harmonics)):
    if np.mod(h,2)==1:
        ax2.scatter(h,phase_averaged_filter[maxima_avg[ind],ind],marker ='x',color = 'r')
        res[ind, 2] = phase_averaged_filter[maxima_avg[ind],ind]
ax2.set_yscale('log')


ax1=axes[1]
#ax1.set_facecolor('none')  #

for ind,h in enumerate(np.asarray(harmonics)):
    ax1.scatter(h,ratio[maxima_max[ind]],color ='k')
    res[ind, 3] = ratio[maxima_max[ind]]

ax1.set_ylim([0,0.44])
ax1.set_title("Optimum phase chosen")
ax1.set_xlabel("Harmonic Order")
ax1.set_ylabel("Opt. SH Intensity Fraction")

ax2 = ax1.twinx()
#ax2.set_facecolor('none')  #

ax2.set_ylabel("Maximum Gain", color = 'r')
ax2.spines['right'].set_color('r')
ax2.tick_params(axis='y', colors='r')


for ind,h in enumerate(np.asarray(harmonics)):
    if np.mod(h,2)==1:
        ax2.scatter(h,phase_max_filter[maxima_max[ind],ind],marker ='x',color ='r')
        res[ind, 4] = phase_max_filter[maxima_max[ind],ind]
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(data_file_name[:-5]+plot_name+'.png', dpi = 300,format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None
       )

#

np.savetxt(data_file_name[:-5]+plot_name+".txt",res)

#sys.exit()

## now we look at harmonic 21!
h = 25
fig,axes = plt.subplots(2,2,num=100)
fig.set_facecolor('none')  # or fig.set_facecolor('w', alpha=0)

ax = axes[0,0]
im = ax.pcolormesh(phase, ratio, harmonics_counts[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(harmonics_counts[0, :, np.argmin(abs(h-np.asarray(harmonics)))]), cmap='plasma')
im.set_clim(0, np.max(harmonics_counts[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(harmonics_counts[0, :, np.argmin(abs(h-np.asarray(harmonics)))])))
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("H{} Signal Gain".format(int(h)))
fig.colorbar(im)

ax = axes[0,1]
im = ax.pcolormesh(phase, ratio, sine_fits[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(sine_fits[0, :, np.argmin(abs(h-np.asarray(harmonics)))]), cmap='plasma')
im.set_clim(0, np.max(sine_fits[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(sine_fits[0, :, np.argmin(abs(h-np.asarray(harmonics)))])))
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("H{} Fitted Data: Gain".format(int(h)))
fig.colorbar(im)

h = 31
ax = axes[1,0]
im = ax.pcolormesh(phase, ratio, harmonics_counts[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(harmonics_counts[0, :, np.argmin(abs(h-np.asarray(harmonics)))]), cmap='plasma')
im.set_clim(0, np.max(harmonics_counts[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(harmonics_counts[0, :, np.argmin(abs(h-np.asarray(harmonics)))])))
ax.set_xlabel("Relative Phase (rad)")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("H{}: Signal Gain".format(int(h)))
fig.colorbar(im)

ax = axes[1,1]
im = ax.pcolormesh(phase, ratio, sine_fits[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(sine_fits[0, :, np.argmin(abs(h-np.asarray(harmonics)))]), cmap='plasma')
im.set_clim(0, np.max(sine_fits[:, :, np.argmin(abs(h-np.asarray(harmonics)))] / np.average(sine_fits[0, :, np.argmin(abs(h-np.asarray(harmonics)))])))
ax.set_xlabel("Phase")
ax.set_ylabel("SH Intensity Fraction")
ax.set_title("H{} Fitted Data: Gain".format(int(h)))
fig.colorbar(im)

#cax = fig.add_axes([0.12, 0.02, 0.6, 0.03])  # Adjust the position and size as needed
#cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

plt.tight_layout()

##

normalized_profiles = treated_profiles/np.nanmax(treated_profiles)
high_energy_values = np.sum(normalized_profiles[631:653,:],0)
high_energy_values = np.sum(normalized_profiles[575:595,:],0)

high_energy_values = np.reshape(high_energy_values,(ratio.size, phase.size))


plt.figure(12345)
#plt.plot(normalized_profiles[:,200])
plt.subplot(1,2,1)
high_energy_values = np.sum(normalized_profiles[575:595,:],0)
high_energy_values = np.reshape(high_energy_values,(ratio.size, phase.size))
plt.pcolormesh(phase,ratio,high_energy_values)
plt.title("H32")
plt.clim(0,0.35)
plt.colorbar()

plt.subplot(1,2,2)
high_energy_values = np.sum(normalized_profiles[608:635,:],0)
high_energy_values = np.reshape(high_energy_values,(ratio.size, phase.size))
plt.pcolormesh(phase,ratio,high_energy_values)
plt.title("H33")
plt.clim(0,0.35)

plt.colorbar()

plt.tight_layout()
#605 - 631
#631 - 653
testindex = 455
peaks,_ = find_peaks(gaussian_filter1d(normalized_profiles[:,testindex],3), prominence = 0.001)
plt.figure(123456)
plt.plot(E,normalized_profiles[:,testindex])
plt.plot(E[peaks],normalized_profiles[peaks,testindex], 'o')
plt.yscale('log')

plt.figure(1234567)
plt.subplot(3,1,1)
plt.plot(normalized_profiles[:,testindex])
plt.yscale('log')
plt.subplot(3,1,2)
plt.plot(E,normalized_profiles[:,testindex])
plt.subplot(3,1,3)
plt.plot(E/(Eq/qe),normalized_profiles[:,testindex])
#plt.plot(normalized_profiles[:,500])
#plt.plot(normalized_profiles[:,800])


"""
plt.plot(E/(Eq/qe),normalized_profiles[:,200])
plt.plot(E/(Eq/qe),normalized_profiles[:,400])
plt.plot(E/(Eq/qe),normalized_profiles[:,600])
plt.plot(E/(Eq/qe),normalized_profiles[:,800])
plt.plot(E/(Eq/qe),normalized_profiles[:,1000])
plt.plot(E/(Eq/qe),normalized_profiles[:,1190])
"""



## now with gain as colorscale

fig, axes = plt.subplots(1, int(a), num=91)
plt.cla()
counter_even = 1
counter_odd = 1

import matplotlib.colors as colors
norm = colors.LogNorm(vmin=(sine_fits[:, :, :]/np.average(sine_fits[0, :, :])).min(), vmax=(sine_fits[:, :, :]/np.average(sine_fits[0, :, :])).max())



for i, h in enumerate(harmonics):
    print(i, h)
    if np.mod(h, 2) == 1:
        ax = axes[counter_odd - 1]
        # plt.subplot(1, int(a),counter_odd)
        counter_odd = counter_odd + 1
        im = ax.pcolormesh(phase, ratio, sine_fits[:, :, i]/np.average(sine_fits[0, :, i]), cmap='plasma',norm=norm)
        ax.set_aspect('auto')
        ax.set_title("{}".format(h))
        ax.axis('off')
        im.set_clim(1,15)
cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # Adjust the position and size as needed
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')



##
path = "D:/dlab/2_color_harmonics/2023-06-27/"
end = "p3_best_ratio_and_gain_for_individual_harmonics.txt"
names = ['c2_8', 'c3_0','c3_2','c3_4','c3_6','c3_8','c4_0','c4_2',]
all_res=np.zeros([20,5,len(names)])
for i,name in enumerate(names):
    test = np.loadtxt(path+name+end)
    all_res[:,:,i] = test

hh = all_res[:,0,0]
intensities = np.arange(2.8,4.2,0.2)*100*1e-6*0.9*2/(190e-15*np.pi*(27e-6)**2)*1e-4*1e-14

import matplotlib.colors as colors
norm = colors.LogNorm(vmin=np.nanmin(all_res[:,2,:]),vmax=np.nanmax(all_res[:,2,:]))


plt.figure(888)
plt.subplot(2,2,1)
plt.pcolormesh(intensities, hh, all_res[:,1,:])
plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Harmonic Order")
plt.title("Opt. SH Int. Fraction: Phase averaged")
plt.colorbar()

plt.subplot(2,2,2)
plt.pcolormesh(intensities, hh, all_res[:,3,:])
plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Harmonic Order")
plt.title("Opt. SH Int. Fraction: Best Phase")
plt.colorbar()


plt.subplot(2,2,3)
plt.pcolormesh(intensities, hh, all_res[:,2,:], norm=norm)
plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Harmonic Order")
plt.title("Gain at opt. SH")
plt.colorbar()

plt.subplot(2,2,4)
plt.pcolormesh(intensities, hh, all_res[:,4,:], norm = norm)
plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Harmonic Order")
plt.title("Gain at opt. SH")
plt.colorbar()

plt.tight_layout()

##
plt.figure(889)
index = 1
plt.plot(intensities, all_res[0,index,:],linestyle = '-' ,color ='g', label = "H15")
plt.plot(intensities, all_res[1,index,:], linestyle = '--' ,color ='g',label = "H16")
#plt.plot(intensities, all_res[3,1,:], label = "H17")
#plt.plot(intensities, all_res[4,1,:], label = "H18")
#plt.plot(intensities, all_res[5,1,:], label = "H19")
#plt.plot(intensities, all_res[6,1,:], label = "H20")
#plt.plot(intensities, all_res[7,1,:], label = "H21")
#plt.plot(intensities, all_res[8,1,:], label = "H22")
plt.plot(intensities, all_res[8,index,:],  linestyle = '-' ,color ='b',label = "H23")
plt.plot(intensities, all_res[10,index,:], linestyle = '--' ,color ='b',label = "H25")
plt.plot(intensities, all_res[12,index,:], linestyle = ':' ,color ='b',label = "H27")


plt.plot(intensities, all_res[15,index,:], linestyle = '-' ,color ='k',label = "H28")
plt.plot(intensities, all_res[17,index,:], linestyle = '--' ,color ='k',label = "H32")
plt.plot(intensities, all_res[19,index,:], linestyle = ':' ,color ='k',label = "H34")

plt.plot(intensities, all_res[16,index,:], linestyle = '-' ,color ='r', label = "H31")
plt.plot(intensities, all_res[18,index,:], linestyle = '--' ,color ='r',label = "H33")

plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Optimum SH intensity Ratio")
plt.legend()


plt.figure(890)
index = 2
plt.plot(intensities, all_res[0,index,:],linestyle = '-' ,color ='g', label = "H15")
plt.plot(intensities, all_res[1,index,:], linestyle = '--' ,color ='g',label = "H16")
#plt.plot(intensities, all_res[3,1,:], label = "H17")
#plt.plot(intensities, all_res[4,1,:], label = "H18")
#plt.plot(intensities, all_res[5,1,:], label = "H19")
#plt.plot(intensities, all_res[6,1,:], label = "H20")
#plt.plot(intensities, all_res[7,1,:], label = "H21")
#plt.plot(intensities, all_res[8,1,:], label = "H22")
plt.plot(intensities, all_res[8,index,:],  linestyle = '-' ,color ='b',label = "H23")
plt.plot(intensities, all_res[10,index,:], linestyle = '--' ,color ='b',label = "H25")
plt.plot(intensities, all_res[12,index,:], linestyle = ':' ,color ='b',label = "H27")


plt.plot(intensities, all_res[15,index,:], linestyle = '-' ,color ='k',label = "H28")
plt.plot(intensities, all_res[17,index,:], linestyle = '--' ,color ='k',label = "H32")
plt.plot(intensities, all_res[19,index,:], linestyle = ':' ,color ='k',label = "H34")

plt.plot(intensities, all_res[16,index,:], linestyle = '-' ,color ='r', label = "H31")
plt.plot(intensities, all_res[18,index,:], linestyle = '--' ,color ='r',label = "H33")

plt.xlabel("Intensity (1e14 W/cm^2)")
plt.ylabel("Gain at best SH Intensity Fraction")
plt.yscale('log')
plt.legend()



#plt.plot(intensities, all_res[17,1,:])
#plt.plot(intensities, all_res[18,1,:])
#plt.plot(intensities, all_res[19,1,:])