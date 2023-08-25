import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

me = 9.1e-31
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam

e = qe  # Example value for e
E0 = 2e11  # Example value for E0, V/m
w = c / lam * 2 * np.pi  # Example value for w
laser_cycle = 1 / (c / lam)

total = 1
sh_harmonic_intensity_ratio = 0.5
f_intensity_ratio = total - sh_harmonic_intensity_ratio

two_color = True

phase_temp = np.pi / 2


def function(u, t):
    if not two_color:
        return (u[1], np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(t * w))
    else:
        return (u[1], np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
            sh_harmonic_intensity_ratio) * E0 * e / me * np.cos(t * 2 * w + phase_temp))


def velocity(u, t):
    if not two_color:
        return np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(t * w)
    else:
        return np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
            sh_harmonic_intensity_ratio) * E0 * e / me * np.cos(t * 2 * w + phase_temp)


def find_zero_crossing(x, y):
    zero_crossing_index = None

    for i in range(1, len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]):
            zero_crossing_index = i
            break

    return zero_crossing_index


def find_local_maxima(x, y):
    local_maxima_indices = []

    for i in range(1, len(y) - 1):
        if np.isnan(y[i - 1]) or np.isnan(y[i]) or np.isnan(y[i + 1]):
            continue
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_maxima_indices.append(i)

    return local_maxima_indices, x[local_maxima_indices], y[local_maxima_indices]


y0 = [0, 0]

Up = e ** 2 * E0 ** 2 / (4 * me * w ** 2)

ionization_time_steps = 150
time_axis_steps = 500

time_axis = np.zeros([time_axis_steps, ionization_time_steps]) * np.nan
trajectories = np.zeros([time_axis_steps, ionization_time_steps]) * np.nan
return_energies = np.zeros([ionization_time_steps, 1]) * np.nan

T_i = np.linspace(-1.1, 1, ionization_time_steps)

for time_ind, ti in enumerate(T_i):
    ts = np.linspace(ti, 2, time_axis_steps) * laser_cycle
    Ts = ts / 3.435e-15
    us = odeint(function, y0, ts)
    ys = us[:, 0]
    ind = find_zero_crossing(Ts, ys)
    if ind is not None:
        start_time = ti * laser_cycle
        return_time = ts[ind]
        new_t = np.linspace(start_time, return_time, time_axis.shape[0])
        v = odeint(velocity, 0, new_t)
        E_kin_return = 0.5 * me * v[-1] ** 2
        return_energies[time_ind] = E_kin_return

    time_axis[:ind, time_ind] = Ts[:ind]
    trajectories[:ind, time_ind] = ys[:ind]

colors = plt.cm.jet(np.linspace(0, 1, 500))
max_energy = np.nanmax(return_energies)
color_indices = (return_energies / (Up * 3.17)) * 500
color_indices = color_indices.astype(int)

fig, ax = plt.subplots(num=2000)
fig.set_facecolor('none')  # or fig.set_facecolor('w', alpha=0)
ax.set_facecolor('none')  # or ax.set_facecolor('w', alpha=0)sample = treated_images_cropped[:, :, 1173]
ax.cla()
ax.set_xlim([-1.1, 1.1])
for i in np.arange(0, time_axis.shape[1]):
    col = color_indices[i]
    if col < 0:
        asdf = 1
        #
        plt.plot(time_axis[:, i], trajectories[:, i], 'k')
    else:

        index = color_indices[i]
        if index == len(colors): index = index - 1
        print(index)
        col = colors[index]
        plt.plot(time_axis[:, i], trajectories[:, i], color=col)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
ax.set_yticks([])

# ax.set_xlabel("Ionization Time ", fontsize=18)
# ax.set_ylabel("Signal (arb.u.)",fontsize=18)
ax.tick_params(axis='x', labelsize=28)

##

plt.figure(2001)
plt.plot(T_i, return_energies)
##
plt.figure(2000)
ax2 = plt.gca().twinx()
if not two_color:
    field = np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(T_i * laser_cycle * w)
    ax2.plot(T_i, field / np.max(field))
else:
    field = np.sqrt(f_intensity_ratio) * E0 * e / me * np.cos(T_i * laser_cycle * w) + np.sqrt(
        sh_harmonic_intensity_ratio) * E0 * e / me * np.cos(T_i * laser_cycle * 2 * w)
    ax2.plot(T_i, field / np.max(field))

##

plt.figure(1)
plt.clf()

two_color = True

sh_harmonic_intensity_ratio = np.linspace(0, 0.5, 42)
phases = np.linspace(0, 2 * np.pi, 30)

resolution = 500

f_intensity_ratio = total - sh_harmonic_intensity_ratio
T_i = np.linspace(-1.1, 0.99, resolution)
E_kin_returns_all = np.zeros([len(sh_harmonic_intensity_ratio), len(phases), resolution])
T_returns_all = np.zeros([len(sh_harmonic_intensity_ratio), len(phases), resolution])

for ind_ratio, sh_ratio in enumerate(sh_harmonic_intensity_ratio):
    print(ind_ratio)
    f_ratio = total - sh_ratio

    for ind_phase, phase in enumerate(phases):
        print(ind_phase)


        def function(u, t):
            if not two_color:
                return (u[1], np.sqrt(f_ratio) * E0 * e / me * np.cos(t * w))
            else:
                return (u[1], np.sqrt(f_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
                    sh_ratio) * E0 * e / me * np.cos(t * 2 * w + phase))


        def velocity(u, t):
            if not two_color:
                return np.sqrt(f_ratio) * E0 * e / me * np.cos(t * w)
            else:
                return np.sqrt(f_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
                    sh_ratio) * E0 * e / me * np.cos(t * 2 * w + phase)


        E_kin_returns = np.zeros_like(T_i) * np.nan
        T_returns = np.zeros_like(T_i) * np.nan
        for time_ind, ti in enumerate(T_i):
            ts = np.linspace(ti, ti+1, resolution*2) * laser_cycle
            Ts = ts / laser_cycle
            us = odeint(function, y0, ts)
            ys = us[:, 0]
            ind = find_zero_crossing(Ts, ys)
            if ind is not None:
                start_time = ti * laser_cycle
                return_time = ts[ind]
                new_t = np.linspace(start_time, return_time, resolution*2)
                v = odeint(velocity, 0, new_t)
                E_kin_return = 0.5 * me * v[-1] ** 2
                E_kin_returns[time_ind] = E_kin_return
                T_returns[time_ind] = return_time
        # plt.plot(T_i,E_kin_returns)
        E_kin_returns_all[ind_ratio, ind_phase, :] = E_kin_returns
        T_returns_all[ind_ratio, ind_phase, :] = T_returns

##
plt.figure(111)
plt.plot(T_returns_all[0,0,:]/laser_cycle,E_kin_returns_all[0,0,:]/Up)
plt.plot(T_returns_all[1,0,:]/laser_cycle,E_kin_returns_all[1,0,:]/Up)
plt.plot(T_returns_all[2,0,:]/laser_cycle,E_kin_returns_all[2,0,:]/Up)
plt.plot(T_returns_all[3,0,:]/laser_cycle,E_kin_returns_all[3,0,:]/Up)
##
import os
import h5py
date = '2023-08-09'
path_save = 'D:/dlab/2_color_harmonics/'+date
if os.path.exists(path_save):
    print("Path exists!")
    data_filename = 'D:/dlab/2_color_harmonics/' + date + '/3stepmodel_more.hdf5'
else:
    os.makedirs(path_save)
    data_filename = 'D:/dlab/2_color_harmonics/' + date + '/3stepmodel_more.hdf5'

hf = h5py.File(data_filename, 'w')
##
hf.create_dataset('T_i', data=T_i)
hf.create_dataset('ratio', data=sh_harmonic_intensity_ratio)
hf.create_dataset('phases', data=phases)

hf.create_dataset('T_returns_all', data=T_returns_all)
hf.create_dataset('E_kin_returns_all', data=E_kin_returns_all)
hf.create_dataset('laser_cycle', data=laser_cycle)
hf.create_dataset('Up', data=Up)
hf.create_dataset('E0', data=E0)
##
hf.close()
##


plt.figure(111)


maxE = np.nanmax(E_kin_returns_all)
ratio_index = 20
Es = E_kin_returns_all[ratio_index, :, :]
Ts = T_returns_all[ratio_index, :, :]
for ind_phase, ph in enumerate(phases):
    for ind_t, tr in enumerate(Ts[ind_phase,:]):
        print(tr)
        E = Es[ind_phase,ind_t]

        plt.scatter(tr,ph)




##
max_ekins = np.nanmax(E_kin_returns_all, axis=2)

plt.figure(111)
plt.pcolormesh(phases, sh_harmonic_intensity_ratio, max_ekins / (Up))
plt.colorbar()
plt.xlabel("Relative Phase (rad)")
plt.ylabel("SH Intensity Ratio")
plt.title("Cut-off (normalized to single-color Up)")
## find peaks in all return energies
from scipy.signal import find_peaks

for ind_ratio, ratio in enumerate(sh_harmonic_intensity_ratio):
    for ind_phase, phase in enumerate(phases):
        profile = E_kin_returns_all[ind_ratio, ind_phase, :]
        profile[np.isnan(profile)] = 0
        profile = profile / np.max(profile)
        peaks, _ = find_peaks(profile, prominence=0.3)
        print(peaks.shape)

##
example = np.squeeze(E_kin_returns_all[0, :, :])
plt.figure(11)
plt.clf()
plt.pcolormesh(phases, T_i, 20 * np.log10(example.T / np.max(example.T)), cmap='plasma')
plt.colorbar()
plt.clim(-30, 0)  ##

##
fig, ax = plt.subplots()
fig.set_facecolor('none')  # or fig.set_facecolor('w', alpha=0)
ax.set_facecolor('none')  #
sh_ratio = 0
phase = 0
ogfield = np.sqrt(1 - sh_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
    sh_ratio) * E0 * e / me * np.cos(t * 2 * w + phase)
t = (np.linspace(0, 2, 100) - 1) * laser_cycle

plt.plot(t / laser_cycle, ogfield, label="1 color", color='k', linewidth=3)

sh_ratio = 0.5
phase = 0
field = np.sqrt(1 - sh_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
    sh_ratio) * E0 * e / me * np.cos(t * 2 * w)
plt.plot(t / laser_cycle, field, label="2 color, phase = 0", color='r', linewidth=3)

sh_ratio = 0.5
phase = np.pi / 2
field = np.sqrt(1 - sh_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
    sh_ratio) * E0 * e / me * np.cos(t * 2 * w + phase)
plt.plot(t / laser_cycle, field, label="2 color, phase = pi", color='b', linewidth=3)

sh_ratio = 1
phase = 1 * np.pi
field = np.sqrt(1 - sh_ratio) * E0 * e / me * np.cos(t * w) + np.sqrt(
    sh_ratio) * E0 * e / me * np.cos(t * 2 * w + phase)
# plt.plot(t / laser_cycle, field, label="2 color, phase = pi", color = 'g')

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=15)

# Remove the axis ticks and tick labels
# ax.set_xticks([])
ax.set_yticks([])
plt.xlim([-1, 1])

##
plt.figure(88)
e_sc = E_kin_returns_all[0, 0, :]
e_2c_0 = E_kin_returns_all[2, 0, :]
e_2c_pi2 = E_kin_returns_all[2, 1, :]

fig, ax = plt.subplots()

ax.plot(T_i, e_sc, 'k', linewidth=3)
ax.plot(T_i, e_2c_0, 'r', linewidth=3)
ax.plot(T_i, e_2c_pi2, 'b', linewidth=3)

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)

# Remove the axis ticks and tick labels
# ax.set_xticks([])
ax.set_yticks([])
