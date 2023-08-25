import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


data_file_name = 'D:/dlab/2_color_harmonics/2023-08-09/3stepmodel.hdf5'
hfr = h5py.File(data_file_name, 'r')
phases = np.asarray(hfr.get('phases'))
ratios = np.asarray(hfr.get('ratio'))
T_i = np.asarray(hfr.get('T_i'))
T_returns_all = np.asarray(hfr.get('T_returns_all'))
E_kin_returns_all = np.asarray(hfr.get('E_kin_returns_all'))
laser_cycle = np.asarray(hfr.get('laser_cycle'))
Up = np.asarray(hfr.get('Up'))
E0 = np.asarray(hfr.get('E0'))

## Divide it into halfcycle subarrays!!


tex = T_returns_all[6,2,:]
Eex = E_kin_returns_all[6,2,:]


numeric_indices = np.where(~np.isnan(tex))[0]
diff_numeric_indices = np.diff(numeric_indices)
split_indices = numeric_indices[np.where(diff_numeric_indices > 1)[0] + 1]

# Split the 'x' array based on the split indices
split_t_arrays = np.split(tex, split_indices)
split_E_arrays = np.split(Eex, split_indices)

first_half_cycle_t = split_t_arrays[0]
second_half_cycle_t = split_t_arrays[1]

first_half_cycle_E = split_E_arrays[0]
second_half_cycle_E = split_E_arrays[1]

plt.figure(1)
plt.plot(first_half_cycle_t/laser_cycle,first_half_cycle_E)
plt.plot(second_half_cycle_t/laser_cycle,second_half_cycle_E)

tr = np.linspace(-1,0.5,500)*laser_cycle
f_interp_1 = interp1d(first_half_cycle_t, first_half_cycle_E, bounds_error=False,fill_value=np.nan)
f_interp_2 = interp1d(second_half_cycle_t, second_half_cycle_E, bounds_error=False,fill_value=np.nan)

test_E1 = f_interp_1(tr)
test_E2 = f_interp_2(tr)

plt.plot(tr/laser_cycle,test_E1)
plt.plot(tr/laser_cycle,test_E2)

##
tr = np.linspace(-1,0.5,500)*laser_cycle
ekins = np.zeros([E_kin_returns_all.shape[0],E_kin_returns_all.shape[1],tr.size,2])*np.nan

for ind_phase, ph in enumerate(phases):
    for ind_ratio, r in enumerate(ratios):

        tex = T_returns_all[ind_ratio, ind_phase, :]
        Eex = E_kin_returns_all[ind_ratio, ind_phase, :]

        numeric_indices = np.where(~np.isnan(tex))[0]
        diff_numeric_indices = np.diff(numeric_indices)
        split_indices = numeric_indices[np.where(diff_numeric_indices > 1)[0] + 1]

        # Split the 'x' array based on the split indices
        split_t_arrays = np.split(tex, split_indices)
        split_E_arrays = np.split(Eex, split_indices)

        first_half_cycle_t = split_t_arrays[0]
        second_half_cycle_t = split_t_arrays[1]

        first_half_cycle_E = split_E_arrays[0]
        second_half_cycle_E = split_E_arrays[1]

        f_interp_1 = interp1d(first_half_cycle_t, first_half_cycle_E, bounds_error=False, fill_value=np.nan)
        f_interp_2 = interp1d(second_half_cycle_t, second_half_cycle_E, bounds_error=False, fill_value=np.nan)

        test_E1 = f_interp_1(tr)
        test_E2 = f_interp_2(tr)

        ekins[ind_ratio, ind_phase, :,0] = test_E1
        ekins[ind_ratio, ind_phase, :,1] = test_E2

##
plt.figure(1)
plt.subplot(1,2,1)
plt.pcolormesh(phases,tr/laser_cycle,ekins[37,:,:,0].T)
plt.subplot(1,2,2)
plt.pcolormesh(phases,tr/laser_cycle,ekins[37,:,:,1].T)


plt.figure(2)
plt.subplot(1,2,1)
plt.pcolormesh(ratios,tr/laser_cycle,ekins[:,1,:,0].T)
plt.subplot(1,2,2)
plt.pcolormesh(ratios,tr/laser_cycle,ekins[:,1,:,1].T)

##
plt.figure(4)
plt.subplot(1,2,1)
plt.pcolormesh(phases,ratios,np.nanmax(ekins[:,:,:,0],2))
plt.subplot(1,2,2)
plt.pcolormesh(phases,ratios,np.nanmax(ekins[:,:,:,1],2))


##

plt.figure(3)
for ind_ratio,ratio in enumerate(ratios):
    plt.subplot(7,7,ind_ratio+1)
    plt.pcolormesh(phases,tr/laser_cycle,ekins[ind_ratio,:,:,0].T/Up,cmap='plasma')
    plt.clim(0,3.5)
    plt.title(np.round(ratio,2))

plt.figure(4)
for ind_ratio, ratio in enumerate(ratios):
    plt.subplot(7, 7, ind_ratio + 1)
    plt.pcolormesh(phases, tr / laser_cycle, ekins[ind_ratio, :, :, 1].T / Up, cmap='plasma')
    plt.clim(0, 3.5)
    plt.title(np.round(ratio, 2))
    #plt.xlabel("Phase (rad)")
    #plt.ylabel("Return Time ")
    #plt.tight_layout()
##
ekins = np.zeros_like(E_kin_returns_all)*np.nan
min_tr = np.nanmin(T_returns_all)/laser_cycle
max_tr = np.nanmax(T_returns_all)/laser_cycle
tr = np.linspace(min_tr,max_tr,ekins.shape[2])
valid_indices = np.isfinite(E_kin_returns_all[0,0,:])
f_interp = interp1d(T_returns_all[0,0, valid_indices]/laser_cycle, E_kin_returns_all[0,0, valid_indices], bounds_error=False,fill_value=np.nan)

e_new = f_interp(tr)

plt.figure(2)
plt.plot(T_returns_all[0,0,:]/laser_cycle,E_kin_returns_all[0,0,:] )
plt.plot(tr,e_new)



##
ekins = np.zeros_like(E_kin_returns_all)*np.nan
min_tr = np.nanmin(T_returns_all)/laser_cycle
max_tr = np.nanmax(T_returns_all)/laser_cycle

tr = np.linspace(min_tr,max_tr,ekins.shape[2])
for ind_phase, ph in enumerate(phases):
    for ind_ratio, r in enumerate(ratios):
        valid_indices = np.isfinite(E_kin_returns_all[ind_ratio,ind_phase,:])
        f_interp = interp1d(T_returns_all[ind_ratio,ind_phase, valid_indices]/laser_cycle, E_kin_returns_all[ind_ratio,ind_phase, valid_indices], bounds_error=False,fill_value=np.nan)
        ekins[ind_ratio,ind_phase,:] = f_interp(tr)
##
plt.figure(1)
plt.pcolormesh(tr,phases,ekins[25,:,:])
## interpolate all tr vs E
min_tr = np.nanmin(T_returns_all)
max_tr = np.nanmax(T_returns_all)

tr = np.linspace(min_tr,max_tr,500)
ekin = np.apply_over_axes(np.interp)
##
maxE = np.nanmax(E_kin_returns_all)
ratio_index = 20
Es = E_kin_returns_all[ratio_index, :, :]
Ts = T_returns_all[ratio_index, :, :]
##
plt.figure(11)
plt.pcolormesh(phases,ratios,T_returns_all[:,:,80])



##
for ind_phase, ph in enumerate(phases[0:5]):
    #for ind_t, tr in enumerate(Ts[ind_phase,:]):
    E = Es[ind_phase,:]
    Tr = Ts[ind_phase,:]
    plt.scatter(ph,Tr, c = E/maxE,cmap='plasma')