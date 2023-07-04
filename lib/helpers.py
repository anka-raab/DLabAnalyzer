import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import sys
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import lib.helpers as help
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

me = 9.1e-31
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


import numpy as np


def find_fwhm(x, y):
    """Find the Full Width Half Maximum (FWHM) of a peak.

    Args:
        x (array): Array of x-values.
        y (array): Array of y-values.

    Returns:
        float: Full Width Half Maximum (FWHM) value.
    """
    # Find the maximum y-value and its corresponding x-value
    max_index = np.argmax(y)
    max_x = x[max_index]
    max_y = y[max_index]

    # Find the half maximum value
    half_max = max_y / 2

    # Find the indices where y is greater than or equal to half maximum
    indices = np.where(y >= half_max)[0]

    # Find the left and right indices where y crosses the half maximum
    left_index = indices[0]
    right_index = indices[-1]

    # Interpolate the x-values corresponding to the left and right indices
    left_x = np.interp(half_max, y[left_index:right_index + 1][::-1], x[left_index:right_index + 1][::-1])
    right_x = np.interp(half_max, y[left_index:right_index + 1], x[left_index:right_index + 1])

    # Calculate the Full Width Half Maximum (FWHM) value
    fwhm = right_x - left_x

    return fwhm


def open_images(start, end, path, date, roix, show_status=0, image_dim=[1000, 1600]):
    image_matrix = np.zeros([image_dim[0], np.size(np.arange(roix[0], roix[1])), end - start + 1])
    for ind in range(start, end + 1):
        if show_status:
            print(ind)
        file = path + date + '-' + str(int(ind)) + '.bmp'
        im_temp = np.asarray(Image.open(file))
        im_temp = im_temp[:, roix[0]:roix[1]]
        # print(ind-start)
        image_matrix[:, :, ind - start] = im_temp
    return image_matrix


def open_autologfile(filename):
    lines = np.loadtxt(filename, comments='#', dtype=str, delimiter="\t", unpack=False)
    return lines


def subtract_background(images, backgrounds):
    nr_scans = np.shape(backgrounds)[2]
    nr_total = np.shape(images)[2]
    scan_size = nr_total / nr_scans

    new = np.zeros_like(images)

    # print(nr_scans, nr_total, scan_size)

    for i in np.arange(0, nr_scans):
        start = int(i * scan_size)
        end = int(i * scan_size + scan_size)
        for j in np.arange(start, end):
            print(j, i)
            new[:, :, j] = images[:, :, j] - backgrounds[:, :, i]

    return new


def fit_energy_calibration_peaks(prof, prom=2000, roi=[640, 1600], smoothing=21):
    dat = prof
    dat[0: roi[0]] = 0
    calibration = help.savitzky_golay(prof, smoothing, 3)  # window size 51, polynomial order 3
    peak, _ = find_peaks(calibration, prominence=prom)
    return calibration, peak


def shear_image(image_old, val):
    T = np.float32([[1, val / 100, 0], [0, 1, 0]])
    size_T = (image_old.shape[1], image_old.shape[0])
    image_new = cv2.warpAffine(image_old, T, size_T)
    return image_new


def find_best_shear(image_old, roi, region=np.linspace(-8, 2, 50)):
    nr = 20
    res = np.zeros([np.size(region), 1])
    for ind, i in enumerate(region):
        im_new = shear_image(image_old, i)
        prof = np.sum(im_new, 0)
        dat = prof[roi[0]:roi[1]]
        fwhm = help.find_fwhm(np.arange(0, np.size(dat)), dat)
        res[ind] = fwhm
    index = np.argmin(abs(res))
    return region[index]


def redistribute_image(sheared_image, E_axis):
    Jacobian_vect = h * c / (E_axis ** 2)
    Jacobian_vect_norm = Jacobian_vect / np.max(Jacobian_vect)
    Jacobian_mat = np.tile(Jacobian_vect_norm, [np.shape(sheared_image)[0], 1])
    redistributed_image = np.multiply(sheared_image, Jacobian_mat)
    return redistributed_image


def redistribute_profile(sheared_profile, E_axis):
    Jacobian_vect = h * c / (E_axis ** 2)
    Jacobian_vect_norm = Jacobian_vect / np.max(Jacobian_vect)
    redistributed_profile = np.multiply(sheared_profile, Jacobian_vect_norm)
    return redistributed_profile


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


def remove_background_profile(profile):
    f = gaussian_filter1d(profile, 3)
    peaks, _ = find_peaks(-f)
    y_bg = np.interp(np.arange(0, 1600), peaks, f[peaks])
    p_new = profile - y_bg
    p_new[p_new<0] = 0
    return p_new


def treat_image_profiles(image_old, energy_axis, return_correct_E_axis=0):
    sheared_image = shear_image(image_old, -2.5)
    profile = np.sum(sheared_image, 0)
    cleaned_profile = remove_background_profile(profile)
    redistributed_profile = redistribute_profile(cleaned_profile, energy_axis)
    y_axis = np.arange(0, np.shape(sheared_image)[0])
    correct_E_axis = np.arange(energy_axis[0], energy_axis[-1],
                               abs((energy_axis[0] - energy_axis[-1]) / np.shape(redistributed_profile)[0]))
    interp_func = interp1d(energy_axis, np.flip(redistributed_profile), kind='linear')
    profile_new = interp_func(correct_E_axis)

    if return_correct_E_axis:
        return correct_E_axis, profile_new
    else:
        return profile_new
