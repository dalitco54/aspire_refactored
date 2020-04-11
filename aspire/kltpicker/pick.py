#!/usr/bin/env python

import glob
import operator as op
import os
import warnings
from pyfftw import FFTW
from sys import exit
import argparse
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import scipy.special as ssp
from numpy.matlib import repmat
from numpy.polynomial.legendre import leggauss
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from scipy import signal
from scipy.fftpack import fftshift
from scipy.ndimage import uniform_filter, correlate
from tqdm import tqdm
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# Globals:
PERCENT_EIG_FUNC = 0.99
EPS = 10 ** (-2)  # Convergence term for ALS.
PERCENT_EIG_FUNC = 0.99
NUM_QUAD_NYS = 2 ** 10
NUM_QUAD_KER = 2 ** 10
MAX_FUN = 400


# Utils:

def f_trans_2(b):
    """
    2-D FIR filter using frequency transformation.

    Produces the 2-D FIR filter h that corresponds to the 1-D FIR
    filter b using the McClellan transform.
    :param b: 1-D FIR filter.
    :return h: 2-D FIR filter.
    """
    # McClellan transformation:
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8
    n = int((b.size - 1) / 2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2 * b[0:n + 1]
    a[0] = a[0] / 2
    # Use Chebyshev polynomials to compute h:
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = 1
    cols = 1
    h[rows, cols] = h[rows, cols] + a[0] * p0
    p2 = 2 * signal.convolve2d(t, p1)
    p2[2, 2] = p2[2, 2] - p0
    for i in range(2, n + 1):
        rows = p1.shape[0] + 1
        cols = p1.shape[1] + 1
        hh = h
        h = a[i] * p2
        h[1:rows, 1:cols] = h[1:rows, 1:cols] + hh
        p0 = p1
        p1 = p2
        rows += 1
        cols += 1
        p2 = 2 * signal.convolve2d(t, p1)
        p2[2:rows, 2:cols] = p2[2:rows, 2:cols] - p0
    h = np.rot90(h, k=2)
    return h


def radial_avg(z, m):
    """
    Radially average 2-D square matrix z into m bins.

    Computes the average along the radius of a unit circle
    inscribed in the square matrix z. The average is computed in m bins. The radial average is not computed beyond
    the unit circle, in the corners of the matrix z. The radial average is returned in zr and the mid-points of the
    m bins are returned in vector R.
    :param z: 2-D square matrix.
    :param m: Number of bins.
    :return zr: Radial average of z.
    :return R: Mid-points of the bins.
    """
    N = z.shape[1]
    X, Y = np.meshgrid(np.arange(N) * 2 / (N - 1) - 1, np.arange(N) * 2 / (N - 1) - 1)
    r = np.sqrt(np.square(X) + np.square(Y))
    dr = 1 / (m - 1)
    rbins = np.linspace(-dr / 2, 1 + dr / 2, m + 1, endpoint=True)
    R = (rbins[0:-1] + rbins[1:]) / 2
    zr = np.zeros(m)
    for j in range(m - 1):
        bins = np.where(np.logical_and(r >= rbins[j], r < rbins[j + 1]))
        n = np.count_nonzero(np.logical_and(r >= rbins[j], r < rbins[j + 1]))
        if n != 0:
            zr[j] = sum(z[bins]) / n
        else:
            zr[j] = np.nan
    bins = np.where(np.logical_and(r >= rbins[m - 1], r <= 1))
    n = np.count_nonzero(np.logical_and(r >= rbins[m - 1], r <= 1))
    if n != 0:
        zr[m - 1] = sum(z[bins]) / n
    else:
        zr[m - 1] = np.nan
    return zr, R


def stdfilter(a, nhood):
    """Local standard deviation of image."""
    c1 = uniform_filter(a, nhood, mode='reflect')
    c2 = uniform_filter(a * a, nhood, mode='reflect')
    return np.sqrt(c2 - c1 * c1) * np.sqrt(nhood ** 2. / (nhood ** 2 - 1))


def als_find_min(sreal, eps, max_iter):
    """
    ALS method for RPSD factorization.

    Approximate Clean and Noise PSD and the particle location vector alpha.
    :param sreal: PSD matrix to be factorized
    :param eps: Convergence term
    :param max_iter: Maximum iterations
    :return approx_clean_psd: Approximated clean PSD
    :return approx_noise_psd: Approximated noise PSD
    :return alpha_approx: Particle location vector alpha.
    :return stop_par: Stop algorithm if an error occurred.
    """
    sz = sreal.shape
    patch_num = sz[1]
    One = np.ones(patch_num)
    s_norm_inf = np.apply_along_axis(lambda x: max(np.abs(x)), 0, sreal)
    max_col = np.argmax(s_norm_inf)
    min_col = np.argmin(s_norm_inf)
    clean_sig_tmp = np.abs(sreal[:, max_col] - sreal[:, min_col])
    s_norm_1 = np.apply_along_axis(lambda x: sum(np.abs(x)), 0, sreal)
    min_col = np.argmin(s_norm_1)
    noise_sig_tmp = np.abs(sreal[:, min_col])
    s = sreal - np.outer(noise_sig_tmp, One)
    alpha_tmp = (np.dot(clean_sig_tmp, s)) / np.sum(clean_sig_tmp ** 2)
    alpha_tmp = alpha_tmp.clip(min=0, max=1)
    stop_par = 0
    cnt = 1
    while stop_par == 0:
        if np.linalg.norm(alpha_tmp, 1) == 0:
            alpha_tmp = np.random.random(alpha_tmp.size)
        approx_clean_psd = np.dot(s, alpha_tmp) / sum(alpha_tmp ** 2)
        approx_clean_psd = approx_clean_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_clean_psd, alpha_tmp)
        approx_noise_psd = np.dot(s, np.ones(patch_num)) / patch_num
        approx_noise_psd = approx_noise_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_noise_psd, One)
        if np.linalg.norm(approx_clean_psd, 1) == 0:
            approx_clean_psd = np.random.random(approx_clean_psd.size)
        alpha_approx = np.dot(approx_clean_psd, s) / sum(approx_clean_psd ** 2)
        alpha_approx = alpha_approx.clip(min=0, max=1)
        if np.linalg.norm(noise_sig_tmp - approx_noise_psd) / np.linalg.norm(approx_noise_psd) < eps:
            if np.linalg.norm(clean_sig_tmp - approx_clean_psd) / np.linalg.norm(approx_clean_psd) < eps:
                if np.linalg.norm(alpha_approx - alpha_tmp) / np.linalg.norm(alpha_approx) < eps:
                    break
        noise_sig_tmp = approx_noise_psd
        alpha_tmp = alpha_approx
        clean_sig_tmp = approx_clean_psd
        cnt += 1
        if cnt > max_iter:
            stop_par = 1
            break
    return approx_clean_psd, approx_noise_psd, alpha_approx, stop_par


def trig_interpolation(x, y, xq):
    n = x.size
    h = 2 / n
    scale = (x[1] - x[0]) / h
    xs = x / scale
    xi = xq / scale
    p = np.zeros(xi.size)
    for k in range(n):
        if n % 2 == 1:
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.sin(np.pi * (xi - xs[k]) / 2))
        else:
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.tan(np.pi * (xi - xs[k]) / 2))
        a[(xi - xs[k]) == 0] = 1
        p = p + y[k] * a
    return p


def picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size):
    idx_row = np.arange(log_test_n.shape[0])
    idx_col = np.arange(log_test_n.shape[1])
    [col_idx, row_idx] = np.meshgrid(idx_col, idx_row)
    r_del = np.floor(kltpicker.patch_size_pick_box)
    shape = log_test_n.shape
    scoring_mat = log_test_n
    if kltpicker.num_of_particles == -1:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, np.iinfo(np.int32(10)).max, op.gt,
                                                  kltpicker.threshold + 1, kltpicker.threshold,
                                                  kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf,
                                                  kltpicker.patch_size_pick_box)
    else:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_particles, op.gt,
                                                  kltpicker.threshold + 1, kltpicker.threshold,
                                                  kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf,
                                                  kltpicker.patch_size_pick_box)
    if kltpicker.num_of_noise_images != 0:
        num_picked_noise = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_noise_images, op.lt,
                                              kltpicker.threshold - 1, kltpicker.threshold, kltpicker.patch_size_func,
                                              row_idx, col_idx, kltpicker.output_noise, mrc_name,
                                              kltpicker.mgscale, mg_big_size, np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_noise = 0
    return num_picked_particles, num_picked_noise


def write_output_files(scoring_mat, shape, r_del, max_iter, oper, oper_param, threshold, patch_size_func, row_idx,
                       col_idx, output_path, mrc_name, mgscale, mg_big_size, replace_param, patch_size_pick_box):
    num_picked = 0
    box_path = output_path + '/box'
    star_path = output_path + '/star'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(box_path):
        os.mkdir(box_path)
    if not os.path.isdir(star_path):
        os.mkdir(star_path)
    box_file = open("%s/%s.box" % (box_path, mrc_name.replace('.mrc', '')), 'w')
    star_file = open("%s/%s.star" % (star_path, mrc_name.replace('.mrc', '')), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    iter_pick = 0
    log_max = np.max(scoring_mat)
    while iter_pick <= max_iter and oper(oper_param, threshold):
        max_index = np.argmax(scoring_mat.transpose().flatten())
        oper_param = scoring_mat.transpose().flatten()[max_index]
        if not oper(oper_param, threshold):
            break
        else:
            [index_col, index_row] = np.unravel_index(max_index, shape)
            ind_row_patch = (index_row - 1) + patch_size_func
            ind_col_patch = (index_col - 1) + patch_size_func
            row_idx_b = row_idx - index_row
            col_idx_b = col_idx - index_col
            rsquare = row_idx_b ** 2 + col_idx_b ** 2
            scoring_mat[rsquare <= (r_del ** 2)] = replace_param
            box_file.write(
                '%i\t%i\t%i\t%i\n' % ((1 / mgscale) * (ind_col_patch + 1 - np.floor(patch_size_pick_box / 2)),
                                      (mg_big_size[0] + 1) - (1 / mgscale) * (
                                                  ind_row_patch + 1 + np.floor(patch_size_pick_box / 2)),
                                      (1 / mgscale) * patch_size_pick_box, (1 / mgscale) * patch_size_pick_box))
            star_file.write('%i\t%i\t%f\n' % (
            (1 / mgscale) * (ind_col_patch + 1), (mg_big_size[0] + 1) - ((1 / mgscale) * (ind_row_patch + 1)),
            oper_param / log_max))
            iter_pick += 1
            num_picked += 1
    star_file.close()
    box_file.close()
    return num_picked


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Input directory.')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('-s', '--particle_size', help='Expected size of particles in pixels.', default=300, type=int)
    parser.add_argument('--num_of_particles',
                        help='Number of particles to pick per micrograph. If set to -1 will pick all particles.',
                        default=-1, type=int)
    parser.add_argument('--num_of_noise_images', help='Number of noise images to pick per micrograph.',
                        default=0, type=int)
    parser.add_argument('--max_iter', help='Maximum number of iterations.', default=6 * (10 ** 4), type=int)
    parser.add_argument('--gpu_use', action='store_true', default=False)
    parser.add_argument('--max_order', help='Maximum order of eigenfunction.', default=100, type=int)
    parser.add_argument('--percent_eigen_func', help='', default=0.99, type=float)
    parser.add_argument('--max_functions', help='', default=400, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose.', default=False)
    parser.add_argument('--threshold', help='Threshold for the picking', default=0, type=float)
    parser.add_argument('--show_figures', action='store_true', help='Show figures', default=False)
    parser.add_argument('--preprocess', action='store_false', help='Do not run preprocessing.', default=True)
    args = parser.parse_args()
    return args


def crop(x, out_shape):
    """

    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def downsample(stack, n, mask=None, stack_in_fourier=False):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the *first* dimension of 'img' is interpreted as the index of
        each image in the stack. The size argument side is an integer, the size of the
        output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
        msk = fuzzymask(n,2,.45*n,.05*n)
        out = downsample(img, n, 0, msk)
        The size of the mask must be the size of output. The optional fx output
        argument is the padded or cropped, masked, FT of in, with zero
        frequency at the origin.
    """

    size_in = np.square(stack.shape[1])
    size_out = np.square(n)
    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    output = np.zeros((num_images, n, n), dtype='float64')
    images_batches = np.array_split(np.arange(num_images), 500)
    for batch in images_batches:
        if batch.size:
            curr_batch = np.array(stack[batch])
            curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
            fx = crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
            output[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
    return output


def cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


def fast_cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = fft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(ifft2(np.transpose(np.fft.ifftshift(x)))))

    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y

    else:
        raise ValueError("x must be 2D or 3D")


def lgwt(n, a, b):
    """
    Get n leggauss points in interval [a, b]

    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :returns SamplePoints(x, w): sample points, weight
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return x, w


def cryo_epsds(imstack, samples_idx, max_d):
    p = imstack.shape[0]
    if max_d >= p:
        max_d = p - 1
        print('max_d too large. Setting max_d to {}'.format(max_d))

    r, x, _ = cryo_epsdr(imstack, samples_idx, max_d)

    r2 = np.zeros((2 * p - 1, 2 * p - 1))
    dsquare = np.square(x)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                r2[i + p - 1, j + p - 1] = r[idx - 1]

    w = gwindow(p, max_d)
    p2 = fast_cfft2(r2 * w)

    p2 = p2.real

    e = 0
    for i in range(imstack.shape[2]):
        im = imstack[:, :, i]
        e += np.sum(np.square(im[samples_idx] - np.mean(im[samples_idx])))

    mean_e = e / (len(samples_idx[0]) * imstack.shape[2])
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    neg_idx = np.where(p2 < 0)
    p2[neg_idx] = 0
    return p2, r, r2, x


def cryo_epsdr(vol, samples_idx, max_d):
    p = vol.shape[0]
    k = vol.shape[2]
    i, j = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = np.square(i) + np.square(j)
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))

    corrs = np.zeros(len(dsquare))
    corr_count = np.zeros(len(dsquare))
    x = np.sqrt(dsquare)

    dist_map = np.zeros(dists.shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx

    dist_map = dist_map.astype('int') - 1
    valid_dists = np.where(dist_map != -1)

    mask = np.zeros((p, p))
    mask[samples_idx] = 1
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = fft2(tmp)
    c = ifft2(ftmp * np.conj(ftmp))
    c = c[:max_d + 1, :max_d + 1]
    c = np.round(c.real).astype('int')

    r = np.zeros(len(corrs))

    # optimized version
    vol = vol.transpose((2, 0, 1)).copy()
    input_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    a_fft2 = FFTW(input_fft2, output_fft2, axes=(0, 1), direction='FFTW_FORWARD', flags=flags)
    a_ifft2 = FFTW(input_ifft2, output_ifft2, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags)
    sum_s = np.zeros(output_ifft2.shape, output_ifft2.dtype)
    sum_c = c * vol.shape[0]
    for i in range(k):
        proj = vol[i]

        input_fft2[samples_idx] = proj[samples_idx]
        a_fft2()
        np.multiply(output_fft2, np.conj(output_fft2), out=input_ifft2)
        a_ifft2()
        sum_s += output_ifft2

    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]

    idx = np.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]

    idx = np.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt


def gwindow(p, max_d):
    x, y = np.meshgrid(np.arange(-(p - 1), p), np.arange(-(p - 1), p))
    alpha = 3.0
    w = np.exp(-alpha * (np.square(x) + np.square(y)) / (2 * max_d ** 2))
    return w


def bsearch(x, lower_bound, upper_bound):
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(np.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw - 1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(np.ceil((upper_idx_a + upper_idx_b) / 2))
        if x[up - 1] <= upper_bound:
            upper_idx_a = up
        else:
            upper_idx_b = up
            if lower_idx_a < up < lower_idx_b:
                lower_idx_b = up

    if x[lower_idx_a - 1] >= lower_bound:
        lower_idx = lower_idx_a
    else:
        lower_idx = lower_idx_b
    if x[upper_idx_b - 1] <= upper_bound:
        upper_idx = upper_idx_b
    else:
        upper_idx = upper_idx_a

    if upper_idx < lower_idx:
        return None, None

    return lower_idx, upper_idx


def cryo_prewhiten(proj, noise_response, rel_threshold=None):
    """
    Pre-whiten a stack of projections using the power spectrum of the noise.


    :param proj: stack of images/projections
    :param noise_response: 2d image with the power spectrum of the noise. If all
                           images are to be whitened with respect to the same power spectrum,
                           this is a single image. If each image is to be whitened with respect
                           to a different power spectrum, this is a three-dimensional array with
                           the same number of 2d slices as the stack of images.

    :param rel_threshold: The relative threshold used to determine which frequencies
                          to whiten and which to set to zero. If empty (the default)
                          all filter values less than 100*eps(class(proj)) are
                          zeroed out, while otherwise, all filter values less than
                          threshold times the maximum filter value for each filter
                          is set to zero.

    :return: Pre-whitened stack of images.
    """

    delta = np.finfo(proj.dtype).eps

    L1, L2, num_images = proj.shape
    l1 = L1 // 2
    l2 = L2 // 2
    K1, K2 = noise_response.shape
    k1 = int(np.ceil(K1 / 2))
    k2 = int(np.ceil(K2 / 2))

    filter_var = np.sqrt(noise_response)
    filter_var /= np.linalg.norm(filter_var)

    filter_var = (filter_var + np.flipud(filter_var)) / 2
    filter_var = (filter_var + np.fliplr(filter_var)) / 2

    if rel_threshold is None:
        nzidx = np.where(filter_var > 100 * delta)
    else:
        raise NotImplementedError('not implemented for rel_threshold != None')

    fnz = filter_var[nzidx]
    one_over_fnz = 1 / fnz

    # matrix with 1/fnz in nzidx, 0 elsewhere
    one_over_fnz_as_mat = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    one_over_fnz_as_mat[nzidx] += one_over_fnz
    pp = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    p2 = np.zeros((num_images, L1, L2), dtype='complex128')
    proj = proj.transpose((2, 0, 1)).copy()

    row_start_idx = k1 - l1 - 1
    row_end_idx = k1 + l1
    col_start_idx = k2 - l2 - 1
    col_end_idx = k2 + l2

    if L1 % 2 == 0 and L2 % 2 == 0:
        row_end_idx -= 1
        col_end_idx -= 1

    for i in range(num_images):
        pp[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = proj[i]
        fp = fast_cfft2(pp)
        fp *= one_over_fnz_as_mat
        pp2 = fast_icfft2(fp)
        p2[i] = np.real(pp2[row_start_idx:row_end_idx, col_start_idx:col_end_idx])

    # change back to x,y,z convention
    proj = p2.real.transpose((1, 2, 0)).copy()
    return proj, filter_var, nzidx


# Micrograph:

class Micrograph:
    """
    Object that contains all the variables and methods needed for the particle picking.

    ...
    Attributes
    ----------
    micrograph : np.ndarray
        Micrograph after downsampling.
    micrograph_pic : np.ndarray
        Original micrograph data.
    mc_size : tuple
        Size of micrograph after downsampling.
    mg_big_size : tuple
        Size of original micrograph.
    noise_mc : np.ndarray
    approx_clean_psd : np.ndarray
        Approximated clean PSD.
    approx_noise_psd : np.ndarray
        Approximated noise PSD.
    approx_noise_var : float
        Approximate noise variance.
    r : np.ndarray
    stop_par : int
        Flag to stop algorithm if maximal number of iterations in PSD approximation was exceeded.
    psd : np.ndarray
    eig_func : np.ndarray
        Eigenfunctions.
    eig_val : np.ndarray
        Eigenvalues.
    num_of_func : int
        Number of eigenfunctions.
    mrc_name : str
        Name of .mrc file.

    Methods
    -------
    cutoff_filter(self, patch_size)
        Radial bandpass filter.
    estimate_rpsd(self, patch_size, max_iter)
    prewhiten_micrograph(self)
    construct_klt_templates(self, kltpicker)
        Constructing the KLTpicker templates as the eigenfunctions of a given kernel.
    detect_particles(self, kltpicker)
        Construct the scoring matrix and then use the picking_from_scoring_mat function to pick particles and noise
        images.
    picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size)
        Pick particles and noise images from the scoring matrix.
    """

    def __init__(self, micrograph, micrograph_pic, mc_size, mrc_name, mg_big_size):
        self.micrograph = micrograph
        self.micrograph_pic = micrograph_pic
        self.mc_size = mc_size
        self.mg_big_size = mg_big_size
        self.noise_mc = 0
        self.approx_clean_psd = 0
        self.approx_noise_psd = 0
        self.approx_noise_var = 0
        self.r = 0
        self.stop_par = 0
        self.psd = 0
        self.eig_func = 0
        self.eig_val = 0
        self.num_of_func = 0
        self.mrc_name = mrc_name

    def cutoff_filter(self, patch_size):
        """Radial bandpass filter."""
        bandpass1d = signal.firwin(int(patch_size), np.array([0.05, 0.95]), pass_zero=False)
        bandpass2d = f_trans_2(bandpass1d)
        micrograph = correlate(self.micrograph, bandpass2d, mode='constant')
        self.noise_mc = micrograph

    def estimate_rpsd(self, patch_size, max_iter):
        """Approximate clean and noise RPSD per micrograph."""
        micro_size = self.noise_mc.shape[0]
        m = np.floor(micro_size / patch_size)
        M = (m ** 2).astype(int)
        L = int(patch_size)  # np.ceil(np.sqrt(2)*(patch_size-1)+1).astype(int)
        s = np.zeros((L, M))
        num_quads = 2 ** 9
        quad, nodes = lgwt(num_quads, -np.pi, np.pi)
        x = repmat(quad, num_quads, 1)
        y = x.transpose()
        rho_mat = np.sqrt(x ** 2 + y ** 2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        r_tmp = np.zeros((L, 1))
        for k in tqdm(range(M)):
            row = np.ceil((k + 1) / m).astype(int)
            col = (k + 1 - (row - 1) * m).astype(int)
            noisemc_block = self.noise_mc[(row - 1) * patch_size.astype(int):row * patch_size.astype(int),
                            (col - 1) * patch_size.astype(int): col * patch_size.astype(int)]
            noisemc_block = noisemc_block - np.mean(noisemc_block)
            psd_block = cryo_epsds(noisemc_block[:, :, np.newaxis],
                                   np.where(np.zeros((int(patch_size), int(patch_size))) == 0),
                                   np.floor(0.3 * patch_size).astype(int))
            psd_block = psd_block[0]
            if np.count_nonzero(np.isnan(psd_block)) != 0:
                print("got NaN")
            [r_block, r] = radial_avg(psd_block, L)
            block_var = np.var(noisemc_block.transpose().flatten(), ddof=1)
            psd_rad = np.abs(trig_interpolation(r * np.pi, r_block, rho_samp))
            psd_mat = np.reshape(psd_rad[idx], [num_quads, num_quads])
            var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, psd_mat, nodes.transpose()])
            scaling_psd = block_var / var_psd
            r_block = scaling_psd * r_block
            s[:, k] = r_block
            if k == 1:
                r_tmp = r
        # find min arg using ALS:
        r = r_tmp
        approx_clean_psd, approx_noise_psd, alpha, stop_par = als_find_min(s, EPS, max_iter)
        std_mat = stdfilter(self.noise_mc, patch_size)
        var_mat = std_mat ** 2
        cut = int((patch_size - 1) / 2 + 1)
        var_mat = var_mat[cut - 1:-cut, cut - 1:-cut]
        var_vec = var_mat.transpose().flatten()
        var_vec.sort()
        j = np.floor(0.25 * var_vec.size).astype('int')
        noise_var_approx = np.mean(var_vec[0:j])
        num_of_quad = 2 ** 12
        quad, nodes = lgwt(num_of_quad, -np.pi, np.pi)
        y = repmat(quad, num_of_quad, 1)
        x = y.transpose()
        rho_mat = np.sqrt(x ** 2 + y ** 2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        clean_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_clean_psd, rho_samp))
        noise_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_noise_psd, rho_samp))
        clean_psd_mat = np.reshape(clean_psd_nodes[idx], (num_of_quad, num_of_quad))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], (num_of_quad, num_of_quad))
        scaling_psd_approx = (np.linalg.multi_dot([nodes, noise_psd_mat, nodes.transpose()]) - (
                    4 * np.pi ** 2) * noise_var_approx) / np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        noise_psd_approx_sigma = approx_noise_psd - scaling_psd_approx * approx_clean_psd
        noise_psd_approx_sigma = noise_psd_approx_sigma.clip(min=0, max=None)
        s_mean = np.mean(s, 1)
        s_mean_psd_nodes = np.abs(trig_interpolation(r * np.pi, s_mean, rho_samp))
        s_mean_psd_mat = np.reshape(s_mean_psd_nodes[idx], (num_of_quad, num_of_quad))
        s_mean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, s_mean_psd_mat, nodes.transpose()])
        clean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        clean_var = s_mean_var_psd - noise_var_approx
        approx_scaling = clean_var / clean_var_psd
        self.approx_clean_psd = approx_scaling * approx_clean_psd
        self.approx_noise_psd = noise_psd_approx_sigma
        self.approx_noise_var = noise_var_approx
        self.r = r
        self.stop_par = stop_par

    def prewhiten_micrograph(self):
        r = np.floor((self.mc_size[1] - 1) / 2).astype('int')
        c = np.floor((self.micrograph.shape[0] - 1) / 2).astype('int')
        col = np.arange(-c, c + 1) * np.pi / c
        row = np.arange(-r, r + 1) * np.pi / r
        Row, Col = np.meshgrid(row, col)
        rad_mat = np.sqrt(Col ** 2 + Row ** 2)
        rad_samp, idx = np.unique(rad_mat, return_inverse=True)
        rad_samp_tmp = rad_samp[rad_samp < np.max(self.r * np.pi)]
        noise_psd_nodes = np.abs(trig_interpolation(self.r * np.pi, self.approx_noise_psd, rad_samp_tmp))
        noise_psd_nodes = np.pad(noise_psd_nodes, (0, rad_samp.size - noise_psd_nodes.size), 'constant',
                                 constant_values=noise_psd_nodes[-1])
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], [col.size, row.size])
        noise_mc_prewhite = cryo_prewhiten(self.noise_mc[:, :, np.newaxis], noise_psd_mat)
        noise_mc_prewhite = noise_mc_prewhite[0][:, :, 0]
        noise_mc_prewhite = noise_mc_prewhite - np.mean(noise_mc_prewhite)
        noise_mc_prewhite = noise_mc_prewhite / np.linalg.norm(noise_mc_prewhite, 'fro')
        self.noise_mc = noise_mc_prewhite

    def construct_klt_templates(self, kltpicker):
        """Constructing the KLTpicker templates as the eigenfunctions of a given kernel."""
        eig_func_tot = np.zeros((NUM_QUAD_NYS, NUM_QUAD_NYS, kltpicker.max_order))
        eig_val_tot = np.zeros((NUM_QUAD_NYS, kltpicker.max_order))
        sqrt_rr = np.sqrt(kltpicker.r_r)
        d_rho_psd_quad_ker = np.diag(kltpicker.rho) * np.diag(self.psd) * np.diag(kltpicker.quad_ker)
        sqrt_diag_quad_nys = np.sqrt(np.diag(kltpicker.quad_nys))
        for n in tqdm(range(kltpicker.max_order)):
            h_nodes = sqrt_rr * np.linalg.multi_dot([kltpicker.j_r_rho[:, :, n], d_rho_psd_quad_ker,
                                       kltpicker.j_r_rho[:, :, n].transpose()])
            tmp = np.linalg.multi_dot([sqrt_diag_quad_nys, h_nodes, sqrt_diag_quad_nys.transpose()])
            eig_vals, eig_funcs = np.linalg.eig(tmp)
            eig_vals = np.real(eig_vals)
            sort_idx = np.argsort(eig_vals)
            sort_idx = sort_idx[::-1]  # Descending.
            eig_vals = eig_vals[sort_idx]
            eig_funcs = eig_funcs[:, sort_idx]
            eig_vals = np.where(np.abs(eig_vals) < np.spacing(1), 0, eig_vals)
            eig_funcs[:, eig_vals == 0] = 0
            eig_func_tot[:, :, n] = eig_funcs
            eig_val_tot[:, n] = eig_vals
        r_idx = np.arange(0, NUM_QUAD_NYS)
        c_idx = np.arange(0, kltpicker.max_order)
        r_idx = repmat(r_idx, 1, kltpicker.max_order)
        c_idx = repmat(c_idx, NUM_QUAD_NYS, 1)
        eig_val_tot = eig_val_tot.transpose().flatten()
        r_idx = r_idx.transpose().flatten()
        c_idx = c_idx.transpose().flatten()
        sort_idx = np.argsort(eig_val_tot)
        sort_idx = sort_idx[::-1]
        eig_val_tot = eig_val_tot[sort_idx]
        r_idx = r_idx[sort_idx]
        c_idx = c_idx[sort_idx]
        sum_of_eig = np.sum(eig_val_tot)
        cum_sum_eig_val = np.cumsum(eig_val_tot / sum_of_eig)
        last_eig_idx = (cum_sum_eig_val > PERCENT_EIG_FUNC).argmax() + 1
        eig_val = np.zeros((1, 2 * last_eig_idx))
        eig_func = np.zeros((kltpicker.rsamp_length, 2 * last_eig_idx))
        count = 0
        for i in range(last_eig_idx):
            order = c_idx[i]
            idx_of_eig = r_idx[i]
            h_samp = np.sqrt(kltpicker.rsamp_r) * np.linalg.multi_dot([kltpicker.j_samp[:, :, order],
                                                                np.diag(kltpicker.rho * self.psd * kltpicker.quad_ker),
                                                         kltpicker.j_r_rho[:, :, order].transpose()])
            v_correct = (1 / np.sqrt(kltpicker.quad_nys)) * eig_func_tot[:, idx_of_eig, order]
            v_nys = np.dot(h_samp, (kltpicker.quad_nys * v_correct)) / eig_val_tot[i]
            if order == 0:
                eig_func[:, count] = (1 / np.sqrt(2 * np.pi)) * v_nys
                eig_val[0, count] = eig_val_tot[i]
                count += 1
            else:
                eig_func[:, count] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.cosine[:, order]
                eig_val[0, count] = eig_val_tot[i]
                count += 1
                eig_func[:, count] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.sine[:, order]
                eig_val[0, count] = eig_val_tot[i]
                count += 1
        eig_val = eig_val[eig_val > 0]
        eig_func = eig_func[:, 0:len(eig_val)]
        if eig_func.shape[1] < MAX_FUN:
            num_of_fun = eig_func.shape[1]
        else:
            num_of_fun = MAX_FUN
        self.eig_func = eig_func
        self.eig_val = eig_val
        self.num_of_func = num_of_fun

    def detect_particles(self, kltpicker):
        """
        Construct the scoring matrix and then use the picking_from_scoring_mat function to pick particles and noise
        images.
        """
        eig_func_stat = self.eig_func[:, 0:self.num_of_func]
        eig_val_stat = self.eig_val[0:self.num_of_func]
        for i in range(self.num_of_func):
            tmp_func = np.reshape(eig_func_stat[:, i], (kltpicker.patch_size_func, kltpicker.patch_size_func))
            tmp_func[kltpicker.rad_mat > np.floor((kltpicker.patch_size_func - 1) / 2)] = 0
            eig_func_stat[:, i] = tmp_func.flatten()
        [q, r] = np.linalg.qr(eig_func_stat, 'complete')
        r = r[0:self.num_of_func, 0:self.num_of_func]
        kappa = np.linalg.multi_dot([r, np.diag(eig_val_stat), r.transpose()]) + (
                    self.approx_noise_var * np.eye(self.num_of_func))
        kappa_inv = np.linalg.inv(kappa)
        t_mat = (1 / self.approx_noise_var) * np.eye(self.num_of_func) - kappa_inv
        mu = np.linalg.slogdet((1 / self.approx_noise_var) * kappa)[1]
        last_block_row = self.mc_size[0] - kltpicker.patch_size_func + 1
        last_block_col = self.mc_size[1] - kltpicker.patch_size_func + 1
        num_of_patch_row = last_block_row
        num_of_patch_col = last_block_col
        v = np.zeros((num_of_patch_row, num_of_patch_col, self.num_of_func))
        cnt = 0
        for i in tqdm(range(self.num_of_func)):
            cnt += 1
            q_tmp = np.reshape(q[:, i], (kltpicker.patch_size_func, kltpicker.patch_size_func)).transpose()
            q_tmp = q_tmp - np.mean(q_tmp)
            q_tmp = np.flip(q_tmp, 1)
            if kltpicker.gpu_use == 1:
                pass
                # noiseMcGpu = gpuArray(single(noiseMc))
                # v_tmp = conv2(noiseMcGpu, q_tmp, 'valid')
                # v(:,:, i) = single(gather(v_tmp))
            else:
                v_tmp = signal.fftconvolve(self.noise_mc, q_tmp, 'valid')
                v[:, :, i] = v_tmp.astype('single')
        log_test_mat = np.zeros((num_of_patch_row, num_of_patch_col))
        cnt = 0
        for j in range(num_of_patch_col):
            cnt += 1
            vc = np.reshape(v[:, j, :], (num_of_patch_row, self.num_of_func))
            log_test_mat[:, j] = np.sum(np.dot(vc, t_mat) * vc, 1) - mu
        if kltpicker.gpu_use == 1:
            pass
            # neigh = gpuArray(ones(kltpicker.patch_sz_func))
            # logTestN = gather(conv2(logTestMat, neigh, 'valid'))
        else:
            neigh = np.ones((kltpicker.patch_size_func, kltpicker.patch_size_func))
            log_test_n = signal.fftconvolve(log_test_mat, neigh, 'valid')
        [num_picked_particles, num_picked_noise] = picking_from_scoring_mat(log_test_n, self.mrc_name, kltpicker,
                                                                            self.mg_big_size)
        return num_picked_particles, num_picked_noise


# KLTpicker:

class Picker:
    """
    KLTpicker object that holds all variables that are used in the computations.

    ...
    Attributes
    ----------
    particle_size : float
        Size of particles to look for in micrographs.
    input_dir : str
        Directory from which to read .mrc files.
    output_dir : str
        Output directory in which to write results.
    gpu_use : bool
        Optional - whether to use GPU or not.
    mgscale : float
        Scaling parameter.
    max_order : int
        Maximal order of eigenfunctions.
    micrographs : np.ndarray
        Array of 2-D micrographs.
    patch_size_pick_box : int
        Particle box size to use.
    num_of_particles : int
        Number of particles to pick per micrograph.
    num_of_noise_images : int
        Number of noise images.
    threshold : float
        Threshold for the picking.
    patch_size : int
        Approximate size of particle after downsampling.
    patch_size_func : int
        Size of disc for computing the eigenfunctions.
    max_iter : int
        Maximal number of iterations for PSD approximation.
    rsamp_length : int
    rad_mat : np.ndarray
    quad_ker : np.ndarray
    quad_nys : np.ndarray
    rho : np.ndarray
    j_r_rho : np.ndarray
    j_samp : np.ndarray
    cosine : np.ndarray
    sine : np.ndarray
    rsamp_r : np.ndarray
    r_r : np.ndarray


    Methods
    -------
    preprocess()
        Initializes parameters needed for the computation.
    get_micrographs()
        Reads .mrc files, downsamples them and adds them to the KLTpicker object.
    """

    def __init__(self, args):
        self.particle_size = args.particle_size
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.output_noise = '%s/PickedNoise_ParticleSize_%d' % (args.output_dir, args.particle_size)
        self.output_particles = '%s/PickedParticles_ParticleSize_%d' % (args.output_dir, args.particle_size)
        self.gpu_use = args.gpu_use
        self.mgscale = 100 / args.particle_size
        self.max_order = args.max_order
        self.micrographs = np.array([])
        self.quad_ker = 0
        self.quad_nys = 0
        self.rho = 0
        self.j_r_rho = 0
        self.j_samp = 0
        self.cosine = 0
        self.sine = 0
        self.rsamp_r = 0
        self.r_r = 0
        self.patch_size_pick_box = np.floor(self.mgscale * args.particle_size)
        self.num_of_particles = args.num_of_particles
        self.num_of_noise_images = args.num_of_noise_images
        self.threshold = args.threshold
        self.show_figures = 0  # args.show_figures
        patch_size = np.floor(0.8 * self.mgscale * args.particle_size)  # need to put the 0.8 somewhere else.
        if np.mod(patch_size, 2) == 0:
            patch_size -= 1
        self.patch_size = patch_size
        patch_size_function = np.floor(0.4 * self.mgscale * args.particle_size)  # need to put the 0.4 somewhere else.
        if np.mod(patch_size_function, 2) == 0:
            patch_size_function -= 1
        self.patch_size_func = int(patch_size_function)
        self.max_iter = args.max_iter
        self.rsamp_length = 0
        self.rad_mat = 0

    def preprocess(self):
        """Initializes parameters."""
        radmax = np.floor((self.patch_size_func - 1) / 2)
        x = np.arange(-radmax, radmax + 1, 1).astype('float64')
        X, Y = np.meshgrid(x, x)
        rad_mat = np.sqrt(np.square(X) + np.square(Y))
        rsamp = rad_mat.transpose().flatten()
        self.rsamp_length = rsamp.size
        theta = np.arctan2(Y, X).transpose().flatten()
        rho, quad_ker = lgwt(NUM_QUAD_KER, 0, np.pi)
        rho = np.flipud(rho.astype('float64'))
        quad_ker = np.flipud(quad_ker.astype('float64'))
        r, quad_nys = lgwt(NUM_QUAD_NYS, 0, radmax)
        r = np.flipud(r.astype('float64'))
        quad_nys = np.flipud(quad_nys.astype('float64'))
        r_r = np.outer(r, r)
        r_rho = np.outer(r, rho)
        rsamp_r = np.outer(np.ones(len(rsamp)), r)
        rsamp_rho = np.outer(rsamp, rho)
        j_r_rho = np.zeros([NUM_QUAD_KER, NUM_QUAD_NYS, self.max_order]).astype('float64')
        j_samp = np.zeros([len(rsamp), NUM_QUAD_NYS, self.max_order]).astype('float64')
        cosine = np.zeros([len(theta), self.max_order]).astype('float64')
        sine = np.zeros([len(theta), self.max_order]).astype('float64')
        pool = Pool()
        res_j_r_rho = pool.starmap(ssp.jv, [(n, r_rho) for n in range(self.max_order)])
        res_j_samp = pool.starmap(ssp.jv, [(n, rsamp_rho) for n in range(self.max_order)])
        res_cosine = pool.map(np.cos, [n * theta for n in range(self.max_order)])
        res_sine = pool.map(np.sin, [n * theta for n in range(self.max_order)])
        pool.close()
        pool.join()
        j_r_rho = np.squeeze(res_j_r_rho).transpose((1, 2, 0))
        j_samp = np.squeeze(res_j_samp).transpose((1, 2, 0))
        cosine = np.squeeze(res_cosine).transpose()
        sine = np.squeeze(res_sine).transpose()
        cosine[:, 0] = 0
        self.quad_ker = quad_ker
        self.quad_nys = quad_nys
        self.rho = rho
        self.j_r_rho = j_r_rho
        self.j_samp = j_samp
        self.cosine = cosine
        self.sine = sine
        self.rsamp_r = rsamp_r
        self.r_r = r_r
        self.rad_mat = rad_mat

    def get_micrographs(self):
        """Reads .mrc files, downsamples them and adds them to the Picker object."""
        micrographs = []
        mrc_files = glob.glob("%s/*.mrc" % self.input_dir)
        for mrc_file in mrc_files:
            mrc = mrcfile.open(mrc_file)
            mrc_data = mrc.data.astype('float64').transpose()
            mrc.close()
            mrc_size = mrc_data.shape
            mrc_data = np.rot90(mrc_data)
            data = downsample(mrc_data[np.newaxis, :, :], int(np.floor(self.mgscale * mrc_size[0])))[0]
            if np.mod(data.shape[0], 2) == 0:  # Odd size is needed.
                data = data[0:-1, :]
            if np.mod(data.shape[1], 2) == 0:  # Odd size is needed.
                data = data[:, 0:-1]
            pic = data  # For figures before standardization.
            data = data - np.mean(data.transpose().flatten())
            data = data / np.linalg.norm(data, 'fro')
            mc_size = data.shape
            micrograph = Micrograph(data, pic, mc_size, mrc_file.split('/')[-1], mrc_size)
            micrographs.append(micrograph)
        micrographs = np.array(micrographs)
        self.micrographs = micrographs


def main():
    args = parse_args()
    num_files = len(glob.glob("%s/*.mrc" % args.input_dir))
    if num_files > 0:
        print("Running on %i files." % len(glob.glob("%s/*.mrc" % args.input_dir)))
    else:
        print("Could not find any .mrc files in %s. \nExiting..." % args.input_dir)
        exit(0)
    picker = Picker(args)
    if args.preprocess:
        print("Preprocessing...")
        picker.preprocess()
        print("Preprocess finished.")
    else:
        print("Skipping preprocessing.")
    picker.get_micrographs()
    for micrograph in picker.micrographs:
        print("Processing %s" % micrograph.mrc_name)
        print("Cutoff filter...")
        micrograph.cutoff_filter(picker.patch_size)
        print("Done cutoff filter.\nEstimating RPSD I...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD I.")
        if picker.show_figures:
            plt.figure(1)
            plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
            plt.title('Approx Clean PSD stage I')
            plt.legend()
            plt.show()
            plt.figure(2)
            plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
            plt.title('Approx Noise PSD stage I')
            plt.legend()
            plt.show()
        micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
        print("Prewhitening...")
        micrograph.prewhiten_micrograph()
        print("Done prewhitening.\nEstimating RPSD II...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD II.\nConstructing KLT templates...")
        if picker.show_figures:
            plt.figure(3)
            plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
            plt.title('Approx Clean PSD stage II')
            plt.legend()
            plt.show()
            plt.figure(4)
            plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
            plt.title('Approx Noise PSD stage II')
            plt.legend()
            plt.show()
        micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                                   picker.rho.astype('float64')))
        if picker.show_figures:
            plt.figure(5)
            plt.plot(picker.rho, micrograph.psd)
            plt.title('Clean Sig Samp at nodes max order %i, percent of eig %f' % (picker.max_order, PERCENT_EIG_FUNC))
            plt.show()
        micrograph.construct_klt_templates(picker)
        print("Done constructing KLT templates.\nPicking particles...")
        num_picked_particles, num_picked_noise = micrograph.detect_particles(picker)
        print("Picked %i particles and %i noise images from %s.\n\n" % (
        num_picked_particles, num_picked_noise, micrograph.mrc_name))
    print("Finished successfully.")


if __name__ == "__main__":
    main()
