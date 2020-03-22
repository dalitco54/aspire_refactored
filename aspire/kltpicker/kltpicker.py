import logging
import argparse
import glob
import os
import operator as op
import mrcfile
import numpy as np
from numpy.matlib import repmat
import scipy.special as ssp
from scipy import signal, interpolate
from scipy.ndimage import correlate, uniform_filter
from scipy.fftpack import fftshift
from aspire.utils.common import lgwt
from aspire.preprocessor.downsample import downsample
from aspire.preprocessor.prewhiten  import cryo_prewhiten, cryo_epsds
from dev_utils import *
import matplotlib.pyplot as plt


# for testing, will be removed
PARTICLE_SIZE = 300
MICRO_PATH = '/scratch/dalit/KLTpicker/sizlessPicking/300&10028/'
OUTPUT_PATH = '/scratch/dalit/KLTpicker/sizlessPicking/300&10028/pick'
MAX_ITER = 6*(10**4)
MG_SCALE = 100 / PARTICLE_SIZE
MAX_ORDER = 100
EPS = 10 ** (-2)  # convergence term in ALS process
PERCENT_EIG_FUNC = 0.99
NUM_QUAD_NYS = 2**10
NUM_QUAD_KER = 2**10
MAX_FUN = 400
GPU_USE = 0

logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument('particle_size', type=float, help='Expected size of particles in pixels')
# parser.add_argument('input_dir', type=str, help='Path of directory of .mrc files to be processed')
# parser.add_argument('output_dir', type=str, help='Path of directory in which to write output')
# parser.add_argument('num_of_particles', type=int, help='Number of particles to pick per micrograph. If set to -1 will pick all particles', default=-1)
# parser.add_argument('num_of_noise_images', type=int, help='Number of noise images to pick per micrograph', default=0)
# parser.add_argument('num_of_noise_images', type=int, help='Number of noise images to pick per micrograph', default=0)
# parser.add_argument('--max_iter', type=int, help='Maximum number of iterations', default=6*(10**4))
# parser.add_argument('--gpu_use', type=bool, action='store_const', default=0)
# parser.add_argument('--max_order', type=int, help='Maximum order of eigenfunction', default=100)
# parser.add_argument('percent_eigen_func', type=float, help='', default=0.99)
# parser.add_argument('max_functions', type=int, help='', default=400)
# parser.add_argument('verbose', type=bool, help='Verbose', default=0)
# parser.add_argument('threshold', type=float, help='Threshold for the picking', default=0)
# parser.add_argument('--show_figures', type=bool, action='store_const', help='Show figures', default=0)
# parser.add_argument('--preprocess', type=bool, action='store_const', help='Run preprocessing', default=1)
# args = parser.parse_args()

# utils


def f_trans_2(b):
    """
    2-D FIR filter using frequency transformation.

    Produces the 2-D FIR filter h that corresponds to the 1-D FIR
    filter b using the McClellan transform.
    :param b: 1-D FIR filter.
    :return h: 2-D FIR filter.
    """
    # McClellan transformation:
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]])/8
    n = int((b.size - 1)/2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2*b[0:n+1]
    a[0] /= 2
    # Use Chebyshev polynomials to compute h:
    p0 = 1
    p1 = t
    h = a[1]*p1
    rows = 1
    cols = 1
    h[rows, cols] = h[rows, cols] + a[0]*p0
    p2 = 2 * signal.convolve2d(t, p1)
    p2[2, 2] = p2[2, 2] - p0
    for i in range(2, n+1):
        rows = p1.shape[0]+1
        cols = p1.shape[1]+1
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
    X, Y = np.meshgrid(np.arange(N) * 2 / (N-1) - 1, np.arange(N) * 2 / (N-1) - 1)
    r = np.sqrt(np.square(X) + np.square(Y))
    dr = 1 / (m - 1)
    rbins = np.linspace(-dr / 2, 1 + dr / 2, m + 1, endpoint=True)
    R = (rbins[0:-1] + rbins[1:]) / 2
    zr = np.zeros(m)
    for j in range(m - 1):
        bins = np.where(np.logical_and(r >= rbins[j], r < rbins[j+1]))
        n = np.count_nonzero(np.logical_and(r >= rbins[j], r < rbins[j+1]))
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
    return np.sqrt(c2 - c1 * c1)*np.sqrt(nhood**2./(nhood**2-1))


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
    alpha_tmp = (clean_sig_tmp@s)/np.sum(clean_sig_tmp**2)
    alpha_tmp = alpha_tmp.clip(min=0, max=1)
    stop_par = 0
    cnt = 1
    while stop_par == 0:
        if np.linalg.norm(alpha_tmp, 1) == 0:
            alpha_tmp = np.random.random(alpha_tmp.size)
        approx_clean_psd = (s @ alpha_tmp)/sum(alpha_tmp ** 2)
        approx_clean_psd = approx_clean_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_clean_psd, alpha_tmp)
        approx_noise_psd = (s@np.ones(patch_num))/patch_num
        approx_noise_psd = approx_noise_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_noise_psd, One)
        if np.linalg.norm(approx_clean_psd, 1) == 0:
            approx_clean_psd = np.random.random(approx_clean_psd.size)
        alpha_approx = (approx_clean_psd@s)/sum(approx_clean_psd**2)
        alpha_approx = alpha_approx.clip(min=0, max=1)
        if np.linalg.norm(noise_sig_tmp-approx_noise_psd) / np.linalg.norm(approx_noise_psd) < eps:
            if np.linalg.norm(clean_sig_tmp-approx_clean_psd) / np.linalg.norm(approx_clean_psd) < eps:
                if np.linalg.norm(alpha_approx-alpha_tmp) / np.linalg.norm(alpha_approx) < eps:
                    break
        noise_sig_tmp = approx_noise_psd
        alpha_tmp = alpha_approx
        clean_sig_tmp = approx_clean_psd
        cnt += 1
        if cnt > max_iter:
            stop_par = 1
            break
    return approx_clean_psd, approx_noise_psd, alpha_approx, stop_par


# def spline(x, y, xq):
#     """
#     returns a vector of interpolated values s corresponding to the query points in xq. The values of s are determined by cubic spline interpolation of x and y.
#     """
#     interpolator = interpolate.InterpolatedUnivariateSpline(x, y)
#     return interpolator(xq)


def trigcardinal(x, n):
    if n%2 == 1:
        return np.sin(n * np.pi * x / 2) / (n * np.sin(np.pi * x / 2))
    else:
        return np.sin(n * np.pi * x / 2) / (n * np.tan(np.pi * x / 2))


def trig_interpolation(x, y, xq):
    n = x.size
    h = 2 / n
    scale = (x[1] - x[0]) / h
    x /= scale
    xq /= scale
    p = np.zeros((xq.size, xq.size))
    for k in range(n):
        p = p + y[k] * trigcardinal(xq - x[k], n)
    return p


class KLTPicker:
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
    def __init__(self, particle_size, input_directory, output_directory, gpu_use):
        self.particle_size = int(particle_size)
        self.input_dir = input_directory
        self.output_dir = output_directory
        self.output_noise = '%s/PickedNoise_ParticleSize_%d' % (output_directory, particle_size)
        self.output_particles = '%s/PickedParticles_ParticleSize_%d' % (output_directory, particle_size)
        self.gpu_use = GPU_USE
        self.mgscale = MG_SCALE
        self.max_order = MAX_ORDER
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
        self.patch_size_pick_box = np.floor(MG_SCALE * particle_size)
        self.num_of_particles = -1
        self.num_of_noise_images = 0
        self.threshold = 0
        patch_size = np.floor(0.8 * self.mgscale * self.particle_size)  # need to put the 0.8 somewhere else.
        if np.mod(patch_size, 2) == 0:
            patch_size -= 1
        self.patch_size = patch_size
        patch_size_function = np.floor(0.4 * self.mgscale * self.particle_size)  # need to put the 0.4 somewhere else.
        if np.mod(patch_size_function, 2) == 0:
            patch_size_function -= 1
        self.patch_size_func = int(patch_size_function)
        self.max_iter = MAX_ITER
        self.rsamp_length = 0
        self.rad_mat = 0

    def preprocess(self):
        """Initializes parameters."""
        radmax = np.floor((self.patch_size_func - 1) / 2)
        x = np.arange(-radmax, radmax+1, 1).astype('float32')
        X, Y = np.meshgrid(x, x)
        rad_mat = np.sqrt(np.square(X) + np.square(Y))
        rsamp = rad_mat.transpose().flatten()
        self.rsamp_length = rsamp.size
        theta = np.arctan2(Y, X).transpose().flatten()
        quad_ker_sample_points = lgwt(NUM_QUAD_KER, 0, np.pi)
        rho = quad_ker_sample_points.x
        quad_ker = quad_ker_sample_points.w
        rho = np.flipud(rho.astype('float32'))
        quad_ker = np.flipud(quad_ker.astype('float32'))
        quad_nys_sample_points = lgwt(NUM_QUAD_NYS, 0, radmax)
        r = quad_nys_sample_points.x
        quad_nys = quad_nys_sample_points.w
        r = np.flipud(r.astype('float32'))
        quad_nys = np.flipud(quad_nys.astype('float32'))
        r_r = np.outer(r, r)
        r_rho = np.outer(r, rho)
        rsamp_r = np.outer(np.ones(len(rsamp)), r)
        rsamp_rho = np.outer(rsamp, rho)
        j_r_rho = np.zeros([NUM_QUAD_KER, NUM_QUAD_NYS, self.max_order]).astype('float32')
        j_samp = np.zeros([len(rsamp), NUM_QUAD_NYS, self.max_order]).astype('float32')
        cosine = np.zeros([len(theta), self.max_order]).astype('float32')
        sine = np.zeros([len(theta), self.max_order]).astype('float32')
        for n in range(self.max_order):
            print(n)
            j_r_rho[:, :, n] = ssp.jv(n, r_rho).astype('float32')
            j_samp[:, :, n] = ssp.jv(n, rsamp_rho).astype('float32')
            if n != 0:
                cosine[:, n] = np.cos(n * theta).astype('float32')
                sine[:, n] = np.sin(n * theta).astype('float32')
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
        """Reads .mrc files, downsamples them and adds them to the KLTpicker object."""
        micrographs = []
        mrc_files = glob.glob("%s/*.mrc" % self.input_dir)
        for mrc_file in mrc_files:
            mrc = mrcfile.open(mrc_file)
            mrc_data = mrc.data.astype('float64').transpose()
            mrc.close()
            mrc_size = mrc_data.shape
            mrc_data = np.rot90(mrc_data)
            data = downsample(mrc_data[np.newaxis, :, :], int(np.floor(MG_SCALE * mrc_size[0])))[0]
            if np.mod(data.shape[0], 2) == 0:  # Odd size is needed.
                data = data[0:-1, :]
            if np.mod(data.shape[1], 2) == 0:  # Odd size is needed.
                data = data[:, 0:-1]
            pic = data   # For figures before standardization.
            data -= np.mean(data.transpose().flatten())
            data /= np.linalg.norm(data, 'fro')
            mc_size = data.shape
            micrograph = Micrograph(data, pic, mc_size, mrc_file.split('/')[-1], mrc_size)
            micrographs.append(micrograph)
        micrographs = np.array(micrographs)
        self.micrographs = micrographs


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
        micro_size = self.noise_mc.shape[1]
        m = np.floor(micro_size/patch_size)
        M = (m**2).astype(int)
        L = np.ceil(np.sqrt(2)*(patch_size-1)+1).astype(int)
        s = np.zeros((L, M))
        num_quads = 2**9
        quad_sample_points = lgwt(num_quads, -np.pi, np.pi)
        quad = np.flipud(quad_sample_points.x)
        nodes = np.flipud(quad_sample_points.w)
        x = repmat(quad, num_quads, 1)
        y = x.transpose()
        rho_mat = np.sqrt(x**2+y**2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        r_tmp = np.zeros((L, 1))
        for k in range(M):
            row = np.ceil((k+1) / m).astype(int)
            col = (k+1 - (row - 1) * m).astype(int)
            noisemc_block = self.noise_mc[(row - 1) * patch_size.astype(int):row * patch_size.astype(int), (col - 1) * patch_size.astype(int): col * patch_size.astype(int)]
            noisemc_block -= np.mean(noisemc_block)
            psd_block = cryo_epsds(noisemc_block[:, :, np.newaxis], np.where(np.zeros((int(patch_size), int(patch_size))) == 0), np.floor(0.3 * patch_size).astype(int))
            psd_block = psd_block[0]
            if np.count_nonzero(np.isnan(psd_block)) != 0:
                print("got NaN")
            [r_block, r] = radial_avg(psd_block, L)
            block_var = np.var(noisemc_block.transpose().flatten(), ddof=1)
            psd_rad = np.abs(trig_interpolation(r*np.pi, r_block, rho_samp))  #  here stuff starts to be a bit different
            psd_mat = np.reshape(psd_rad[idx], [num_quads, num_quads])
            var_psd = (1/(2*np.pi)**2) * (nodes@psd_mat@nodes)
            scaling_psd = block_var / var_psd
            r_block = scaling_psd * r_block
            s[:, k] = r_block
            if k == 1:
                r_tmp = r
        # find min arg using ALS:
        r = r_tmp
        approx_clean_psd, approx_noise_psd, alpha, stop_par = als_find_min(s, EPS, max_iter)
        std_mat = stdfilter(self.noise_mc, patch_size)
        var_mat = std_mat**2
        cut = int((patch_size - 1) / 2 + 1)
        var_mat = var_mat[cut-1:-cut, cut-1:-cut]
        var_vec = var_mat.transpose().flatten()
        var_vec.sort()
        j = np.floor(0.25 * var_vec.size).astype('int')
        noise_var_approx = np.mean(var_vec[0:j])
        num_of_quad = 2 ** 12
        sample_points = lgwt(num_of_quad, -np.pi, np.pi)
        quad = np.flipud(sample_points.x)
        nodes = np.flipud(sample_points.w)
        y = repmat(quad, num_of_quad, 1)
        x = y.transpose()
        rho_mat = np.sqrt(x**2 + y**2)
        rho_mat = np.where(rho_mat > np.pi,0,rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        clean_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_clean_psd, rho_samp))
        noise_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_noise_psd, rho_samp))
        clean_psd_mat = np.reshape(clean_psd_nodes[idx], (num_of_quad, num_of_quad))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], (num_of_quad, num_of_quad))
        scaling_psd_approx = ((nodes@noise_psd_mat@nodes)-(4*np.pi**2)*noise_var_approx)/(nodes@clean_psd_mat@nodes)
        noise_psd_approx_sigma = approx_noise_psd - scaling_psd_approx * approx_clean_psd
        noise_psd_approx_sigma = noise_psd_approx_sigma.clip(min=0, max=None)
        s_mean = np.mean(s, 1)
        s_mean_psd_nodes = np.abs(trig_interpolation(r * np.pi, s_mean, rho_samp))
        s_mean_psd_mat = np.reshape(s_mean_psd_nodes[idx], (num_of_quad, num_of_quad))
        s_mean_var_psd = (1 / (2 * np.pi) ** 2) * (nodes@s_mean_psd_mat@nodes)
        clean_var_psd = (1 / (2 * np.pi) ** 2)*(nodes@clean_psd_mat@nodes)
        clean_var = s_mean_var_psd - noise_var_approx
        approx_scaling = clean_var / clean_var_psd
        self.approx_clean_psd = approx_scaling * approx_clean_psd
        self.approx_noise_psd = noise_psd_approx_sigma
        self.approx_noise_var = noise_var_approx
        self.r = r
        self.stop_par = stop_par

    def prewhiten_micrograph(self):
        # L = np.floor((self.micrograph.shape[0] - 1) / 2).astype('int')
        r = np.floor((self.micrograph.shape[0] - 1) / 2).astype('int')
        c = np.floor((self.micrograph.shape[1] - 1) / 2).astype('int')
        col = np.linspace(-1, 1, 2 * c + 1) * np.pi
        row = np.linspace(-1, 1, 2 * r + 1) * np.pi
        Row, Col = np.meshgrid(row, col)
        rad_mat = np.sqrt(Row ** 2 + Col ** 2)
        rad_samp, idx = np.unique(rad_mat, return_inverse=True)
        rad_samp[rad_samp>np.max(self.r*np.pi)] = 0
        noise_psd_nodes = np.abs(trig_interpolation(self.r * np.pi, self.approx_noise_psd, rad_samp))
        noise_psd_nodes = np.pad(noise_psd_nodes, (0, rad_samp.size - noise_psd_nodes.size), 'constant', constant_values=noise_psd_nodes[-1])
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], [col.size, row.size])
        noise_mc_prewhite = cryo_prewhiten(self.noise_mc[:, :, np.newaxis], noise_psd_mat)
        noise_mc_prewhite = noise_mc_prewhite[0][:, :, 0]
        noise_mc_prewhite -= np.mean(noise_mc_prewhite)
        noise_mc_prewhite /= np.linalg.norm(noise_mc_prewhite, 'fro')
        self.noise_mc = noise_mc_prewhite

    def construct_klt_templates(self, kltpicker):
        """Constructing the KLTpicker templates as the eigenfunctions of a given kernel."""
        eig_func_tot = np.zeros((NUM_QUAD_NYS, NUM_QUAD_NYS, kltpicker.max_order))
        eig_val_tot = np.zeros((NUM_QUAD_NYS, kltpicker.max_order))
        sqrt_rr = np.sqrt(kltpicker.r_r)
        d_rho_psd_quad_ker = np.diag(kltpicker.rho)*np.diag(self.psd)*np.diag(kltpicker.quad_ker)
        sqrt_diag_quad_nys = np.sqrt(np.diag(kltpicker.quad_nys))
        for n in range(kltpicker.max_order):
            h_nodes = sqrt_rr*(kltpicker.j_r_rho[:, :, n] @ d_rho_psd_quad_ker @ kltpicker.j_r_rho[:, :, n])
            tmp = sqrt_diag_quad_nys @ h_nodes @ sqrt_diag_quad_nys
            eig_vals, eig_funcs = np.linalg.eig(tmp)
            eig_vals = np.real(eig_vals)
            sort_idx = np.argsort(eig_vals)
            sort_idx = sort_idx[::-1]  #  Descending.
            eig_vals = eig_vals[sort_idx]
            eig_funcs = eig_funcs[sort_idx]
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
        eig_val_tot=eig_val_tot[sort_idx]
        r_idx = r_idx[sort_idx]
        c_idx = c_idx[sort_idx]
        sum_of_eig = np.sum(eig_val_tot)
        cum_sum_eig_val = np.cumsum(eig_val_tot / sum_of_eig)
        last_eig_idx = (cum_sum_eig_val > PERCENT_EIG_FUNC).argmax() + 1
        eig_val = np.zeros((1, 2 * last_eig_idx))
        eig_func = np.zeros((kltpicker.rsamp_length, 2 * last_eig_idx))
        count = 0
        for i in range(last_eig_idx):
            print(i)
            order = c_idx[i]
            idx_of_eig = r_idx[i]
            h_samp = np.sqrt(kltpicker.rsamp_r) * (kltpicker.j_samp[:, :, order] @ (np.diag(kltpicker.rho * self.psd * kltpicker.quad_ker) @ kltpicker.j_r_rho[:, :, order]))
            v_correct = (1 / np.sqrt(kltpicker.quad_nys)) * eig_func_tot[:, idx_of_eig, order]
            v_nys = (h_samp @ (kltpicker.quad_nys * v_correct)) * (1 / eig_val_tot[i])
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
        kappa = (r @ np.diag(eig_val_stat) @ r.transpose()) + (self.approx_noise_var * np.eye(self.num_of_func))
        kappa_inv = np.linalg.inv(kappa)
        t_mat = (1 / self.approx_noise_var) * np.eye(self.num_of_func) - kappa_inv
        mu = np.linalg.slogdet((1 / self.approx_noise_var) * kappa)[1]
        #last_block = self.mc_size - kltpicker.patch_size_func + 1
        #num_of_patch = last_block
        last_block_row = self.mc_size[0] - kltpicker.patch_size_func + 1
        last_block_col = self.mc_size[1] - kltpicker.patch_size_func + 1
        num_of_patch_row = last_block_row
        num_of_patch_col = last_block_col

        v = np.zeros((num_of_patch_row, num_of_patch_col, self.num_of_func))
        cnt = 0
        for i in range(self.num_of_func):
            print("%d/%d"%(i,self.num_of_func))
            cnt += 1
            q_tmp = np.reshape(q[:,i], (kltpicker.patch_size_func, kltpicker.patch_size_func)).transpose()
            q_tmp = q_tmp - np.mean(q_tmp)
            q_tmp = np.flip(q_tmp, 1)
            if kltpicker.gpu_use == 1:
                pass
                # noiseMcGpu = gpuArray(single(noiseMc))
                # v_tmp = conv2(noiseMcGpu, q_tmp, 'valid')
                # v(:,:, i) = single(gather(v_tmp))
            else:
                v_tmp = signal.fftconvolve(self.noise_mc, q_tmp, 'valid')
                v[:, :, i] = v_tmp.astype('single')  # different signs for values compared to matlab output, possible issue.
        log_test_mat = np.zeros((num_of_patch_row, num_of_patch_col))
        cnt = 0
        for j in range(num_of_patch_col):
            cnt += 1
            vc = np.reshape(v[:, j, :], (num_of_patch_row, self.num_of_func))
            log_test_mat[:, j] = np.sum((vc @ t_mat) * vc, 1) - mu
        if kltpicker.gpu_use == 1:
            pass
            # neigh = gpuArray(ones(kltpicker.patch_sz_func))
            # logTestN = gather(conv2(logTestMat, neigh, 'valid'))
        else:
            neigh = np.ones((kltpicker.patch_size_func, kltpicker.patch_size_func))
            log_test_n = signal.fftconvolve(log_test_mat, neigh, 'valid')
        # log_test_n = mat_to_npy('logTestN', '/home/dalitcohen/Documents/kltdata/matlab')
        [num_picked_particles, num_picked_noise] = picking_from_scoring_mat(log_test_n, self.mrc_name, kltpicker, self.mg_big_size)
        return num_picked_particles, num_picked_noise


def picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size):
    idx_row = np.arange(log_test_n.shape(0))
    idx_col = np.arange(log_test_n.shape(1))
    [col_idx, row_idx] = np.meshgrid(idx_col, idx_row)
    r_del = np.floor(kltpicker.patch_size_pick_box)
    shape = log_test_n.shape
    scoring_mat = log_test_n
    if kltpicker.num_of_particles == -1:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, np.iinfo(np.int32(10)).max, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_particles, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf, kltpicker.patch_size_pick_box)
    if kltpicker.num_of_noise_images != 0:
        num_picked_noise = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_noise_images, op.lt,
                                              kltpicker.threshold-1, kltpicker.threshold, kltpicker.patch_size_func,
                                              row_idx, col_idx, kltpicker. output_noise, mrc_name,
                                              kltpicker.mgscale, mg_big_size, np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_noise = 0
    return num_picked_particles, num_picked_noise


def write_output_files(scoring_mat, shape, r_del, max_iter, oper, oper_param, threshold, patch_size_func, row_idx,
                       col_idx, output_path, mrc_name, mgscale, mg_big_size, replace_param, patch_size_pick_box):
    num_picked = 0
    box_path = output_path+'/box'
    star_path = output_path+'/star'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(box_path):
        os.mkdir(box_path)
    if not os.path.isdir(star_path):
        os.mkdir(star_path)
    box_file = open("%s/%s.box" % (box_path, mrc_name), 'w')
    star_file = open("%s/%s.star" % (star_path, mrc_name), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    iter_pick = 0
    while iter_pick <= max_iter and oper(oper_param, threshold):
        max_index = np.argmax(scoring_mat.transpose().flatten())
        oper_param = scoring_mat.transpose().flatten()[max_index]
        print(oper_param)
        [index_col, index_row] = np.unravel_index(max_index, shape)
        ind_row_patch = (index_row - 1) + patch_size_func
        ind_col_patch = (index_col - 1) + patch_size_func
        row_idx_b = row_idx - index_row
        col_idx_b = col_idx - index_col
        rsquare = row_idx_b**2 + col_idx_b**2
        scoring_mat[rsquare <= (r_del**2)] = replace_param
        box_file.write('%i\t%i\t%i\t%i\n' % ((1 / mgscale) * (ind_col_patch + 1 - np.floor(patch_size_pick_box / 2)),
                                        (mg_big_size[0] + 1) - (1 / mgscale) * (ind_row_patch + 1 + np.floor(patch_size_pick_box / 2)),
                                        (1 / mgscale) * patch_size_pick_box, (1 / mgscale) * patch_size_pick_box))
        star_file.write('%i\t%i\t%f\n' % ((1 / mgscale) * (ind_col_patch + 1), (mg_big_size[0] + 1) - ((1 / mgscale) * (ind_row_patch + 1)), oper_param/np.max(scoring_mat)))
        iter_pick += 1
        num_picked += 1
        print("%d\tpmax: %f"%(num_picked, oper_param))
    star_file.close()
    box_file.close()
    return num_picked


def main():
    kltpicker = KLTPicker(PARTICLE_SIZE, MICRO_PATH, OUTPUT_PATH, GPU_USE)
    print("Starting preprocessing.")
    # kltpicker.preprocess()
    print("Preprocess finished.\nStarting particle picking.")
    kltpicker.rsamp_length = 1521
    kltpicker.j_samp = mat_to_npy('Jsamp','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.cosine = mat_to_npy('cosine','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.sine = mat_to_npy('sine','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.j_r_rho = mat_to_npy('JrRho','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.quad_ker = mat_to_npy_vec('quadKer','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.quad_nys = mat_to_npy_vec('quadNys','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.rho = mat_to_npy_vec('rho','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.rsamp_r = mat_to_npy('rSampr','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.r_r = mat_to_npy('rr','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.rad_mat = mat_to_npy('radMat','/home/dalitcohen/Documents/kltdata/matlab')
    kltpicker.get_micrographs()
    for micrograph in kltpicker.micrographs:
        micrograph.cutoff_filter(kltpicker.patch_size)
        micrograph.estimate_rpsd(kltpicker.patch_size, kltpicker.max_iter)
        micrograph.approx_noise_psd += np.median(micrograph.approx_noise_psd)/10
        micrograph.prewhiten_micrograph()
        micrograph.estimate_rpsd(kltpicker.patch_size, kltpicker.max_iter)
        # micrograph.r = mat_to_npy_vec('R','/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.approx_clean_psd = mat_to_npy_vec('apprxCleanPsd','/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.approx_noise_psd = mat_to_npy_vec('apprxNoisePsd','/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.approx_noise_var = mat_to_npy_vec('noiseVar','/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.micrograph = mat_to_npy('mg','/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.noise_mc = mat_to_npy('noiseMc','/home/dalitcohen/Documents/kltdata/matlab')
        micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r, micrograph.approx_clean_psd, kltpicker.rho))
        # micrograph.psd = mat_to_npy_vec('psd', '/home/dalitcohen/Documents/kltdata/matlab')
        micrograph.construct_klt_templates(kltpicker)
        # micrograph.eig_func = mat_to_npy('eigFun', '/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.eig_val = mat_to_npy_vec('eigVal', '/home/dalitcohen/Documents/kltdata/matlab')
        # micrograph.num_of_func = 400
        micrograph.detect_particles(kltpicker)
    print("")


if __name__ == "__main__":
    main()