import logging
import glob
import os
import operator as op
import mrcfile
import numpy as np
from numpy.matlib import repmat
import scipy.special as ssp
from scipy import signal,interpolate
from scipy.ndimage import correlate, uniform_filter
from scipy.fftpack import fftshift
from src.aspire.aspire.class_averaging import lgwt
from aspire.aspire.preprocessor import PreProcessor
from aspire.aspire.utils.array_utils import cryo_epsds
import matplotlib.pyplot as plt
# added a comment to see
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

logger = logging.getLogger(__name__)

# utils


def f_trans_2(b):
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]])/8
    n = int((b.size - 1)/2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2*b[0:n+1]
    a[0] /= 2
    # use Chebyshev polynomials to compute h:
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


def radial_avg(image, num_bins):
    N = image.shape[1]
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    r = np.sqrt(X ** 2 + Y ** 2)
    dr = 1 / (num_bins - 1)
    rbins = np.linspace(-dr/2, 1+dr/2, num_bins+1, endpoint=True)
    R = (rbins[0:-1] + rbins[1:]) / 2
    zr = np.zeros(num_bins)
    for j in range(num_bins-1):
        bins = np.where(np.logical_and(r >= rbins[j], r < rbins[j+1]))
        n = np.count_nonzero(np.logical_and(r >= rbins[j], r < rbins[j+1]))
        if n != 0:
            zr[j] = sum(image[bins]) / n
        else:
            zr[j] = np.nan
    bins = np.where(np.logical_and(r >= rbins[num_bins-1], r <= 1))
    n = np.count_nonzero(np.logical_and(r >= rbins[num_bins-1], r <= 1))
    if n != 0:
        zr[num_bins-1] = sum(image[bins]) / n
    else:
        zr[num_bins-1] = np.nan
    return zr, R


def stdfilter(a, nhood):
    c1 = uniform_filter(a, nhood, mode='reflect')
    c2 = uniform_filter(a * a, nhood, mode='reflect')
    return np.sqrt(c2 - c1 * c1)*np.sqrt(nhood**2./(nhood**2-1))


def als_find_min(sreal, eps, max_iter):
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


def spline(x, y, xq):
    """
    returns a vector of interpolated values s corresponding to the query points in xq.
    The values of s are determined by cubic spline interpolation of x and y.
    """
    interpolator = interpolate.InterpolatedUnivariateSpline(x, y)
    return interpolator(xq)


class KLTPicker:
    """   """
    def __init__(self, particle_size, input_directory, output_directory):
        self.particle_size = int(particle_size)
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.mgscale = MG_SCALE
        self.max_order = MAX_ORDER
        self.mrc_files = glob.glob("%s/*.mrc" % self.input_directory)
        self.micrographs = np.array([])
        self.micrograph_pics = np.array([])
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
        self.mrc_stack = 0
        self.num_mrcs = len(self.mrc_files)
        patch_size = np.floor(0.8 * self.mgscale * self.particle_size)  # need to put the 0.8 somewhere else.
        if np.mod(patch_size, 2) == 0:
            patch_size -= 1
        self.patch_size = patch_size
        patch_size_function = np.floor(0.4 * self.mgscale * self.particle_size)  # need to put the 0.4 somewhere else.
        if np.mod(patch_size_function, 2) == 0:
            patch_size_function -= 1
        self.patch_size_func = patch_size_function
        self.max_iter = MAX_ITER

    def preprocess(self):
        radmax = np.floor((self.patch_size_func - 1) / 2)
        x = np.arange(-radmax, radmax+1, 1).astype('float32')
        X, Y = np.meshgrid(x, x)
        rsamp = np.sqrt(np.square(X)+np.square(Y)).flatten()
        theta = np.arctan2(Y, X).flatten()
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

    def get_micrographs(self):
        micrographs = []  # maybe in the first place this should be an array
        preprocessor = PreProcessor()
        for mrc_file in self.mrc_files:
            mrc = mrcfile.open(mrc_file)
            mrc_data = mrc.data.astype('float64').transpose()
            mrc.close()
            mrc_size = min(mrc_data.shape)
            mrc_data = mrc_data[0:mrc_size, 0:mrc_size]
            mrc_data = np.rot90(mrc_data)
            data = preprocessor.downsample(mrc_data, np.floor(MG_SCALE * mrc.shape[0]))
            if np.mod(data.shape[0], 2) == 0:  # we need odd size
                data = data[0:-1, 0:-1]
            pic = data   # for figures before standardization.
            data -= np.mean(data.flatten())
            data /= np.linalg.norm(data, 'fro')  # normalization
            mc_size = data.shape[0]
            micrograph = Micrograph(data, pic, mc_size, mrc_file, mrc_size)
            micrographs.append(micrograph)
        micrographs = np.array(micrographs)
        self.micrographs = micrographs


class Micrograph:
    """   """
    def __init__(self, micrograph, micrograph_pic,mc_size,mrc_name,mg_big_size):
        self.micrograph = micrograph
        self.micrograph_pic = micrograph_pic
        self.mc_size = mc_size
        self.mg_big_size = mg_big_size
        self.noise_mc = 0
        self.approx_clean_psd = 0
        self.approx_noise_psd = 0
        self.approx_noise_var = 0
        self.r = 0
        self.approx_scaling = 0
        self.stop_par = 0
        self.psd = 0
        self.eig_func = 0
        self.eig_val = 0
        self.num_of_func = 0
        self.eps = "???"
        self.rad_mat = 0
        self.mrc_name = mrc_name

    def cutoff_filter(self, patch_size):
        bandpass1d = signal.firwin(int(patch_size), np.array([0.05, 0.95]), pass_zero=False)
        bandpass2d = f_trans_2(bandpass1d)
        micrograph = correlate(self.micrograph, bandpass2d, mode='constant')
        self.noise_mc = micrograph

    def estimate_rpsd(self, patch_size, max_iter):
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
            noisemc_block = noisemc_block - np.mean(noisemc_block)
            psd_block = cryo_epsds(noisemc_block[:, :, np.newaxis], np.where(np.zeros((int(patch_size), int(patch_size))) == 0), np.floor(0.3 * patch_size).astype(int))
            psd_block = psd_block[0]
            if np.count_nonzero(np.isnan(psd_block)) != 0:
                print("got NaN")
            [r_block, r] = radial_avg(psd_block, L)
            block_var = np.var(noisemc_block.flatten(), ddof=1)
            psd_rad = np.abs(spline(r*np.pi, r_block, rho_samp))
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
        var_mat = var_mat[cut:-cut, cut:-cut]
        var_vec = var_mat.flatten()
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
        clean_psd_nodes = np.abs(spline(r * np.pi, approx_clean_psd, rho_samp))
        noise_psd_nodes = np.abs(spline(r * np.pi, approx_noise_psd, rho_samp))
        clean_psd_mat = np.reshape(clean_psd_nodes[idx], (num_of_quad, num_of_quad))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], (num_of_quad, num_of_quad))
        scaling_psd_approx = ((nodes@noise_psd_mat@nodes)-(4*np.pi**2)*noise_var_approx)/(nodes@clean_psd_mat@nodes)
        noise_psd_approx_sigma = approx_noise_psd - scaling_psd_approx * approx_clean_psd
        noise_psd_approx_sigma = noise_psd_approx_sigma.clip(min=0, max=None)
        s_mean = np.mean(s, 1)
        s_mean_psd_nodes = np.abs(spline(r * np.pi, s_mean, rho_samp))
        s_mean_psd_mat = np.reshape(s_mean_psd_nodes[idx], (num_of_quad, num_of_quad))
        s_mean_var_psd = (1 / (2 * np.pi) ** 2) * (nodes@s_mean_psd_mat@nodes)
        clean_var_psd = (1 / (2 * np.pi) ** 2)*(nodes@clean_psd_mat@nodes)
        clean_var = s_mean_var_psd - noise_var_approx
        noise_psd_approx_sigma += np.median(noise_psd_approx_sigma)/10
        self.approx_clean_psd = self.approx_scaling * approx_clean_psd
        self.approx_noise_psd = noise_psd_approx_sigma
        self.approx_noise_var = noise_var_approx
        self.r = r
        self.approx_scaling = clean_var / clean_var_psd
        self.stop_par = stop_par

    def prewhiten(self):
        L = np.floor((self.micrograph.shape[0] - 1) / 2).astype('int')
        x = np.linspace(-1, 1, 2 * L + 1) * np.pi
        X, Y = np.meshgrid(x, x)
        rad_mat = np.sqrt(X ** 2 + Y ** 2)
        rad_samp, idx = np.unique(rad_mat, return_inverse=True)
        noise_psd_nodes = np.abs(spline(self.r * np.pi, self.approx_noise_psd, rad_samp))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], [2 * L + 1, 2 * L + 1])
        preprocessor = PreProcessor()
        noise_mc_prewhite = preprocessor.cryo_prewhiten(self.noise_mc[:, :, np.newaxis], noise_psd_mat)
        noise_mc_prewhite = noise_mc_prewhite[0][:, :, 0]
        noise_mc_prewhite -= np.mean(noise_mc_prewhite)
        noise_mc_prewhite /= np.linalg.norm(noise_mc_prewhite, 'fro')
        self.noise_mc = noise_mc_prewhite
        self.rad_mat = rad_mat

    def construct_klt_templates(self, kltpicker):
        eig_func_tot = np.zeros((NUM_QUAD_NYS, NUM_QUAD_NYS, kltpicker.max_order))
        eig_val_tot = np.zeros((NUM_QUAD_NYS, kltpicker.max_order))
        sqrt_rr = np.sqrt(kltpicker.r_r)
        d_rho_psd_quad_ker = np.diag(kltpicker.rho)*np.diag(self.psd)*np.diag(kltpicker.quad_ker)
        sqrt_diag_quad_nys = np.sqrt(np.diag(kltpicker.quad_nys))
        for n in range(kltpicker.max_order):
            h_nodes = sqrt_rr*(kltpicker.j_r_rho[:, :, n] @ d_rho_psd_quad_ker @ kltpicker.j_r_rho[:, :, n])
            tmp = sqrt_diag_quad_nys @ h_nodes @ sqrt_diag_quad_nys
            V, D = np.linalg.eig(tmp)
            D = np.real(D)
            I = np.argsort(np.diag(D))
            D = np.sort(np.diag(D))
            V = V[:, I]
            D = np.where(np.abs(D) < kltpicker.eps, 0, D)
            V[:, D == 0] = 0
            eig_func_tot[:, :, n] = V
            eig_val_tot[:, n] = D
        r_idx = np.arange(0, NUM_QUAD_NYS)
        c_idx = np.arange(0, kltpicker.max_order)
        r_idx = repmat(r_idx, 1, kltpicker.max_order)
        c_idx = repmat(c_idx, NUM_QUAD_NYS, 1)
        eig_val_tot = eig_val_tot.flatten()
        r_idx = r_idx.flatten()
        c_idx = c_idx.flatten()
        idx = np.argsort(eig_val_tot)
        eig_val_tot = np.sort(eig_val_tot)
        r_idx = r_idx[idx, 1]
        c_idx = c_idx[idx, 1]
        sum_of_eig = np.sum(eig_val_tot)
        cum_sum_eig_val = np.cumsum(eig_val_tot / sum_of_eig)
        last_eig_idx = (cum_sum_eig_val > PERCENT_EIG_FUNC).argmax()
        eig_val = np.zeros((1, 2 * last_eig_idx))
        eig_func = np.zeros((kltpicker.r_samp.size, 2 * last_eig_idx))
        count = 1
        for i in range(last_eig_idx):
            order = c_idx[i] - 1
            idx_of_eig = r_idx[i]
            h_samp = np.sqrt(kltpicker.rsamp_r) * (kltpicker.j_samp[:, :, order] @ (np.diag(kltpicker.rho * self.psd * kltpicker.quad_ker) @ kltpicker.j_r_rho[:, :, order + 1]))
            v_correct = (1 / np.sqrt(kltpicker.quad_nys)) * eig_func_tot[:, idx_of_eig, order]
            v_nys = (h_samp @ (kltpicker.quad_nys * v_correct)) * (1 / eig_val_tot[i])
            if order == 0:
                eig_func[:, count] = (1 / np.sqrt(2 * np.pi)) * v_nys
                eig_val[count] = eig_val_tot[i]
                count += 1
            else:
                eig_func[:, count] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.cosine[:, order]
                eig_val[count] = eig_val_tot[i]
                count += 1
                eig_func[:, count] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.sine[:, order]
                eig_val[count] = eig_val_tot[i]
                count += 1
        eig_val = eig_val[eig_val > 0]
        eig_func = eig_func[:, 1:len(eig_val)]
        if eig_func.shape[1] < MAX_FUN:
            num_of_fun = eig_func.shape[1]
        else:
            num_of_fun = MAX_FUN
        self.eig_func = eig_func
        self.eig_val = eig_val
        self.num_of_func = num_of_fun

    def detect_particles(self, kltpicker):
        eig_func_stat = self.eig_func[:, 0:self.num_of_func]
        eig_val_stat = self.eig_val[0, 0:self.num_of_func]
        for i in range(self.num_of_func):
            tmp_func = np.reshape(eig_func_stat[:, i], (kltpicker.patch_size_func, kltpicker.patch_size_func))
            tmp_func[self.rad_mat > np.floor((kltpicker.patch_sz_func - 1) / 2)] = 0
            eig_func_stat[:, i] = tmp_func.flatten()
        [q, r] = np.linalg.qr(eig_func_stat, 'complete')
        self.r = self.r[0:self.num_of_func, 0:self.num_of_func]
        kappa = self.r @ np.diag(eig_val_stat) @ self.r + self.approx_noise_var * np.eye(self.num_of_func)
        kappa_inv = np.linalg.inv(kappa)
        t_mat = (1 / self.approx_noise_var) * np.eye(self.num_of_func) - kappa_inv
        mu = np.linalg.slogdet((1 / self.approx_noise_var) * kappa)
        last_block = self.mc_size - kltpicker.patch_sz_func + 1
        num_of_patch = last_block
        v = np.zeros((num_of_patch, num_of_patch, self.num_of_func))
        cnt = 0
        for i in range(self.num_of_func):
            cnt += 1
            q_tmp = np.reshape(q[:,i], (kltpicker.patch_sz_func, kltpicker.patch_sz_func))
            q_tmp = q_tmp - np.mean(q_tmp)
            q_tmp = np.flip(np.flip(q_tmp, 1), 2)
            if kltpicker.gpu_use == 1:
                pass
                # noiseMcGpu = gpuArray(single(noiseMc))
                # v_tmp = conv2(noiseMcGpu, q_tmp, 'valid')
                # v(:,:, i) = single(gather(v_tmp))
            else:
                v_tmp = signal.convolve2d(self.noise_mc, q_tmp, 'valid')
                v[:, :, i] = v_tmp.astype('single')
        log_test_mat = np.zeros((num_of_patch, num_of_patch))
        cnt = 0
        for j in range(num_of_patch):
            cnt += 1
            vc = np.reshape(v[:, j, :], (num_of_patch, num_of_patch, 1))
            log_test_mat[:, j] = np.sum((vc * t_mat) * vc, 2) - mu
        if kltpicker.gpu_use == 1:
            pass
            # neigh = gpuArray(ones(kltpicker.patch_sz_func))
            # logTestN = gather(conv2(logTestMat, neigh, 'valid'))
        else:
            neigh = np.ones(kltpicker.patch_sz_func)
            log_test_n = signal.convolve2d(log_test_mat, neigh, 'valid')
        [num_picked_particles, num_picked_noise] = picking_from_scoring_mat(log_test_n, self.mrc_name, kltpicker)
        return num_picked_particles, num_picked_noise


def picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size):
    idx = np.arange((log_test_n[0]).size)
    [col_idx, row_idx] = np.meshgrid(idx, idx)
    r_del = np.floor(kltpicker.patch_size_pick_box)
    shape = log_test_n.shape
    scoring_mat = log_test_n
    if kltpicker.num_of_particles == -1:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, np.iinfo(np.int32(10)).max, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_sz_func,
                                                  row_idx, col_idx, kltpicker.coordinates_path_particle, mrc_name,
                                                  kltpicker.mgscale, mg_big_size)
    else:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_particles, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_sz_func,
                                                  row_idx, col_idx, kltpicker.coordinates_path_particle, mrc_name,
                                                  kltpicker.mgscale, mg_big_size)
    if kltpicker.num_of_noise_images != 0:
        num_picked_noise = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_noise_images, op.lt,
                                              kltpicker.threshold-1, kltpicker.threshold, kltpicker.patch_sz_func,
                                              row_idx, col_idx, kltpicker. coordinates_path_noise, mrc_name,
                                              kltpicker.mgscale, mg_big_size)
    else:
        num_picked_noise = 0
    return num_picked_particles, num_picked_noise


def write_output_files(scoring_mat, shape, r_del, max_iter, oper, oper_param, threshold, patch_sz_func, row_idx,
                       col_idx, output_path, mrc_name, mgscale, mg_big_size):
    num_picked = 0
    box_path = output_path+'/box'
    star_path = output_path+'/star'
    if not os.path.isdir(box_path):
        os.mkdir(box_path)
    if not os.path.isdir(star_path):
        os.mkdir(star_path)
    box_file = open("%s/%s.box" % (box_path, mrc_name), 'w')
    star_file = open("%s/%s.star" % (star_path, mrc_name), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    iter_pick = 0
    while iter_pick <= max_iter and oper(oper_param, threshold):
        max_index = np.argmax(scoring_mat)
        oper_param = scoring_mat.flatten()[max_index]
        [index_row, index_col] = np.unravel_index(max_index, shape)
        ind_row_patch = (index_row - 1) + patch_sz_func
        ind_col_patch = (index_col - 1) + patch_sz_func
        row_idx_b = row_idx - index_row
        col_idx_b = col_idx - index_col
        rsquare = row_idx_b**2 + col_idx_b**2
        scoring_mat[rsquare <= (r_del**2)] = np.inf
        box_file.write('%i\t%i\t%i\t%i\n' % ((1 / mgscale) * (ind_col_patch - r_del / 2),
                                        (mg_big_size + 1) - (1 / mgscale) * (ind_row_patch + r_del / 2),
                                        (1 / mgscale) * r_del, (1 / mgscale) * r_del))
        star_file.write('%i\t%i\n' % ((1 / mgscale) * ind_col_patch, (mg_big_size + 1) - (1 / mgscale) * ind_row_patch))
        iter_pick += 1
        num_picked += 1
    star_file.close()
    box_file.close()
    return num_picked


def main():
    kltpicker = KLTPicker(PARTICLE_SIZE, MICRO_PATH, OUTPUT_PATH)
    kltpicker.preprocess()
    kltpicker.get_micrographs()
    for micrograph in kltpicker.micrographs:
        micrograph.cutoff_filter(kltpicker.patch_size)
        micrograph.estimate_rpsd(kltpicker.patch_size, kltpicker.max_iter)
        micrograph.prewhiten()
        micrograph.estimate_rpsd(kltpicker.patch_size, kltpicker.max_iter)
        micrograph.psd = np.abs(spline(np.pi * micrograph.r, micrograph.approx_clean_psd, kltpicker.rho))
        micrograph.construct_klt_templates(kltpicker)
        micrograph.detect_particles(kltpicker)
    print("woohoo")


if __name__ == "__main__":
    main()