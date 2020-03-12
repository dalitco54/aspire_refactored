from numpy.fft import fft2, ifft2
import numpy as np

def np_fftconvolve(a, b):
    return np.real(ifft2(fft2(A)*fft2(B, s=A.shape)))

