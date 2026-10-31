# utils_features.py
import numpy as np
import librosa
import pywt
from scipy import stats

def extract_rme(path, sr_target=None):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    n_fft = 1024
    hop_length = 512
    window = 'hamming'
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window))**2
    n_mels = 20
    rmel_max = 1127 * np.log(1 + (0.5 * sr) / 700.0)
    rmel_pts = np.linspace(0, rmel_max, n_mels + 2)
    freqs = 0.5 * sr - 700 * (np.exp(rmel_pts / 1127.0) - 1)
    freqs = freqs[::-1]
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_l, f_c):
            if f_c != f_l: filterbank[m - 1, k] = (k - f_l) / (f_c - f_l)
        for k in range(f_c, f_r):
            if f_r != f_c: filterbank[m - 1, k] = (f_r - k) / (f_r - f_c)
    energies = np.dot(filterbank, S)
    energies = np.log(energies + 1e-8)
    features = []
    for m in range(n_mels):
        vals = energies[m, :]
        features.extend([np.mean(vals), np.std(vals), stats.skew(vals), stats.kurtosis(vals)])
    return np.array(features)

def extract_dhs(path, sr_target=None):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    kernel = np.array([1.0, -2.0, 1.0])
    y_diff2 = np.convolve(y, kernel, mode='same')
    n_fft = 1024
    hop_length = 512
    S2 = np.abs(librosa.stft(y_diff2, n_fft=n_fft, hop_length=hop_length, window='hamming'))**2
    f_threshold = 0.75 * (sr / 2)
    bin_thresh = int(np.floor((n_fft + 1) * f_threshold / sr))
    bin_thresh = min(max(bin_thresh, 0), n_fft // 2)
    high_energy = np.sum(S2[bin_thresh:, :], axis=0)
    return np.array([np.mean(high_energy), np.std(high_energy), stats.skew(high_energy), stats.kurtosis(high_energy)])

def extract_wmc(path, sr_target=None):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    coeffs = pywt.wavedec(y, 'db4', level=2)
    features = []
    n_mfcc = 13
    for sub_signal in coeffs:
        if len(sub_signal) < 2:
            mfcc = np.zeros((n_mfcc, 1))
        else:
            mfcc = librosa.feature.mfcc(y=sub_signal, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512, window='hamming')
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
    return np.array(features)

def extract_dmc(path, sr_target=None):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    kernel = np.array([1.0, -2.0, 1.0])
    y_diff2 = np.convolve(y, kernel, mode='same')
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y_diff2, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512, window='hamming')
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

EXTRACTORS = {
    "rme": extract_rme,
    "dhs": extract_dhs,
    "wmc": extract_wmc,
    "dmc": extract_dmc
}