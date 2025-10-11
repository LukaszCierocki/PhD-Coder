import sys
import os
import logging
import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.Method import MultiLevelSVDProcessor, FrameProcessor
from lib.Tools import Tools

np.set_printoptions(precision=4, suppress=True, linewidth=120)

def hankel_k_row(vector, k):
    """
    Tworzy macierz Hankela o k wierszach z wektora wejściowego.
    
    Parametry:
    - vector: 1D array-like, dane wejściowe
    - k: int, liczba wierszy
    
    Zwraca:
    - 2D numpy array o wymiarach (k, len(vector) - k + 1)
    """
    vector = np.asarray(vector)
    if k > len(vector):
        raise ValueError("Liczba wierszy k nie może być większa niż długość wektora")
    
    cols = len(vector) - k + 1
    return np.vstack([vector[i:i + cols] for i in range(k)])

def diagonal_average(matrix):
    rows, cols = matrix.shape
    result = np.zeros(rows + cols - 1)
    count = np.zeros(rows + cols - 1)
    for i in range(rows):
        for j in range(cols):
            result[i + j] += matrix[i, j]
            count[i + j] += 1
    return result / count

def multi_level_decompose(signal, num_levels):
    levels = []
    current = signal

    print("XXX STARTING MULTI-LEVEL DECOMPOSITION XXX")

    for _ in range(num_levels):
        H = hankel_k_row(current, 2)
        print(f"Current Hankel matrix (level {_}):\n", np.asarray(H))
        print('\n')
        U, s, Vh = svd(H, full_matrices=False)

        # print(f"SVD U (level {_}):\n", U)
        print('\n')
        print(f"SVD s (level {_}):\n", s)
        print('\n')
        # print(f"SVD Vh (level {_}):\n", Vh)
        
        levels.append((U, s, Vh))
        
        Ha = s[0] * np.outer(U[:, 0], Vh[0, :])
        print('\n')

        # print(f"Ha (level {_}):\n", Ha)
        
        current = diagonal_average(Ha)

        print('\n')

        print(f"Current signal after diagonal average (level {_}):\n", current)
        print('\n')

    print("XXX MULTI-LEVEL DECOMPOSITION COMPLETED XXX")

    return levels

def reconstruct_from_levels(levels):

    U, s, Vh = levels[-1]


    Ha = s[0] * np.outer(U[:, 0], Vh[0, :])

    Hd = s[1] * np.outer(U[:, 1], Vh[1, :])

    currentL = diagonal_average(Ha + Hd)

    i = 0

    for U, s, Vh in reversed(levels[:-1]):

        Hd = s[1] * np.outer(U[:, 1], Vh[1, :])

        currentR = diagonal_average(Hd)

        Hd_reconstructed = hankel_k_row(currentR, 2)

        Ha_reconstructed = hankel_k_row(currentL, 2)

        H = Ha_reconstructed + Hd_reconstructed

        currentL = diagonal_average(H)

    return currentL

def filter_dct_coeffs(coeffs: np.ndarray, Fs: float, low_freq: float, high_freq: float):
    """
    Zwraca współczynniki DCT tylko z określonego zakresu częstotliwości.

    :param coeffs: numpy array z współczynnikami DCT (1D)
    :param Fs: częstotliwość próbkowania (Hz)
    :param low_freq: dolna granica zakresu (Hz)
    :param high_freq: górna granica zakresu (Hz)
    :return: numpy array zawierający tylko współczynniki z określonego zakresu częstotliwości
    """
    N = len(coeffs)

    # Konwersja zakresu z Hz na indeksy
    k_low = int(low_freq / (Fs / 2) * N)
    k_high = int(high_freq / (Fs / 2) * N)

    # Upewniamy się, że indeksy są w dozwolonym zakresie
    k_low = max(0, k_low)
    k_high = min(N, k_high)

    # Zwracamy tylko wycinek - same istotne współczynniki

    return coeffs[k_low:k_high], k_low, k_high

if __name__ == "__main__":

    tools = Tools(logging.DEBUG)
    logger = tools.get_logger()
    
    carriers = [tools.load_carrier('arctic_a0001')]

    carrier_data = carriers[0].data_signed_int16
    carrier_fs = carriers[0].fs

    print("Carrier:", carrier_data)

    frame_processor = FrameProcessor(logger)

    voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices, remaining_samples = frame_processor.divide_into_voiced_unvoiced_frames('rVAD', carrier_data, carrier_fs, 10, 10)

    # print("Voiced frames: \n",   voiced_frames)
    # print("Unvoiced frames: \n", unvoiced_frames)

    for voiced_frame in voiced_frames:

        print("Voiced frame:\n", voiced_frame)
        print('\n')

        # Przetwarzanie ramki głosowej

        frame_dct = dct(voiced_frame, norm='ortho')

        filtered_dct_coeffs, idx_low, idx_high = filter_dct_coeffs(frame_dct, carrier_fs, 1500, 2500)

        print("Filtered DCT coefficients:\n", filtered_dct_coeffs)

        evn_dct_coeffs = []
        odd_dct_coeffs = []

        for i, coeff in enumerate(filtered_dct_coeffs):
            if i % 2 == 0:
                evn_dct_coeffs.append(coeff)
            else:
                odd_dct_coeffs.append(coeff)

        evn_dct_coeffs = np.array(evn_dct_coeffs)
        odd_dct_coeffs = np.array(odd_dct_coeffs)

        print('\n')

        print("Even DCT coefficients:\n", evn_dct_coeffs)

        print('\n')

        print("Odd DCT coefficients:\n", odd_dct_coeffs)

        print('\n')

        s_even = []
        s_odd = []

        level = 8

        multi_level_decomposed_evn = multi_level_decompose(evn_dct_coeffs, level)
        multi_level_decomposed_odd = multi_level_decompose(odd_dct_coeffs, level)

        alphas = []
        betas = []

        new_s_even = []
        new_s_odd = []

        bit = 0

        alpha = 0.7
        beta = 0.7

        new_levels_evn = []
        new_levels_odd = []

        i = 0

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(multi_level_decomposed_evn, multi_level_decomposed_odd):

            if i == level - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                print(f"Max even singular value: {ro_11}, Max odd singular value: {ro_21}")
                print(f"Difference: {ro_11 - ro_21}, level: {i}")

                # s_even.append(ro_11)
                # s_odd.append(ro_21)

                T_bu_1 = alpha + 2.0*(ro_21 - ro_11)/(ro_21 + ro_11) 
                T_bu_2 = beta + 2.0*(ro_11 - ro_21)/(ro_21 + ro_11)

                alphas.append(T_bu_1)
                betas.append(T_bu_2)

                ro_dash_11 = 0
                ro_dash_21 = 0

                davg = (ro_11 + ro_21) / 2.0

                if bit == 0:

                    if ro_11 - ro_21 >= T_bu_1*davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_21 + T_bu_1*davg/2
                        ro_dash_21 = ro_21 - T_bu_1*davg/2

                elif bit == 1:
                    if (ro_21 - ro_11) >= T_bu_2*davg:
                        ro_dash_11 = ro_11
                        ro_dash_21 = ro_21
                    else:
                        ro_dash_11 = ro_11 - T_bu_2*davg/2
                        ro_dash_21 = ro_21 + T_bu_2*davg/2

                s_evn[s_evn_max_idx] = ro_dash_11
                s_odd[s_odd_max_idx] = ro_dash_21

                new_s_even.append(ro_dash_11)
                new_s_odd.append(ro_dash_21)
            
            else:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                print(f"Max even singular value: {ro_11}, Max odd singular value: {ro_21}")
                print(f"Difference: {ro_11 - ro_21}, level: {i}")

            new_levels_evn.append((U_evn, s_evn, VT_evn))
            new_levels_odd.append((U_odd, s_odd, VT_odd))
            i += 1

        multi_level_reconstructed_evn = reconstruct_from_levels(multi_level_decomposed_evn)
        multi_level_reconstructed_odd = reconstruct_from_levels(multi_level_decomposed_odd)

        print('\n')
        print('Oryginal even DCT coefficients:\n', evn_dct_coeffs)
        print("Multi-level reconstructed even DCT coefficients:\n", multi_level_reconstructed_evn)
        print('\n')
        print('Oryginal odd DCT coefficients:\n', odd_dct_coeffs)
        print("Multi-level reconstructed odd DCT coefficients:\n", multi_level_reconstructed_odd)

        # Reconstruct the full DCT coefficients
        evn_len = len(evn_dct_coeffs)
        odd_len = len(odd_dct_coeffs)

        assert len(evn_dct_coeffs) == len(odd_dct_coeffs), "Błąd: Nieparzysta liczba współczynników!"

        coefficients = []
        for i in range(evn_len + odd_len):
            if i % 2 == 0:
                coefficients.append(multi_level_reconstructed_evn[i // 2])
            else:
                coefficients.append(multi_level_reconstructed_odd[i // 2])
        print('\n')


        print("Oryginal DCT coefficients:\n", filtered_dct_coeffs)
        coefficients = np.array(coefficients)
        print("Reconstructed DCT coefficients:\n", coefficients)

        print('\n')

        new_frame_coeffs = frame_dct.copy()
        new_frame_coeffs[idx_low:idx_high] = coefficients

        # Reconstruct the full signal
        reconstructed_frame = idct(new_frame_coeffs, norm='ortho')

        print("Oryginal voiced frame:\n", voiced_frame)

        print("Reconstructed frame:\n", reconstructed_frame)

        print('\n')

        print('DIFFERENCE BETWEEN ORIGINAL AND RECONSTRUCTED FRAME:\n', voiced_frame - reconstructed_frame)

        print('\n')

        print('DIFFERENCE BETWEEN ORIGINAL AND RECONSTRUCTED FRAME (ABSOLUTE):\n', np.abs(voiced_frame - reconstructed_frame))

        print('\n')

        print('AVG DIFFERENCE BETWEEN ORIGINAL AND RECONSTRUCTED FRAME:\n', np.mean(np.abs(voiced_frame - reconstructed_frame)))

        print('\n')


