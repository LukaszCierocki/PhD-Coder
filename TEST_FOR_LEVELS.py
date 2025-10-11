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
    for _ in range(num_levels):
        H = hankel_k_row(current, 2)
        U, s, Vh = svd(H, full_matrices=False)
        levels.append((U, s, Vh))
        Ha = s[0] * np.outer(U[:, 0], Vh[0, :])
        current = diagonal_average(Ha)
    return levels

def reconstruct_from_levels(levels):
    U, s, Vh = levels[-1]
    Ha = s[0] * np.outer(U[:, 0], Vh[0, :])
    Hd = s[1] * np.outer(U[:, 1], Vh[1, :])
    currentL = diagonal_average(Ha + Hd)
    for U, s, Vh in reversed(levels[:-1]):
        Hd = s[1] * np.outer(U[:, 1], Vh[1, :])
        currentR = diagonal_average(Hd)
        Hd_reconstructed = hankel_k_row(currentR, 2)
        Ha_reconstructed = hankel_k_row(currentL, 2)
        H = Ha_reconstructed + Hd_reconstructed
        currentL = diagonal_average(H)
    return currentL

def filter_dct_coeffs(coeffs: np.ndarray, Fs: float, low_freq: float, high_freq: float):
    N = len(coeffs)
    k_low = int(low_freq / (Fs / 2) * N)
    k_high = int(high_freq / (Fs / 2) * N)
    k_low = max(0, k_low)
    k_high = min(N, k_high)
    return coeffs[k_low:k_high], k_low, k_high

if __name__ == "__main__":
    import pandas as pd
    import random

    tools = Tools(logging.DEBUG)
    logger = tools.get_logger()

    carriers = [tools.load_carrier('arctic_a0001')]
    carrier_data = carriers[0].data_signed_int16
    carrier_fs = carriers[0].fs

    frame_processor = FrameProcessor(logger)
    voiced_frames, _, _, _, _ = frame_processor.divide_into_voiced_unvoiced_frames(
        'rVAD', carrier_data, carrier_fs, 10, 10
    )

    alpha = 0.7
    beta = 0.7
    levels_to_test = range(1, 9)

    selected_levels = []
    original_bits = []
    extracted_bits = []
    robustness_scores_0 = []
    robustness_scores_1 = []

    for frame_id, voiced_frame in enumerate(voiced_frames):
        frame_dct = dct(voiced_frame, norm='ortho')
        filtered_coeffs, k_low, k_high = filter_dct_coeffs(frame_dct, carrier_fs, 1500, 2500)
        evn_dct = filtered_coeffs[::2]
        odd_dct = filtered_coeffs[1::2]

        best_level = None
        best_bit = None
        best_coeffs = None
        best_score = -1
        best_avg_diff = float('inf')
        best_r0 = 0
        best_r1 = 0

        for level in levels_to_test:
            bit = random.randint(0, 1)
            evn_lvls = multi_level_decompose(evn_dct, level)
            odd_lvls = multi_level_decompose(odd_dct, level)

            Ue, se, VTe = evn_lvls[-1]
            Uo, so, VTo = odd_lvls[-1]

            idx_e = np.argmax(se)
            idx_o = np.argmax(so)
            ro_11 = se[idx_e]
            ro_21 = so[idx_o]
            davg = (ro_11 + ro_21) / 2
            T_bu_1 = alpha + 2.0 * (ro_21 - ro_11) / (ro_21 + ro_11)
            T_bu_2 = beta + 2.0 * (ro_11 - ro_21) / (ro_21 + ro_11)

            if bit == 0:
                if ro_11 - ro_21 >= T_bu_1 * davg:
                    ro_dash_11 = ro_11
                    ro_dash_21 = ro_21
                else:
                    ro_dash_11 = ro_21 + T_bu_1 * davg / 2
                    ro_dash_21 = ro_21 - T_bu_1 * davg / 2
            else:
                if ro_21 - ro_11 >= T_bu_2 * davg:
                    ro_dash_11 = ro_11
                    ro_dash_21 = ro_21
                else:
                    ro_dash_11 = ro_11 - T_bu_2 * davg / 2
                    ro_dash_21 = ro_21 + T_bu_2 * davg / 2

            se[idx_e] = ro_dash_11
            so[idx_o] = ro_dash_21
            evn_lvls[-1] = (Ue, se, VTe)
            odd_lvls[-1] = (Uo, so, VTo)

            rec_evn = reconstruct_from_levels(evn_lvls)
            rec_odd = reconstruct_from_levels(odd_lvls)

            recombined = []
            for i in range(len(filtered_coeffs)):
                recombined.append(rec_evn[i // 2] if i % 2 == 0 else rec_odd[i // 2])
            recombined = np.array(recombined)

            frame_dct_modified = frame_dct.copy()
            frame_dct_modified[k_low:k_high] = recombined
            rec_frame = idct(frame_dct_modified, norm='ortho')
            avg_diff = np.mean(np.abs(voiced_frame - rec_frame)) + 1e-6

            r_0 = max(0, ro_dash_11 - ro_dash_21) / avg_diff
            r_1 = max(0, ro_dash_21 - ro_dash_11) / avg_diff

            current_score = r_0 if bit == 0 else r_1

            if current_score > best_score:
                best_score = current_score
                best_level = level
                best_bit = bit
                best_coeffs = recombined
                best_r0 = r_0
                best_r1 = r_1

        selected_levels.append(best_level)
        original_bits.append(best_bit)
        robustness_scores_0.append(best_r0)
        robustness_scores_1.append(best_r1)

        evn_dct_extr = best_coeffs[::2]
        odd_dct_extr = best_coeffs[1::2]

        evn_lvls_extr = multi_level_decompose(evn_dct_extr, best_level)
        odd_lvls_extr = multi_level_decompose(odd_dct_extr, best_level)

        s_evn_max = np.max(evn_lvls_extr[-1][1])
        s_odd_max = np.max(odd_lvls_extr[-1][1])
        extracted_bit = 0 if s_evn_max - s_odd_max > 0 else 1
        extracted_bits.append(extracted_bit)

    errors = sum(o != e for o, e in zip(original_bits, extracted_bits))
    ber = (errors / len(original_bits)) * 100

    print(f"BER przy adaptacyjnym poziomie osadzania: {ber:.2f}%")

    df = pd.DataFrame({
        "frame": list(range(len(original_bits))),
        "level_used": selected_levels,
        "bit_original": original_bits,
        "bit_extracted": extracted_bits,
        "robustness_score_0": robustness_scores_0,
        "robustness_score_1": robustness_scores_1,
        "error": [o != e for o, e in zip(original_bits, extracted_bits)]
    })
    df.to_csv("adaptive_embedding_robustness.csv", index=False)
    print("Zapisano wyniki do adaptive_embedding_robustness.csv")