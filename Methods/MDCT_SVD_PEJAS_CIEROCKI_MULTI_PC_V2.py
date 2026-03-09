import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from timeit import default_timer as timer

from lib.Method import DIFFERENTIAL_SVD_XUE_BASED_METHOD

class MDCT_SVD_PEJAS_CIEROCKI_MULTI_PC_V2(DIFFERENTIAL_SVD_XUE_BASED_METHOD):
    
    def __init__(self, method_arguments, logger=None):

        super().__init__(method_arguments, logger)

        self.levels = method_arguments.get('LEVELS', 0)

        self.name = f"MDCT_SVD_PEJAS_CIEROCKI_MULTI_PC_V2_{self.alpha}_B_{self.beta}_{self.where_to_hide}_FREQ_L_{self.FREQ_L}_FREQ_H_{self.FREQ_H}_FL_{self.frame_length}"

        self.domain = 'FREQUENCY'
        self.style = 'SEQUENTIAL'

        self.normal_name = 'PC MDCT [v2]'

    def calculate_capacity(self, carrier):
        return len(carrier)

    def embedd_in_frame(self, frame, bit, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        bit : int (0 lub 1) — bit do osadzenia
        fs : int — częstotliwość próbkowania
        """
        
        window = self.MDCTProcessor.kaiser_derived(len(frame), beta = 4.0)
        frame = frame * window

        self.logger.debug(f"Frame after windowing: \n{frame}")
        
        # MDCT
        frame_mdct = self.MDCTProcessor.mdct(frame)

        self.logger.debug(f"Frame after MDCT: \n{frame_mdct}")

        # Filtracja w paśmie
        filtered_mdct_coeffs, idx_low, idx_high = self.MDCTProcessor.filter_mdct_coeffs(
            frame_mdct, fs, len(frame_mdct), self.FREQ_L, self.FREQ_H
        )

        self.logger.debug(f"Filtered MDCT coefficients: \n{filtered_mdct_coeffs}")

        # Podział na parzyste i nieparzyste
        evn_dct_coeffs = filtered_mdct_coeffs[::2]
        odd_dct_coeffs = filtered_mdct_coeffs[1::2]

        evn_matrix, padded_evn, evn_shape = self.vector_to_rectangular_with_padding(evn_dct_coeffs)
        odd_matrix, padded_odd, odd_shape = self.vector_to_rectangular_with_padding(odd_dct_coeffs)

        self.logger.debug(f"Even matrix: \n{evn_matrix}")
        self.logger.debug(f"Odd matrix: \n{odd_matrix}")

        self.logger.debug(f"Even matrix shape: {evn_shape}, Odd matrix shape: {odd_shape}")
        self.logger.debug(f"Padded even coefficients: \n{padded_evn}")
        self.logger.debug(f"Padded odd coefficients: \n{padded_odd}")

        # SVD
        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        i = 0

        new_levels_evn = []
        new_levels_odd = []

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                self.s_even.append(ro_11)
                self.s_odd.append(ro_21)

                T_bu_1 = self.alpha + 2.0*(ro_21 - ro_11)/(ro_21 + ro_11) 
                T_bu_2 = self.beta + 2.0*(ro_11 - ro_21)/(ro_21 + ro_11)

                self.alphas.append(T_bu_1)
                self.betas.append(T_bu_2)

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

                self.new_s_even.append(ro_dash_11)
                self.new_s_odd.append(ro_dash_21)

            new_levels_evn.append((U_evn, s_evn, VT_evn))
            new_levels_odd.append((U_odd, s_odd, VT_odd))
            i += 1

        evn_dct_coeffs = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_evn)
        odd_dct_coeffs = self.MultiLevelSVDProcessor.reconstruct_from_levels(new_levels_odd)

        self.logger.debug(f"Inverse even coefficients: \n{evn_dct_coeffs}")
        self.logger.debug(f"Inverse odd coefficients: \n{odd_dct_coeffs}")

        # Scalanie
        merged_coeffs = np.empty(len(evn_dct_coeffs) + len(odd_dct_coeffs), dtype=evn_dct_coeffs.dtype)
        merged_coeffs[::2] = evn_dct_coeffs
        merged_coeffs[1::2] = odd_dct_coeffs

        self.logger.debug(f"Merged coefficients: \n{merged_coeffs}")

        # Wstawienie
        frame_mdct[idx_low:idx_high] = merged_coeffs

        self.logger.debug(f"Frame MDCT after embedding: \n{frame_mdct}")

        # IMDCT
        embedded_frame = self.MDCTProcessor.imdct(frame_mdct)

        self.logger.debug(f"Embedded frame after IMDCT: \n{embedded_frame}")

        # Okno syntezy
        embedded_frame = embedded_frame * window

        self.logger.debug(f"Embedded frame after windowing: \n{embedded_frame}")

        return embedded_frame

    def extract_from_frame(self, frame, fs):
        """
        frame : ndarray (int16) — sygnał w domenie czasowej
        fs : int — częstotliwość próbkowania
        """
        # Konwersja do float [-1, 1], jeśli frame jest int16
        # if frame.dtype == np.int16:
        #     frame = frame.astype(np.float32) / 32768.0

        window = self.MDCTProcessor.kaiser_derived(len(frame), beta=4.0)
        frame = frame * window
        
        # MDCT
        frame_mdct = self.MDCTProcessor.mdct(frame)

        # Filtracja w paśmie
        filtered_mdct_coeffs, idx_low, idx_high = self.MDCTProcessor.filter_mdct_coeffs(
            frame_mdct, fs, len(frame_mdct), self.FREQ_L, self.FREQ_H
        )

        # Podział na parzyste i nieparzyste
        evn_dct_coeffs = filtered_mdct_coeffs[::2]
        odd_dct_coeffs = filtered_mdct_coeffs[1::2]

        # Konwersja do macierzy
        levels_evn = self.MultiLevelSVDProcessor.multi_level_decompose(evn_dct_coeffs, self.levels)
        levels_odd = self.MultiLevelSVDProcessor.multi_level_decompose(odd_dct_coeffs, self.levels)

        i = 0

        for (U_evn, s_evn, VT_evn), (U_odd, s_odd, VT_odd) in zip(levels_evn, levels_odd):

            if i == self.levels - 1:

                s_evn_max_idx = np.argmax(s_evn)
                s_odd_max_idx = np.argmax(s_odd)

                ro_11 = s_evn[s_evn_max_idx]
                ro_21 = s_odd[s_odd_max_idx]

                dj = ro_11 - ro_21

                if dj > 0:
                    return 0
                else:
                    return 1

            i += 1
    
    # Wewnątrz klasy MDCT_SVD_WEN:

    def embedding(self):
        """
        Nadpisana metoda embedding specyficznie dla metody WEN (MDCT).
        Wymaga oknowania również ramek pomijanych (unvoiced) dla zachowania ciągłości OLA.
        """
        self.logger.info("########### Starting embedding process (MDCT_SVD_WEN) ###########")

        embedded_frames = []
        start_embedding = timer()

        i = 0
        index = 0
        
        # Parametr beta musi być zgodny z tym użytym w embedd_in_frame
        # W Twoim kodzie MDCT_SVD_WEN używasz beta=4.0
        BETA_PARAM = 4.0

        for frame, is_voiced in zip(self.frames_to_use, self.frames_voice_status):

            if is_voiced:
                # --- RAMKI DŹWIĘCZNE (OSADZANIE) ---
                if i < len(self.secret.binary_data):
                    secret_bit = self.secret.binary_data[i]
                    self.logger.debug(f"Embedding bit {secret_bit} in frame idx: {index}")

                    # Ta metoda wewnątrz robi: frame * window -> MDCT -> SVD -> ... -> IMDCT -> * window
                    # Więc zwraca ramkę już "podwójnie" oknowaną (analiza + synteza)
                    embedded_frame = self.embedd_in_frame(frame, secret_bit, self.carrier.fs)
                    embedded_frames.append(embedded_frame)
                    i += 1
                else:
                    self.logger.debug(f"No more bits to embed, using original frame at index {index}")
                    
                    # POPRAWKA DLA NADMIAROWYCH RAMEK DŹWIĘCZNYCH
                    # Musimy je przepuścić przez proces oknowania, żeby pasowały do OLA
                    window = self.MDCTProcessor.kaiser_derived(len(frame), beta=BETA_PARAM)
                    processed_frame = frame * window # Analiza
                    processed_frame = processed_frame * window # Synteza
                    embedded_frames.append(processed_frame)

            else:
                # --- RAMKI BEZDŹWIĘCZNE / CISZA (POMIJANIE) ---
                self.logger.debug(f"Using unvoiced frame at index {index} without embedding")

                # POPRAWKA KLUCZOWA DLA METODY WEN (MDCT)
                # Zamiast wstawiać surową ramkę, symulujemy przejście przez MDCT/IMDCT
                # poprzez nałożenie okien analizy i syntezy.
                
                window = self.MDCTProcessor.kaiser_derived(len(frame), beta=BETA_PARAM)
                
                # Okno analizy (symulacja wejścia do MDCT)
                processed_frame = frame * window 
                
                # Okno syntezy (symulacja wyjścia z IMDCT)
                processed_frame = processed_frame * window 
                
                embedded_frames.append(processed_frame)

            index += 1

        stop_embedding = timer()

        self.logger.info("########### Finished embedding process ###########")
        self.embedding_time = stop_embedding - start_embedding

        # Rekonstrukcja sygnału (OLA)
        stegobject = self.reconstruct_signal_from_frames(embedded_frames)

        return stegobject.astype(np.int16)