import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.linalg import svd

# --- 1. Generowanie syntetycznej ramki mowy dźwięcznej (quasi-okresowej) ---
# Używamy sumy harmonicznych, aby symulować fragment 25 ms z silną korelacją lokalną
fs = 16000
duration = 0.025 # 25 ms
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f0 = 120 # Częstotliwość podstawowa (typowa dla męskiego głosu)

# Sygnał: f0 + harmoniczne + szum
signal = (np.sin(2 * np.pi * f0 * t) +
          0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
          0.25 * np.sin(2 * np.pi * 3 * f0 * t) +
          0.05 * np.random.randn(len(t)))

# --- 2. Transformata i ekstrakcja (zgodnie z metodologią) ---
dct_coeffs = dct(signal, norm='ortho')

# Wycinamy pasmo i wybieramy podciąg (np. parzyste współczynniki)
filtered_dct = dct_coeffs[20:120]
subsequence = filtered_dct[::2]

# --- 3. Budowa macierzy ---
def vector_to_rectangular_with_padding(vector):
    """Przekształcenie arbitralne z paddingiem zerowym"""
    length = len(vector)
    dim = int(np.ceil(np.sqrt(length)))
    pad_size = dim * dim - length
    padded_vector = np.pad(vector, (0, pad_size), 'constant')
    return padded_vector.reshape((dim, dim))

def hankel_trajectory_matrix(vector, L):
    """Budowa macierzy trajektorii zgodnie z ideą SSA"""
    vector = np.asarray(vector)
    K = len(vector) - L + 1
    # Zwraca macierz L x K (gdzie L to okno, a K to liczba przesunięć)
    return np.vstack([vector[i:i + K] for i in range(L)])

# Rozmiar dla macierzy prostokątnej narzuca się sam (pierwiastek)
rect_matrix = vector_to_rectangular_with_padding(subsequence)

# Dla macierzy Hankela (Macierz Trajektorii w SSA) optymalnie wybierać L wokół N/2
# Aby zachować uczciwe ramy porównania, niech L będzie zbliżone do wymiaru macierzy prostokątnej
L = int(np.ceil(np.sqrt(len(subsequence))))
hankel_matrix = hankel_trajectory_matrix(subsequence, L)

# --- 4. Dekompozycja SVD ---
_, s_rect, _ = svd(rect_matrix)
_, s_hankel, _ = svd(hankel_matrix)

# Normalizacja widma wartości osobliwych do maksymalnej wartości
s_rect_norm = s_rect / s_rect[0]
s_hankel_norm = s_hankel / s_hankel[0]

# Obliczenie skumulowanej frakcji energii (Cumulative Energy)
energy_rect = np.cumsum(s_rect**2) / np.sum(s_rect**2)
energy_hankel = np.cumsum(s_hankel**2) / np.sum(s_hankel**2)

# --- 5. Rysowanie Wykresów Dowodowych ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Wykres A: Szybkość spadku wartości osobliwych (skala logarytmiczna)
ax1.plot(s_rect_norm, 'b--o', label='Niestrukturyzowana (Prostokątna)', alpha=0.8)
ax1.plot(s_hankel_norm, 'r-o', label='Strukturyzowana (Hankela)', alpha=0.8, linewidth=2)
ax1.set_yscale('log')
ax1.set_title("Widmo wartości osobliwych (skala log)")
ax1.set_xlabel("Indeks wartości osobliwej (i)")
ax1.set_ylabel(r"$\sigma_i / \sigma_1$")
ax1.grid(True, alpha=0.4, which="both", ls="--")
ax1.legend()

# Wykres B: Skumulowana energia (SSA Information Content)
ax2.plot(energy_rect, 'b--o', label='Niestrukturyzowana (Prostokątna)', alpha=0.8)
ax2.plot(energy_hankel, 'r-o', label='Strukturyzowana (Hankela)', alpha=0.8, linewidth=2)
ax2.axhline(0.95, color='gray', linestyle=':', label='Próg 95% energii')
ax2.set_title("Skumulowana frakcja energii w składowych główych")
ax2.set_xlabel("Liczba składowych (r)")
ax2.set_ylabel("Suma kwadratów wariancji (skumulowana)")
ax2.grid(True, alpha=0.4)
ax2.legend(loc="lower right")

plt.tight_layout()
output_file = "svd_decay_proof.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Zapisano dowód graficzny: {output_file}")