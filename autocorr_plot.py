import numpy as np
import matplotlib.pyplot as plt
import os
import time

def sliding_autocorr_power_metric_safe(r, D=16, Nwin=48, epsilon=1e-6):
    s2 = np.concatenate((np.zeros(D, dtype=r.dtype), r[:-D]))
    prod = r * np.conjugate(s2)

    cs_prod = np.concatenate(([0], np.cumsum(prod)))
    cs_power = np.concatenate(([0], np.cumsum(np.abs(r)**2)))

    C = cs_prod[Nwin:] - cs_prod[:-Nwin]
    P = cs_power[Nwin:] - cs_power[:-Nwin]

    M = np.abs(C) / np.maximum(P, epsilon)  # evita divisão por zero
    return M, np.abs(C), P

if __name__ == "__main__":
    base_dir = "/home/diogod/cdig/gr-ieee802-11/project"
    recordings_dir = os.path.join(base_dir, "recordings")
    filename = "Sample3_20MHz_Channel100.bin"
    filepath = os.path.join(recordings_dir, filename)

    print(f"Lendo arquivo: {filepath}")
    r = np.fromfile(filepath, dtype=np.complex64)
    print(f"Número total de amostras lidas: {len(r)}")

    max_samples = 5_000_000
    if len(r) > max_samples:
        print(f"Reduzindo para as primeiras {max_samples} amostras para evitar OOM.")
        r = r[:max_samples]

    print(f"Tamanho do vetor para processamento: {len(r)}")

    D = 16
    Nwin = 48
    epsilon = 1e-6  # limiar para evitar divisão por zero

    start = time.time()
    M, Cmag, P = sliding_autocorr_power_metric_safe(r, D=D, Nwin=Nwin, epsilon=epsilon)
    print(f"Cálculo concluído em {time.time() - start:.2f} segundos")

    peak_index = np.argmax(M)
    print(f"Pico máximo da métrica em índice: {peak_index}")

    window = 200
    start_sample = max(0, peak_index - window)
    end_sample = min(len(M), peak_index + window)

    # Plotar potência para análise
    plt.figure(figsize=(10,4))
    plt.plot(P)
    plt.title('Potência P(n) ao longo das amostras')
    plt.xlabel('Índice da amostra')
    plt.ylabel('Potência')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico 1: Métrica clipada para [0,1]
    M_clip = np.clip(M[start_sample:end_sample], 0, 1)
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(start_sample, end_sample), M_clip, label='Métrica Normalizada (clipada)')
    plt.title('Autocorrelação Normalizada - Valores clipados a 1')
    plt.xlabel('Índice da amostra (n)')
    plt.ylabel('M(n)')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Métrica em escala logarítmica (dB)
    M_dB = 10 * np.log10(M[start_sample:end_sample] + 1e-12)
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(start_sample, end_sample), M_dB, label='Métrica Normalizada (dB)')
    plt.title('Autocorrelação Normalizada - Escala logarítmica (dB)')
    plt.xlabel('Índice da amostra (n)')
    plt.ylabel('M(n) [dB]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
