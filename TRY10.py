import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# --- CONFIGURACIÓN DE ESTILO ---
st.set_page_config(page_title="Monitor EEG Pro", layout="wide")
plt.style.use('dark_background')

class EEGAdvancedSimulator:
    def __init__(self, fs=128, window_sec=8):
        self.fs = fs
        self.window_sec = window_sec
        self.points = fs * window_sec
        self.bs_timer = 0
        
        # Mapa Maestro de Bandas (Colores Premium)
        self.band_defs = [
            {"name": "SLOW",  "f_range": (0.1, 1.5),  "f_init": 0.8, "a_init": 80.0, "color": "#94E0ED", "text_color": "#8B0000"},
            {"name": "DELTA", "f_range": (1.5, 4.0),  "f_init": 2.5, "a_init": 60.0, "color": "#FF0000", "text_color": "#FF4B4B"},
            {"name": "THETA", "f_range": (4.0, 8.0),  "f_init": 6.0, "a_init": 30.0, "color": "#FF8800", "text_color": "#FFAA55"},
            {"name": "ALPHA", "f_range": (8.0, 13.0), "f_init": 10.0, "a_init": 50.0, "color": "#00FF88", "text_color": "#00FF7F"},
            {"name": "BETA",  "f_range": (14.0, 30.0), "f_init": 22.0, "a_init": 15.0, "color": "#0044BB", "text_color": "#5555FF"},
            {"name": "GAMMA", "f_range": (30.0, 45.0), "f_init": 35.0, "a_init": 8.0,  "color": "#CC00CC", "text_color": "#FF55FF"}
        ]
        self.n_channels = len(self.band_defs) + 1 
        self.reset_data()

    def reset_data(self):
        self.buffer = np.zeros((self.n_channels, self.points))
        self.t = np.linspace(0, self.window_sec, self.points)
        self.phases = np.zeros(len(self.band_defs))

    def get_chunk(self, freqs, amps, noise_lvl, master_amp, chunk_size, bs_active=False, bs_ratio=0.5):
        t_chunk = np.arange(chunk_size) / self.fs
        chunk_data = np.zeros((self.n_channels, chunk_size))
        sum_sig = np.zeros(chunk_size)
        
        # Lógica de Burst Suppression (Ciclo de 4 segundos)
        is_suppressed = False
        if bs_active:
            self.bs_timer = (self.bs_timer + chunk_size / self.fs) % 4.0
            if self.bs_timer > (4.0 * (1 - bs_ratio)):
                is_suppressed = True

        for i, b in enumerate(self.band_defs):
            f, a = freqs[i], amps[i]
            component = a * np.sin(2 * np.pi * f * t_chunk + self.phases[i])
            self.phases[i] = (self.phases[i] + 2 * np.pi * f * (chunk_size / self.fs)) % (2 * np.pi)
            
            val = component + np.random.normal(0, noise_lvl * 0.1, chunk_size)
            if is_suppressed: val *= 0.05 # Silencio eléctrico
            
            chunk_data[i, :] = val
            sum_sig += val

        final_eeg = (sum_sig + np.random.normal(0, noise_lvl, chunk_size)) * master_amp
        chunk_data[-1, :] = final_eeg
        return chunk_data

    def update(self, new_data):
        self.buffer = np.roll(self.buffer, -new_data.shape[1], axis=1)
        self.buffer[:, -new_data.shape[1]:] = new_data
        return self.buffer

# --- INTERFAZ ---
if "sim" not in st.session_state: st.session_state.sim = EEGAdvancedSimulator()

with st.sidebar:
    st.header("⚙️ Ajustes")
    
    # Sistema de Pausa
    if st.session_state.get("run", True):
        if st.button("⏸️ PAUSAR MONITOR", use_container_width=True):
            st.session_state.run = False; st.rerun()
    else:
        if st.button("▶️ REANUDAR MONITOR", use_container_width=True, type="primary"):
            st.session_state.run = True; st.rerun()

    st.markdown("---")
    st.subheader("Simulación Clínica")
    bs_on = st.toggle("Activar Supresión de Brotes (BS)")
    bs_ratio = st.slider("BSR (Nivel de supresión)", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    noise_val = st.slider("Ruido", 0.0, 30.0, 5.0)
    gain = st.slider("Escala", 0.1, 2.0, 1.0)
    
    current_freqs, current_amps = [], []
    for b in st.session_state.sim.band_defs:
        with st.expander(f"Banda {b['name']}"):
            f = st.slider("Hz", b['f_range'][0], b['f_range'][1], float(b['f_init']), 0.1, key=f"f_{b['name']}")
            a = st.slider("µV", 0.0, 150.0, float(b['a_init']), 1.0, key=f"a_{b['name']}")
            current_freqs.append(f); current_amps.append(a)

st.title("Monitor")

c_eeg = st.empty(); c_mid = st.empty(); c_bot = st.empty()

while st.session_state.get("run", True):
    data = st.session_state.sim.update(st.session_state.sim.get_chunk(current_freqs, current_amps, noise_val, gain, 32, bs_on, bs_ratio))
    t = st.session_state.sim.t
    eeg_main = data[-1]

    # 1. MONITOR EEG (Crudo)
    with c_eeg:
        fig1, ax1 = plt.subplots(figsize=(12, 2.5))
        ax1.plot(t, eeg_main, color='#00FF00', lw=1.5) 
        ax1.set_ylim((-350, 350)); ax1.set_xlim((0, 8))
        ax1.grid(True, color='#222222', lw=0.5)
        ax1.text(0.1, -300, "50µV | 1s", color='yellow', fontsize=9, weight='bold')
        ax1.set_title("EEG CORTICAL INTEGRADO", color='gray', fontsize=9)
        st.pyplot(fig1); plt.close(fig1)

    # 2. DSA Y POTENCIA (Limpio)
    with c_mid:
        fig2, (ax_dsa, ax_psd) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})
        
        # DSA
        f_dsa, t_dsa, Sxx = signal.spectrogram(eeg_main, st.session_state.sim.fs, nperseg=128, noverlap=112)
        ax_dsa.pcolormesh(t_dsa, f_dsa, 10 * np.log10(Sxx + 1e-10), cmap='jet', vmin=-10, vmax=35, shading='gouraud')
        ax_dsa.set_ylim(0, 45); ax_dsa.set_title("DSA", color='white', fontsize=8)

        # PSD (Potencia por bandas)
        f_p, psd = signal.welch(eeg_main, st.session_state.sim.fs, nperseg=128)
        max_psd = np.max(psd)*1.2 if np.max(psd)>0 else 1
        
        for b in st.session_state.sim.band_defs:
            ax_psd.axvspan(b['f_range'][0], b['f_range'][1], color=b['color'], alpha=0.15)
            mid_f = (b['f_range'][0] + b['f_range'][1]) / 2
            ax_psd.text(mid_f, max_psd * 1.05, b['name'], color=b['text_color'], ha='center', fontsize=7, weight='bold')

        ax_psd.plot(f_p, psd, color='white', lw=2)
        ax_psd.set_xlim(0, 45); ax_psd.set_ylim(0, max_psd * 1.2)
        ax_psd.set_title("ESPECTRO DE POTENCIA", color='white', fontsize=8)
        st.pyplot(fig2); plt.close(fig2)

    # 3. INDIVIDUALES
    with c_bot:
        fig3, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
        plt.subplots_adjust(hspace=0.3)
        for i, b in enumerate(st.session_state.sim.band_defs):
            axes[i].plot(t, data[i], color=b['color'], lw=1)
            axes[i].set_ylim([-150, 150])
            axes[i].set_ylabel(b['name'], color=b['color'], fontsize=8, weight='bold', rotation=0, labelpad=25)
            axes[i].grid(True, color="#222222", lw=0.3)
            for s in axes[i].spines.values(): s.set_visible(False)
        st.pyplot(fig3); plt.close(fig3)

    time.sleep(0.01)

if not st.session_state.get("run", True):
    st.stop()