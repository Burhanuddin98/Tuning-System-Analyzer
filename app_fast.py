
import io
import math
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import stft, find_peaks, butter, sosfilt, resample_poly

from tuning_generator import pack_defaults

st.set_page_config(page_title="Tuning & Scale Finder — Fast", layout="wide")
st.title("🎼 Tuning & Scale Finder — Fast mode (coarse→fine, subsampling)")

with st.sidebar:
    st.header("Speed/Accuracy Controls")
    # Preprocessing
    target_sr = st.selectbox("Target sample rate", [8000, 12000, 16000, 22050], index=2)
    max_secs   = st.slider("Max audio length (s)", 4, 20, 10, step=1)
    # STFT
    frame_ms = st.slider("STFT window (ms)", 32, 64, 48, step=4)
    hop_ms   = st.slider("Hop (ms)", 8, 32, 24, step=4)
    top_peaks = st.slider("Top peaks per processed frame", 1, 6, 3, step=1)
    frame_stride = st.slider("Process every Nth frame", 1, 8, 4, step=1)
    fmin = st.number_input("Min freq (Hz)", 30.0, 400.0, 60.0, step=5.0)
    fmax = st.number_input("Max freq (Hz)", 1000.0, 6000.0, 2000.0, step=50.0)
    min_mag_db = st.slider("Min peak mag (dB)", -120, -10, -40, step=5)
    # A4 search
    a4_lo  = st.number_input("A4 coarse lo (Hz)", 400.0, 480.0, 430.0, step=0.5)
    a4_hi  = st.number_input("A4 coarse hi (Hz)", 420.0, 500.0, 450.0, step=0.5)
    a4_step_coarse = st.number_input("A4 coarse step (Hz)", 0.2, 5.0, 1.0, step=0.2)
    a4_step_fine   = st.number_input("A4 fine step (Hz)", 0.05, 1.0, 0.1, step=0.05)
    fine_span_hz   = st.number_input("A4 fine ± span (Hz)", 0.5, 5.0, 2.0, step=0.5)
    limit_systems = st.multiselect(
        "Systems to check (fewer = faster)",
        options=list(pack_defaults().keys()),
        default=["12-EDO","Pythagorean_12","Meantone_12_0.25comma","JI_5limit_chromatic_12"]
    )

systems_all = pack_defaults()
systems = {k: systems_all[k] for k in limit_systems}

uploaded = st.file_uploader("Upload WAV/FLAC/OGG", type=["wav","flac","ogg"])

def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin)
    return 20.0 * np.log10(x / ref)

def highpass(y, sr, fc=40.0, order=4):
    sos = butter(order, fc/(sr/2.0), btype='highpass', output='sos')
    return sosfilt(sos, y)

def smart_segment(y, sr, max_seconds=10):
    """Pick the highest-energy window up to max_seconds."""
    N = int(max_seconds * sr)
    if len(y) <= N:
        return y
    # energy via squared amplitude moving sum
    win = int(0.050 * sr)  # 50 ms
    if win < 1: win = 1
    s = y**2
    c = np.convolve(s, np.ones(win, dtype=float), mode='same')
    # choose center of max-energy region
    center = int(np.argmax(c))
    start = max(0, center - N//2)
    end = min(len(y), start + N)
    return y[start:end]

def extract_peaks_fast(y, sr, frame_ms, hop_ms, topk, stride, fmin, fmax, min_db):
    nperseg = int(sr * frame_ms/1000.0)
    noverlap = nperseg - int(sr * hop_ms/1000.0)
    noverlap = max(0, noverlap)
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = np.abs(Zxx)
    mag_db = amplitude_to_db(S, ref=np.max(S) + 1e-12)
    mask = (f >= fmin) & (f <= fmax)
    fbin = f[mask]
    out_f, out_m = [], []
    for ti in range(0, mag_db.shape[1], stride):
        spec = mag_db[mask, ti]
        if spec.size < 5:
            continue
        peaks, props = find_peaks(spec, height=min_db)
        if peaks.size == 0:
            continue
        heights = props["peak_heights"]
        idx = np.argsort(heights)[-topk:]
        sel = peaks[idx]
        out_f.extend(fbin[sel].tolist())
        out_m.extend(heights[idx].tolist())
    if not out_f:
        return np.array([]), np.array([])
    # Deduplicate by rounding pitch class to nearest 5 cents
    # Convert to cents vs 440 to dedup across time
    pc = (1200.0 * np.log2(np.array(out_f)/440.0)) % 1200.0
    bins = np.round(pc / 5.0) * 5.0
    # keep max magnitude per bin
    out = {}
    for fval, mval, b in zip(out_f, out_m, bins):
        if b not in out or mval > out[b][1]:
            out[b] = (fval, mval)
    f_final = np.array([v[0] for v in out.values()])
    m_final = np.array([v[1] for v in out.values()])
    return f_final, m_final

def circular_distance(a, b):
    d = np.abs(a - b) % 1200.0
    return np.minimum(d, 1200.0 - d)

def score_for_a4(obs_pc, weights, system_cents):
    # Broadcast distances to grid, take per-observation min
    diffs = np.abs(obs_pc[:, None] - system_cents[None, :]) % 1200.0
    diffs = np.minimum(diffs, 1200.0 - diffs)
    dmin = diffs.min(axis=1)
    # simple weighted median approximation via sorting
    order = np.argsort(dmin)
    w_sorted = weights[order]
    d_sorted = dmin[order]
    cumw = np.cumsum(w_sorted)
    med_idx = np.searchsorted(cumw, cumw[-1] / 2.0)
    return float(d_sorted[min(med_idx, len(d_sorted)-1)])

def coarse_to_fine(observed_freqs, observed_mags, systems_dict, a4_lo, a4_hi, step_coarse, step_fine, fine_span):
    # Precompute weights & normalize
    weights = np.maximum(observed_mags - observed_mags.min(), 1.0)
    results = []
    # Coarse pass
    coarse_candidates = []
    a_vals = np.arange(a4_lo, a4_hi + 1e-9, step_coarse, dtype=float)
    for name, sys in systems_dict.items():
        cents_grid = np.array([n["cents"] for n in sys["notes"]], dtype=float)
        best = (1e9, None)  # (mad, a4)
        for a in a_vals:
            obs_pc = (1200.0 * np.log2(observed_freqs / a)) % 1200.0
            mad = score_for_a4(obs_pc, weights, cents_grid)
            if mad < best[0]:
                best = (mad, a)
        coarse_candidates.append((name, best[1], best[0], cents_grid))
    # Keep best 2 systems for refinement
    coarse_candidates.sort(key=lambda x: x[2])
    top_refine = coarse_candidates[:2]

    # Fine pass
    for name, a_coarse, _, cents_grid in top_refine:
        a_vals_fine = np.arange(a_coarse - fine_span, a_coarse + fine_span + 1e-9, step_fine, dtype=float)
        best = (1e9, None)
        for a in a_vals_fine:
            obs_pc = (1200.0 * np.log2(observed_freqs / a)) % 1200.0
            mad = score_for_a4(obs_pc, weights, cents_grid)
            if mad < best[0]:
                best = (mad, a)
        results.append({"name": name, "a4": float(best[1]), "mad_cents": float(best[0])})

    results.sort(key=lambda x: x["mad_cents"])
    return results

uploaded_info = st.empty()

if uploaded is not None:
    data = uploaded.read()
    try:
        y, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception as e:
        st.error(f"Failed to decode audio: {e}")
        st.stop()

    # Preprocess: HPF, resample, smart segment
    y = highpass(y, sr, 40.0)
    y = smart_segment(y, sr, max_seconds=max_secs)
    y = resample_poly(y, target_sr, sr)
    sr = target_sr

    uploaded_info.info(f"Using {len(y)/sr:.2f}s @ {sr} Hz after preprocessing.")

    # Extract peaks quickly
    freqs, mags = extract_peaks_fast(
        y, sr, frame_ms=frame_ms, hop_ms=hop_ms, topk=top_peaks,
        stride=frame_stride, fmin=fmin, fmax=fmax, min_db=min_mag_db
    )
    if freqs.size < 8:
        st.warning("Too few peaks after preprocessing. Try longer snippet or relax thresholds.")
        st.stop()

    # Coarse→fine search
    best = coarse_to_fine(freqs, mags, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)

    st.subheader("Best matches")
    for i, r in enumerate(best, 1):
        st.markdown(f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, error ≈ **{r['mad_cents']:.1f}¢**")

    # Quick polar scatter via Plotly (lightweight)
    try:
        import plotly.graph_objects as go
        cents = (1200.0 * np.log2(freqs / best[0]['a4'])) % 1200.0
        mags_n = (mags - mags.min()) / (mags.max() - mags.min() + 1e-9)
        r = 0.2 + 0.8 * mags_n
        fig = go.Figure(go.Scatterpolar(theta=cents, r=r, mode='markers', opacity=0.85))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1.0]),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[k*100 for k in range(12)],
                    ticktext=[f"{k*100}¢" for k in range(12)],
                    direction="clockwise"
                ),
            ),
            showlegend=False,
            margin=dict(l=10,r=10,t=40,b=10),
            title="Polar Pitch-Class Map"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Plotly not installed; skipping polar plot.")
else:
    st.info("Upload a short, sustained passage (4–10 s). The app will auto-pick the most informative segment.")
