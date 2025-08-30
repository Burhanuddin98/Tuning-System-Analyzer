
import base64
import io
import json
import math
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import stft, find_peaks
from tuning_generator import pack_defaults

st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (Synced)", layout="wide")
st.title("🎼 Tuning Analyser — TRUE Spiral (live, dark‑neon)")

with st.sidebar:
    st.header("STFT")
    frame_ms = st.slider("Window (ms)", 32, 96, 64, step=4)
    hop_ms   = st.slider("Hop (ms)", 8, 48, 24, step=4)
    frame_stride = st.slider("Process every Nth frame", 1, 8, 3, step=1)
    fmin = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax = st.number_input("Max freq (Hz)", 500.0, 6000.0, 2200.0, step=50.0)
    peak_db = st.slider("Peak threshold (dB)", -120, -10, -50, step=5)
    top_peaks = st.slider("Top peaks/frame", 2, 20, 8, step=1)

    st.header("Tuning Search")
    systems_all = pack_defaults()
    default_keys = ["12-EDO","Pythagorean_12","Meantone_12_0.25comma","JI_5limit_chromatic_12"]
    choose = st.multiselect("Systems to test", list(systems_all.keys()), default=default_keys)
    a4_lo  = st.number_input("A4 low (Hz)", 400.0, 480.0, 430.0, step=0.5)
    a4_hi  = st.number_input("A4 high (Hz)", 420.0, 500.0, 450.0, step=0.5)
    a4_step_coarse = st.number_input("A4 coarse step (Hz)", 0.2, 5.0, 1.0, step=0.2)
    a4_step_fine   = st.number_input("A4 fine step (Hz)", 0.05, 1.0, 0.1, step=0.05)
    fine_span_hz   = st.number_input("A4 fine ± span (Hz)", 0.5, 5.0, 2.0, step=0.5)

    st.header("Spiral (continuous)")
    turns = st.slider("Octave span (turns)", 2, 10, 4)  # total log2 span
    bins_per_turn = st.slider("Resolution (bins per turn)", 180, 1440, 720, step=60)
    wiggle_gain = st.slider("Wiggle gain", 0.00, 0.75, 0.25, step=0.01)
    smooth_bins = st.slider("Smoothing (bins)", 1, 81, 21, step=2)
    spokes = st.slider("Reference spokes (per 100¢)", 6, 36, 12, step=1)

systems = {k: systems_all[k] for k in choose} if len(choose)>0 else systems_all

uploaded = st.file_uploader("Upload WAV/FLAC/OGG", type=["wav","flac","ogg"])

# ---------- Peak extraction ----------
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin)
    return 20.0 * np.log10(x / ref)

def extract_frame_peaks(y, sr, frame_ms, hop_ms, stride, fmin, fmax, topk, peak_db):
    nperseg = int(sr * frame_ms/1000.0)
    hop = int(sr * hop_ms/1000.0)
    noverlap = max(0, nperseg - hop)
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = np.abs(Zxx)
    mag_db = amplitude_to_db(S, ref=np.max(S) + 1e-12)
    mask = (f >= fmin) & (f <= fmax)
    fbin = f[mask]
    peaks_f, peaks_m = [], []
    t_proc = []
    for ti in range(0, mag_db.shape[1], stride):
        spec = mag_db[mask, ti]
        t_proc.append(float(t[ti]))
        if spec.size < 5:
            peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        pk, props = find_peaks(spec, height=peak_db)
        if pk.size == 0:
            peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        heights = props["peak_heights"]
        idx = np.argsort(heights)[-topk:]
        sel = pk[idx]
        peaks_f.append(fbin[sel])
        peaks_m.append(heights[idx])
    all_f = np.concatenate([p for p in peaks_f if p.size]) if any(p.size for p in peaks_f) else np.array([])
    all_m = np.concatenate([m for m in peaks_m if m.size]) if any(m.size for m in peaks_m) else np.array([])
    return np.array(t_proc), peaks_f, peaks_m, all_f, all_m

# ---------- Tuning detection ----------
def circ_dist(a, b):
    d = np.abs(a - b) % 1200.0
    return np.minimum(d, 1200.0 - d)

def score_system(obs_freqs, obs_mags, cents_grid, a4_ref):
    obs_pc = (1200.0 * np.log2(obs_freqs / a4_ref)) % 1200.0
    w = np.maximum(obs_mags - obs_mags.min(), 1.0)
    best = (1e9, 0.0)
    for off in cents_grid:
        grid = (cents_grid - off) % 1200.0
        diffs = np.abs(obs_pc[:, None] - grid[None, :]) % 1200.0
        diffs = np.minimum(diffs, 1200.0 - diffs)
        dmin = diffs.min(axis=1)
        order = np.argsort(dmin)
        w_sorted = w[order]; d_sorted = dmin[order]
        cumw = np.cumsum(w_sorted)
        med_idx = np.searchsorted(cumw, cumw[-1] / 2.0)
        mad = d_sorted[min(med_idx, len(d_sorted)-1)]
        if mad < best[0]:
            best = (mad, float(off))
    return {"offset": best[1], "mad_cents": float(best[0])}

def coarse_to_fine(obs_freqs, obs_mags, systems_dict, a4_lo, a4_hi, step_coarse, step_fine, fine_span):
    results = []
    coarse = []
    for name, sys in systems_dict.items():
        cents = np.array([n["cents"] for n in sys["notes"]], dtype=float)
        best = {"a4": None, "mad": 1e9, "offset": 0.0}
        a = a4_lo
        while a <= a4_hi + 1e-9:
            sc = score_system(obs_freqs, obs_mags, cents, a)
            if sc["mad_cents"] < best["mad"]:
                best = {"a4": float(a), "mad": sc["mad_cents"], "offset": sc["offset"]}
            a += step_coarse
        coarse.append((name, best))
    coarse.sort(key=lambda x: x[1]["mad"])
    for name, bestc in coarse[:2]:
        cents = np.array([n["cents"] for n in systems_dict[name]["notes"]], dtype=float)
        a_vals = np.arange(bestc["a4"] - fine_span, bestc["a4"] + fine_span + 1e-9, step_fine, dtype=float)
        best = {"a4": None, "mad": 1e9, "offset": 0.0}
        for a in a_vals:
            sc = score_system(obs_freqs, obs_mags, cents, a)
            if sc["mad_cents"] < best["mad"]:
                best = {"a4": float(a), "mad": sc["mad_cents"], "offset": sc["offset"]}
        results.append({"name": name, "a4": best["a4"], "offset": best["offset"], "mad_cents": best["mad"]})
    results.sort(key=lambda x: x["mad_cents"])
    return results

# ---------- TRUE spiral geometry ----------
# Phi := log2(f/a4).  Then theta = 2π*Phi, radius r = Phi  (continuous).
def build_true_spiral_for_frame(peaks_f, peaks_m, a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins):
    # Build a base continuous spiral path over desired span
    # Choose center so that Phi=0 (A4) is roughly in the middle
    half = span_turns/2.0
    phi_grid = np.linspace(-half, +half, int(bins_per_turn*span_turns), endpoint=False)  # continuous log2 axis
    # Energy per phi bin from peaks
    if peaks_f.size == 0:
        energy = np.zeros_like(phi_grid)
    else:
        phi_peaks = np.log2(peaks_f / a4_ref)  # continuous
        w = (peaks_m - peaks_m.min()) / (peaks_m.max() - peaks_m.min() + 1e-9) + 1e-6
        # discard peaks far outside the span
        keep = (phi_peaks >= -half) & (phi_peaks < half)
        phi_peaks = phi_peaks[keep]; w = w[keep]
        bins = len(phi_grid)
        # map to nearest bin
        idx = np.floor((phi_peaks + half) / (span_turns) * bins).astype(int)
        idx = np.clip(idx, 0, bins-1)
        energy = np.zeros(bins, dtype=float)
        for i, ww in zip(idx, w):
            energy[i] += ww
        # smooth for continuity
        if smooth_bins > 1:
            ker = np.hanning(smooth_bins)
            ker /= ker.sum()
            pad = smooth_bins//2
            e = np.r_[energy[-pad:], energy, energy[:pad]]
            e = np.convolve(e, ker, mode='same')
            energy = e[pad:-pad]
        # normalize
        if energy.max() > 0:
            energy = energy / (energy.max() + 1e-9)

    # Base radius is phi itself; add wiggle from energy
    r = phi_grid + wiggle_gain * energy
    theta = 2*np.pi * phi_grid
    # to XY
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, phi_grid

def make_plotly_frames_sync(times, peaks_f_list, peaks_m_list, a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins, spokes):
    frames = []
    traces0 = []
    # Background spokes (constant angular fractions of 2π, e.g., 12 for 100¢)
    # We'll draw faint lines from center outward over the whole bounding box.
    # Determine approximate radius bounds from span.
    half = span_turns/2.0
    Rmax = half + 0.75
    for s in range(spokes):
        ang = 2*np.pi * s/spokes
        traces0.append(dict(
            type='scatter', mode='lines', x=[-Rmax*np.cos(ang), Rmax*np.cos(ang)],
            y=[-Rmax*np.sin(ang), Rmax*np.sin(ang)],
            line=dict(width=1, color='rgba(255,255,255,0.06)'), hoverinfo='skip', showlegend=False
        ))
    # Also draw faint equi-radius circles at integer octaves (phi integer)
    for k in range(int(-half), int(half)+1):
        t = np.linspace(0, 2*np.pi, 361)
        r = k
        traces0.append(dict(
            type='scatter', mode='lines', x=(r*np.cos(t)).tolist(), y=(r*np.sin(t)).tolist(),
            line=dict(width=1, color='rgba(255,255,255,0.06)'), hoverinfo='skip', showlegend=False
        ))
    # Build a frame for each processed STFT time
    for pf, pm in zip(peaks_f_list, peaks_m_list):
        x, y, _ = build_true_spiral_for_frame(
            pf, pm, a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins
        )
        frames.append(dict(name="", data=traces0 + [dict(
            type='scatter', mode='lines', x=x.tolist(), y=y.tolist(),
            line=dict(width=14, color='rgba(0,255,200,0.10)'), hoverinfo='skip', showlegend=False
        ), dict(
            type='scatter', mode='lines', x=x.tolist(), y=y.tolist(),
            line=dict(width=3, color='#39FF14'), name='spiral'
        )]))
    init = frames[0]['data'] if frames else traces0
    return dict(
        data=init,
        layout=dict(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
            margin=dict(l=10,r=10,t=40,b=10),
            title="Live TRUE Spiral (angle=pitch class, radius=octaves)"
        ),
        frames=[dict(name=str(i), data=f['data']) for i,f in enumerate(frames)]
    )

def audio_bytes_to_data_uri(data, mime):
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{b64}"

if uploaded is not None:
    raw = uploaded.read()
    mime = uploaded.type
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        if y.ndim > 1: y = np.mean(y, axis=1)
    except Exception as e:
        st.error(f"Decode failed: {e}")
        st.stop()

    times, pf, pm, all_f, all_m = extract_frame_peaks(
        y, sr, frame_ms, hop_ms, frame_stride, fmin, fmax, top_peaks, peak_db
    )
    if all_f.size == 0:
        st.warning("No usable peaks detected.")
        st.stop()

    # --- tuning readout ---
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)
    st.subheader("Best tuning matches")
    for i, r in enumerate(best, 1):
        st.markdown(f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}¢**, error ≈ **{r['mad_cents']:.1f}¢**")
    a4_ref = best[0]["a4"]

    # --- build synced plot frames and embed with an <audio> ---
    fig_spec = make_plotly_frames_sync(times, pf, pm, a4_ref, turns, bins_per_turn, wiggle_gain, smooth_bins, spokes)
    fig_json = json.dumps(fig_spec)
    audio_uri = audio_bytes_to_data_uri(raw, mime)
    times_list = times.tolist()

    st.components.v1.html(f"""
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-2.31.1.min.js"></script>
      <style>
        body {{ background:#0a0a0a; margin:0; }}
        .wrap {{ display:flex; gap:16px; flex-direction:column; }}
        #plot {{ width:100%; height:560px; }}
        audio {{ width:100%; outline:none; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div id="plot"></div>
        <audio id="aud" src="{audio_uri}" controls></audio>
      </div>
      <script>
        const spec = {fig_json};
        const times = {json.dumps(times_list)};
        const N = times.length;
        const plotDiv = document.getElementById('plot');
        Plotly.newPlot(plotDiv, spec.data, spec.layout).then(() => {{
          Plotly.addFrames(plotDiv, spec.frames);
        }});
        const aud = document.getElementById('aud');
        let rafId = null;
        function step() {{
          if (!aud.paused && N > 0) {{
            const t = aud.currentTime;
            let idx = 0;
            while (idx+1 < N && Math.abs(times[idx+1]-t) < Math.abs(times[idx]-t)) idx++;
            Plotly.animate(plotDiv, [String(idx)], {{
              frame: {{duration:0, redraw:true}},
              mode: 'immediate',
              transition: {{duration:0}}
            }});
          }}
          rafId = requestAnimationFrame(step);
        }}
        aud.addEventListener('play', () => {{ if (!rafId) step(); }});
        aud.addEventListener('pause', () => {{ if (rafId) cancelAnimationFrame(rafId); rafId = null; }});
        aud.addEventListener('ended', () => {{ if (rafId) cancelAnimationFrame(rafId); rafId = null; }});
      </script>
    </body>
    </html>
    """, height=640)
else:
    st.info("Upload audio to see the synced TRUE spiral and tuning readout.")
