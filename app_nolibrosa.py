
# app.py
from __future__ import annotations
import base64, io, json, math
from typing import List, Tuple
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import stft, find_peaks
from tuning_generator import pack_defaults

# --------------------------- Streamlit chrome ---------------------------
st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (Synced)", layout="wide")
st.title("🎼 Tuning Analyser — TRUE Spiral (live, dark-neon)")

with st.sidebar:
    st.header("STFT")
    frame_ms = st.slider("Window (ms)", 32, 96, 64, step=4)
    hop_ms   = st.slider("Hop (ms)", 8, 48, 24, step=4)
    frame_stride = st.slider("Process every Nth frame", 1, 8, 3, step=1)
    fmin = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax = st.number_input("Max freq (Hz)", 500.0, 6000.0, 2200.0, step=50.0)
    peak_db = st.slider("Peak threshold (dB rel. frame max)", -120, -10, -50, step=5)
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

    st.header("Spiral")
    turns = st.slider("Octave span (turns)", 2, 10, 4)  # total log2 span
    bins_per_turn = st.slider("Resolution (bins per turn)", 180, 1440, 720, step=60)
    spokes = st.slider("Reference spokes (per 100¢)", 6, 36, 12, step=1)
    trail_frames = st.slider("Motion trail (frames)", 0, 30, 8, step=1)

systems = {k: systems_all[k] for k in choose} if len(choose)>0 else systems_all

uploaded = st.file_uploader("Upload WAV/FLAC/OGG", type=["wav","flac","ogg"])

# --------------------------- Utils ---------------------------
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin)
    return 20.0 * np.log10(x / ref)

def hsv_to_rgb_hex(h: float, s: float, v: float) -> str:
    """h in [0,1), s,v in [0,1] -> '#RRGGBB'."""
    i = int(h * 6.0) % 6
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def audio_bytes_to_data_uri(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{b64}"

# --------------------------- Peak extraction ---------------------------
def extract_frame_peaks(y, sr, frame_ms, hop_ms, stride, fmin, fmax, topk, peak_db):
    nperseg = max(32, int(sr * frame_ms/1000.0))
    hop = max(8, int(sr * hop_ms/1000.0))
    noverlap = max(0, nperseg - hop)
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = np.abs(Zxx)

    # dB relative to global max to maintain consistent thresholding
    mag_db = amplitude_to_db(S, ref=np.max(S) + 1e-12)
    mask = (f >= fmin) & (f <= fmax)
    fbin = f[mask]

    peaks_f, peaks_m, t_proc = [], [], []
    for ti in range(0, mag_db.shape[1], stride):
        spec = mag_db[mask, ti]
        t_proc.append(float(t[ti]))
        if spec.size < 5:
            peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        pk, props = find_peaks(spec, height=peak_db)
        if pk.size == 0:
            peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        heights = props["peak_heights"]
        # strongest first
        idx = np.argsort(heights)[-topk:][::-1]
        sel = pk[idx]
        peaks_f.append(fbin[sel])
        peaks_m.append(heights[idx])

    all_f = np.concatenate([p for p in peaks_f if p.size]) if any(p.size for p in peaks_f) else np.array([])
    all_m = np.concatenate([m for m in peaks_m if m.size]) if any(m.size for m in peaks_m) else np.array([])
    return np.array(t_proc), peaks_f, peaks_m, all_f, all_m

# --------------------------- Tuning detection ---------------------------
def score_system(obs_freqs, obs_mags, cents_grid, a4_ref):
    # pitch class in cents relative to A4
    obs_pc = (1200.0 * np.log2(obs_freqs / a4_ref)) % 1200.0
    # weights: shifted positive
    w = np.maximum(obs_mags - obs_mags.min(), 1.0)
    best = (1e9, 0.0)  # (mad, offset)
    # search offsets on the same grid (fast + good enough)
    for off in cents_grid:
        grid = (cents_grid - off) % 1200.0
        diffs = np.abs(obs_pc[:, None] - grid[None, :]) % 1200.0
        diffs = np.minimum(diffs, 1200.0 - diffs)
        dmin = diffs.min(axis=1)

        # weighted median (robust)
        order = np.argsort(dmin)
        w_sorted = w[order]; d_sorted = dmin[order]
        cumw = np.cumsum(w_sorted)
        med_idx = np.searchsorted(cumw, cumw[-1] / 2.0)
        mad = d_sorted[min(med_idx, len(d_sorted)-1)]
        if mad < best[0]:
            best = (float(mad), float(off))
    return {"offset": best[1], "mad_cents": best[0]}

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
    # refine top 2
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

# --------------------------- Perfect spiral (fixed grid) ---------------------------
# Phi := log2(f/a4). Angle θ = 2π*Phi (pitch class), Radius r = Phi + offset (octaves)
def build_base_spiral(span_turns: int, bins_per_turn: int):
    half = span_turns / 2.0
    phi_grid = np.linspace(-half, +half, int(bins_per_turn * span_turns), endpoint=False)
    r = phi_grid + half + 0.25  # positive radius; center-ish view
    theta = 2.0 * np.pi * phi_grid
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, phi_grid, half, r.max()

def peaks_to_spiral_points(peaks_f: np.ndarray,
                           peaks_m: np.ndarray,
                           a4_ref: float,
                           half: float):
    """Map this frame's peaks to spiral points. Hue by pitch class, size/brightness by magnitude."""
    if peaks_f.size == 0:
        return [], [], [], []
    phi = np.log2(peaks_f / a4_ref)
    keep = (phi >= -half) & (phi < half)
    phi = phi[keep]
    mags = peaks_m[keep]
    if phi.size == 0:
        return [], [], [], []

    r = phi + half + 0.25
    theta = 2.0 * np.pi * phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pc_cents = (phi % 1.0) * 1200.0     # pitch class angle → hue
    m = (mags - mags.min()) / (mags.max() - mags.min() + 1e-9)

    colors = [hsv_to_rgb_hex(h=pc/1200.0, s=0.95, v=0.35 + 0.65*mm)
              for pc, mm in zip(pc_cents, m)]
    sizes = (6.0 + 14.0 * m).tolist()
    return x.tolist(), y.tolist(), colors, sizes

def make_plotly_frames_sync(times, peaks_f_list, peaks_m_list,
                            a4_ref, span_turns, bins_per_turn, spokes, trail_frames=0):
    frames = []

    # Static spiral backbone
    base_x, base_y, phi_grid, half, rmax = build_base_spiral(span_turns, bins_per_turn)
    Rmax = rmax + 0.75  # padding

    # Guides: spokes
    traces_guides = []
    for s in range(spokes):
        ang = 2*np.pi * s / spokes
        traces_guides.append(dict(
            type='scatter', mode='lines',
            x=[-Rmax*np.cos(ang), Rmax*np.cos(ang)],
            y=[-Rmax*np.sin(ang), Rmax*np.sin(ang)],
            line=dict(width=1, color='rgba(255,255,255,0.06)'),
            hoverinfo='skip', showlegend=False
        ))
    # Guides: octave circles (integer Phi)
    for k in range(int(np.floor(-half)), int(np.ceil(half))+1):
        t = np.linspace(0, 2*np.pi, 361)
        r = k + half + 0.25
        traces_guides.append(dict(
            type='scatter', mode='lines',
            x=(r*np.cos(t)).tolist(), y=(r*np.sin(t)).tolist(),
            line=dict(width=1, color='rgba(255,255,255,0.06)'),
            hoverinfo='skip', showlegend=False
        ))

    # Spiral backbone (glow + line)
    spiral_glow = dict(
        type='scatter', mode='lines',
        x=base_x.tolist(), y=base_y.tolist(),
        line=dict(width=14, color='rgba(0,255,200,0.10)'),
        hoverinfo='skip', name='spiral_glow', showlegend=False
    )
    spiral_line = dict(
        type='scatter', mode='lines',
        x=base_x.tolist(), y=base_y.tolist(),
        line=dict(width=2, color='#39FF14'),
        name='spiral', showlegend=False
    )

    # Build frames
    # Optional motion trail: include markers from last K frames with decaying opacity
    for i, (pf, pm) in enumerate(zip(peaks_f_list, peaks_m_list)):
        layers = traces_guides + [spiral_glow, spiral_line]

        # current + trailing frames
        start = max(0, i - trail_frames)
        for j in range(start, i + 1):
            px, py, pcol, psz = peaks_to_spiral_points(peaks_f_list[j], peaks_m_list[j], a4_ref, half)
            fade = 1.0 if j == i else (0.2 + 0.8 * (j - start) / max(1, i - start))  # older → dimmer
            peaks_scatter = dict(
                type='scatter', mode='markers',
                x=px, y=py,
                marker=dict(size=psz, color=pcol, line=dict(width=0), opacity=fade),
                hoverinfo='skip', name='peaks' if j == i else 'trail', showlegend=False
            )
            layers.append(peaks_scatter)

        frames.append(dict(name=str(i), data=layers))

    init = frames[0]['data'] if frames else (traces_guides + [spiral_glow, spiral_line])

    return dict(
        data=init,
        layout=dict(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            xaxis=dict(visible=False, range=[-Rmax, Rmax]),
            yaxis=dict(visible=False, range=[-Rmax, Rmax], scaleanchor='x', scaleratio=1),
            margin=dict(l=10, r=10, t=40, b=10),
            title="TRUE Spiral — angle: pitch class • radius: octaves (RGB by pitch, size by energy)"
        ),
        frames=frames
    )

# --------------------------- Main flow ---------------------------
if uploaded is not None:
    raw = uploaded.read()
    mime = uploaded.type
    try:
        y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception as e:
        st.error(f"Decode failed: {e}")
        st.stop()

    times, pf, pm, all_f, all_m = extract_frame_peaks(
        y, sr, frame_ms, hop_ms, frame_stride, fmin, fmax, top_peaks, peak_db
    )
    if all_f.size == 0:
        st.warning("No usable peaks detected in the selected band/threshold.")
        st.stop()

    # --- tuning readout ---
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)
    st.subheader("Best tuning matches")
    for i, r in enumerate(best, 1):
        st.markdown(f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}¢**, error ≈ **{r['mad_cents']:.1f}¢**")
    a4_ref = best[0]["a4"]

    # --- build synced plot frames + audio ---
    fig_spec = make_plotly_frames_sync(times, pf, pm, a4_ref, turns, bins_per_turn, spokes, trail_frames)
    fig_json = json.dumps(fig_spec)
    audio_uri = audio_bytes_to_data_uri(raw, mime)
    times_list = times.tolist()

    # HTML embed with binary-search sync
    st.components.v1.html(f"""
    <html>
    <head>
      <meta charset="utf-8" />
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

        function nearestIdx(arr, t) {{
          let lo = 0, hi = arr.length - 1;
          while (lo <= hi) {{
            const mid = (lo + hi) >> 1;
            if (arr[mid] < t) lo = mid + 1;
            else hi = mid - 1;
          }}
          if (lo === 0) return 0;
          if (lo >= arr.length) return arr.length - 1;
          return (Math.abs(arr[lo] - t) < Math.abs(arr[lo - 1] - t)) ? lo : (lo - 1);
        }}

        function step() {{
          if (!aud.paused && N > 0) {{
            const idx = nearestIdx(times, aud.currentTime);
            Plotly.animate(plotDiv, [String(idx)], {{
              frame: {{duration: 0, redraw: true}},
              mode: 'immediate',
              transition: {{duration: 0}}
            }});
          }}
          rafId = requestAnimationFrame(step);
        }}
        aud.addEventListener('play',  () => {{ if (!rafId) step(); }});
        aud.addEventListener('pause', () => {{ if (rafId) cancelAnimationFrame(rafId); rafId = null; }});
        aud.addEventListener('ended', () => {{ if (rafId) cancelAnimationFrame(rafId); rafId = null; }});
      </script>
    </body>
    </html>
    """, height=640)
else:
    st.info("Upload audio to see the synced TRUE spiral and tuning readout.")
