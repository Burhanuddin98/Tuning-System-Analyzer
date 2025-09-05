# app.py
from __future__ import annotations
import base64, io, json
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.signal import stft, find_peaks
from tuning_generator import pack_defaults

# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="Tuning Analyser — TRUE Spiral (Synced)", layout="wide")
st.title("🎼 Tuning Analyser — TRUE Spiral (live, dark-neon)")

with st.sidebar:
    st.header("STFT")
    frame_ms = st.slider("Window (ms)", 32, 96, 64, step=4)
    hop_ms   = st.slider("Hop (ms)", 8, 48, 24, step=4)
    frame_stride = st.slider("Process every Nth frame", 1, 8, 3, step=1)
    fmin = st.number_input("Min freq (Hz)", 20.0, 400.0, 60.0, step=5.0)
    fmax = st.number_input("Max freq (Hz)", 500.0, 6000.0, 2200.0, step=50.0)
    peak_db = st.slider("Peak threshold (dB rel. global max)", -120, -10, -50, step=5)
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
    turns = st.slider("Octave span (turns)", 2, 10, 4)          # total log2 span
    bins_per_turn = st.slider("Resolution (bins/turn)", 180, 1440, 720, step=60)
    wiggle_gain = st.slider("Ripple gain", 0.00, 1.00, 0.30, step=0.01)
    smooth_bins = st.slider("Ripple smoothing (bins)", 1, 81, 21, step=2)
    spokes = st.slider("Reference spokes (per 100¢)", 6, 36, 12, step=1)

systems = {k: systems_all[k] for k in choose} if len(choose)>0 else systems_all
uploaded = st.file_uploader("Upload WAV/FLAC/OGG/MP3", type=["wav","flac","ogg","mp3"])

# --------------------------- Utils ---------------------------
def amplitude_to_db(x, ref=1.0, amin=1e-12):
    x = np.maximum(x, amin); return 20.0 * np.log10(x / ref)

def hsv_to_rgb_hex(h: float, s: float, v: float) -> str:
    i = int(h * 6.0) % 6; f = h*6.0 - i
    p = v*(1-s); q = v*(1 - f*s); t = v*(1 - (1-f)*s)
    if   i==0: r,g,b = v,t,p
    elif i==1: r,g,b = q,v,p
    elif i==2: r,g,b = p,v,t
    elif i==3: r,g,b = p,q,v
    elif i==4: r,g,b = t,p,v
    else:       r,g,b = v,p,q
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def to_wav_bytes(raw: bytes) -> tuple[bytes, str]:
    """Decode with soundfile and return WAV bytes (mono if needed)."""
    y, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue(), "audio/wav"

# --------------------------- Peaks ---------------------------
def extract_frame_peaks(y, sr, frame_ms, hop_ms, stride, fmin, fmax, topk, peak_db):
    nperseg = max(32, int(sr * frame_ms/1000.0))
    hop = max(8, int(sr * hop_ms/1000.0))
    noverlap = max(0, nperseg - hop)
    f, t, Zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = np.abs(Zxx)
    mag_db = amplitude_to_db(S, ref=np.max(S) + 1e-12)
    mask = (f >= fmin) & (f <= fmax)
    fbin = f[mask]
    peaks_f, peaks_m, t_proc = [], [], []
    for ti in range(0, mag_db.shape[1], stride):
        spec = mag_db[mask, ti]; t_proc.append(float(t[ti]))
        if spec.size < 5: peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        pk, props = find_peaks(spec, height=peak_db)
        if pk.size == 0: peaks_f.append(np.array([])); peaks_m.append(np.array([])); continue
        heights = props["peak_heights"]; idx = np.argsort(heights)[-topk:][::-1]
        sel = pk[idx]
        peaks_f.append(fbin[sel]); peaks_m.append(heights[idx])
    all_f = np.concatenate([p for p in peaks_f if p.size]) if any(p.size for p in peaks_f) else np.array([])
    all_m = np.concatenate([m for m in peaks_m if m.size]) if any(m.size for m in peaks_m) else np.array([])
    return np.array(t_proc), peaks_f, peaks_m, all_f, all_m

# --------------------------- Tuning detection ---------------------------
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
        if mad < best[0]: best = (float(mad), float(off))
    return {"offset": best[1], "mad_cents": best[0]}

def coarse_to_fine(obs_freqs, obs_mags, systems_dict, a4_lo, a4_hi, step_coarse, step_fine, fine_span):
    results, coarse = [], []
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

# --------------------------- Spiral (colored, rippling line) ---------------------------
def build_ripple_spiral_for_frame(peaks_f, peaks_m, a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins):
    """
    Build a perfect Archimedean spiral backbone, then add a small radius
    modulation ('ripples') driven by per-frame energy along log-frequency.
    """
    half = span_turns/2.0
    phi_grid = np.linspace(-half, +half, int(bins_per_turn*span_turns), endpoint=False)  # continuous log2 axis

    energy = np.zeros_like(phi_grid)
    if peaks_f.size:
        phi_peaks = np.log2(peaks_f / a4_ref)
        w = (peaks_m - peaks_m.min()) / (peaks_m.max() - peaks_m.min() + 1e-9) + 1e-6
        keep = (phi_peaks >= -half) & (phi_peaks < half)
        phi_peaks = phi_peaks[keep]; w = w[keep]
        if phi_peaks.size:
            bins = len(phi_grid)
            idx = np.floor((phi_peaks + half) / (span_turns) * bins).astype(int)
            idx = np.clip(idx, 0, bins-1)
            np.add.at(energy, idx, w)
            if smooth_bins > 1:
                ker = np.hanning(smooth_bins); ker /= ker.sum()
                pad = smooth_bins//2
                e = np.r_[energy[-pad:], energy, energy[:pad]]
                e = np.convolve(e, ker, mode='same')
                energy = e[pad:-pad]
            if energy.max() > 0: energy = energy / (energy.max() + 1e-9)

    # Base radius + ripple
    r_base = phi_grid + half + 0.25
    r = r_base + wiggle_gain * energy
    theta = 2*np.pi * phi_grid
    x = r * np.cos(theta); y = r * np.sin(theta)
    return x, y, phi_grid, r.max(), half

def colored_line_segments(x, y, phi_grid, max_segments=220):
    """
    Plotly lacks per-vertex line coloring; split into short segments
    and color each by pitch-class hue (from phi % 1).
    """
    n = len(phi_grid)
    if n < 4:
        return []
    chunks = max(1, min(n//2, int(np.ceil(n / max_segments))))
    traces = []
    for i in range(0, n-1, chunks):
        j = min(n-1, i+chunks)
        xx = x[i:j+1]; yy = y[i:j+1]
        phi_mid = 0.5*(phi_grid[i] + phi_grid[j])
        hue = (phi_mid % 1.0 + 1.0) % 1.0
        col = hsv_to_rgb_hex(hue, 0.95, 0.95)
        traces.append(dict(
            type='scatter', mode='lines',
            x=xx.tolist(), y=yy.tolist(),
            line=dict(width=2, color=col),
            hoverinfo='skip', showlegend=False
        ))
    return traces

def make_plotly_frames_sync(times, peaks_f_list, peaks_m_list,
                            a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins, spokes):
    frames = []

    # size view using an initial frame (or backbone)
    if len(peaks_f_list):
        x0, y0, phi0, _, half = build_ripple_spiral_for_frame(
            peaks_f_list[0], peaks_m_list[0],
            a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins
        )
    else:
        half = span_turns/2.0
        phi0 = np.linspace(-half, +half, int(bins_per_turn*span_turns), endpoint=False)
        r0 = phi0 + half + 0.25
        theta0 = 2*np.pi*phi0
        x0 = r0*np.cos(theta0); y0 = r0*np.sin(theta0)
    Rmax = float(np.max(np.sqrt(x0**2 + y0**2)) + 0.75)

    # guides: spokes & octave circles
    traces_guides = []
    for s in range(spokes):
        ang = 2*np.pi * s / spokes
        traces_guides.append(dict(
            type='scatter', mode='lines',
            x=[-Rmax*np.cos(ang), Rmax*np.cos(ang)],
            y=[-Rmax*np.sin(ang), Rmax*np.sin(ang)],
            line=dict(width=1, color='rgba(255,255,255,0.08)'),
            hoverinfo='skip', showlegend=False
        ))
    for k in range(int(np.floor(-half)), int(np.ceil(half))+1):
        t = np.linspace(0, 2*np.pi, 361)
        r = k + half + 0.25
        traces_guides.append(dict(
            type='scatter', mode='lines',
            x=(r*np.cos(t)).tolist(), y=(r*np.sin(t)).tolist(),
            line=dict(width=1, color='rgba(255,255,255,0.08)'),
            hoverinfo='skip', showlegend=False
        ))

    # frames = guides + colored ripple line (no dots)
    for pf, pm in zip(peaks_f_list, peaks_m_list):
        x, y, phi, _, _ = build_ripple_spiral_for_frame(pf, pm, a4_ref, span_turns, bins_per_turn, wiggle_gain, smooth_bins)
        segs = colored_line_segments(x, y, phi, max_segments=220)
        frames.append(dict(name="", data=traces_guides + segs))

    init = frames[0]['data'] if frames else traces_guides
    return dict(
        data=init,
        layout=dict(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            xaxis=dict(visible=False, range=[-Rmax, Rmax]),
            yaxis=dict(visible=False, range=[-Rmax, Rmax], scaleanchor='x', scaleratio=1),
            margin=dict(l=10, r=10, t=40, b=10),
            title="TRUE Spiral — angle: pitch class • radius: octaves (colored; ripples = energy)"
        ),
        frames=[dict(name=str(i), data=f['data']) for i, f in enumerate(frames)]
    )

# --------------------------- Main ---------------------------
if uploaded is not None:
    raw_in = uploaded.read()
    # decode for analysis
    try:
        y, sr = sf.read(io.BytesIO(raw_in), dtype='float32', always_2d=False)
        if y.ndim > 1: y = np.mean(y, axis=1)
    except Exception as e:
        st.error(f"Decode failed: {e}"); st.stop()

    times, pf, pm, all_f, all_m = extract_frame_peaks(
        y, sr, frame_ms, hop_ms, frame_stride, fmin, fmax, top_peaks, peak_db
    )
    if all_f.size == 0:
        st.warning("No usable peaks detected in the selected band/threshold."); st.stop()

    # tuning
    best = coarse_to_fine(all_f, all_m, systems, a4_lo, a4_hi, a4_step_coarse, a4_step_fine, fine_span_hz)
    st.subheader("Best tuning matches")
    for i, r in enumerate(best, 1):
        st.markdown(f"**{i}. {r['name']}** — A4 ≈ **{r['a4']:.2f} Hz**, tonic offset **{r['offset']:.1f}¢**, error ≈ **{r['mad_cents']:.1f}¢**")
    a4_ref = best[0]["a4"]

    # figure spec (colored ripple line; no dots)
    fig_spec = make_plotly_frames_sync(times, pf, pm, a4_ref, turns, bins_per_turn, wiggle_gain, smooth_bins, spokes)
    fig_json = json.dumps(fig_spec)

    # reliable audio in iframe: always WAV → base64 → Blob URL in JS
    wav_bytes, _ = to_wav_bytes(raw_in)
    audio_b64 = base64.b64encode(wav_bytes).decode('ascii')
    times_list = times.tolist()

    # HTML + JS (Blob audio, binary-search sync, correct animate)
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
        <audio id="aud" controls preload="auto"></audio>
      </div>
      <script>
        const spec = {fig_json};
        const times = {json.dumps(times_list)};
        const N = times.length;

        // Rebuild WAV from base64, use Blob URL (reliable inside iframes)
        const b64 = "{audio_b64}";
        function b64ToBlob(b64Data, contentType) {{
          const byteChars = atob(b64Data);
          const byteNums = new Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) byteNums[i] = byteChars.charCodeAt(i);
          const byteArray = new Uint8Array(byteNums);
          return new Blob([byteArray], {{ type: "audio/wav" }});
        }}
        const aud = document.getElementById('aud');
        try {{
          const blob = b64ToBlob(b64, "audio/wav");
          const url = URL.createObjectURL(blob);
          aud.src = url;
        }} catch (e) {{
          console.error("Audio blob creation failed:", e);
        }}

        // Plotly init
        const plotDiv = document.getElementById('plot');
        Plotly.newPlot(plotDiv, spec.data, spec.layout).then(() => {{
          Plotly.addFrames(plotDiv, spec.frames);
        }});

        // Fast nearest index (binary search)
        function nearestIdx(arr, t) {{
          let lo = 0, hi = arr.length - 1;
          while (lo <= hi) {{
            const mid = (lo + hi) >> 1;
            if (arr[mid] < t) lo = mid + 1; else hi = mid - 1;
          }}
          if (lo === 0) return 0;
          if (lo >= arr.length) return arr.length - 1;
          return (Math.abs(arr[lo] - t) < Math.abs(arr[lo - 1] - t)) ? lo : (lo - 1);
        }}

        // Animation loop (frame-accurate)
        let rafId = null;
        function step() {{
          if (!aud.paused && N > 0) {{
            const idx = nearestIdx(times, aud.currentTime);
            Plotly.animate(plotDiv, String(idx), {{
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
        aud.addEventListener('error', (e) => {{ console.error("Audio error:", e); }});
      </script>
    </body>
    </html>
    """, height=640)
else:
    st.info("Upload audio to see the synced TRUE spiral and tuning readout.")
