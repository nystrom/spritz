'use strict';

// Automatic subtitle sync — the ffsubsync (`ffs`) algorithm, in Node.
//
// There is no maintained npm port of ffsubsync, so this re-implements its method on top of the
// ffmpeg Spritz already ships and `fft.js` (pure JS, no native rebuild):
//   1. decode the audio to 16 kHz mono and mark each 10 ms frame speech / non-speech (voice activity),
//   2. rasterise the subtitle cues into the same 10 ms grid (1 while a cue is on screen),
//   3. cross-correlate the two signals with an FFT; the peak is the offset that lines them up,
//   4. repeat for the common framerate-mismatch ratios and keep the best (offset, scale) pair.
// Differences from ffs: its VAD is webrtcvad (native); this one is an adaptive energy gate, which is
// cruder on music/noisy beds. Offsets are capped at ±60 s like ffs; the same cue text is kept.

const { spawn } = require('child_process');
const FFT = require('fft.js');

const RATE = 16000;
const FRAME = 160;                    // samples per 10 ms frame
const FPS = 100;                      // frames per second
const MAX_OFFSET_S = 60;
const RATIOS = [1, 24 / 23.976, 23.976 / 24, 25 / 23.976, 23.976 / 25, 25 / 24, 24 / 25];

// ---- SRT ----
function tsToSec(h, m, s, ms) { return (+h) * 3600 + (+m) * 60 + (+s) + (+ms.padEnd(3, '0')) / 1000; }
function secToTs(t) {
  const ms = Math.round(Math.max(0, t) * 1000);
  const p = (n, w) => String(n).padStart(w, '0');
  return p(Math.floor(ms / 3600000), 2) + ':' + p(Math.floor(ms / 60000) % 60, 2) + ':' + p(Math.floor(ms / 1000) % 60, 2) + ',' + p(ms % 1000, 3);
}
const TIME_RE = /(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})/;

// -> [{ start, end, text }] in seconds. Tolerates BOM, CRLF and missing/odd index lines.
function parseSrt(text) {
  const cues = [];
  for (const block of String(text).replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split(/\n{2,}/)) {
    const lines = block.split('\n');
    const i = lines.findIndex((l) => TIME_RE.test(l));
    if (i < 0) continue;
    const m = TIME_RE.exec(lines[i]);
    const start = tsToSec(m[1], m[2], m[3], m[4]), end = tsToSec(m[5], m[6], m[7], m[8]);
    if (end > start) cues.push({ start, end, text: lines.slice(i + 1).join('\n').trim() });
  }
  return cues;
}
function formatSrt(cues) {
  return cues.map((c, i) => (i + 1) + '\n' + secToTs(c.start) + ' --> ' + secToTs(c.end) + '\n' + c.text + '\n').join('\n');
}

// Map every cue through t' = t * scale + offset; cues pushed wholly before 0 are dropped.
function shiftCues(cues, scale, offset) {
  const out = [];
  for (const c of cues) {
    const end = c.end * scale + offset;
    if (end <= 0) continue;
    out.push({ start: Math.max(0, c.start * scale + offset), end, text: c.text });
  }
  return out;
}

// ---- signals ----
// Per-frame RMS (dB) -> 0/1 speech. Threshold sits a third of the way from the noise floor (10th
// percentile) to the loud end (95th), then a 3-frame close/open smooths flicker.
function speechFromEnergy(db) {
  const n = db.length;
  const out = new Float32Array(n);
  if (!n) return out;
  const sorted = Float32Array.from(db).sort();
  const lo = sorted[Math.floor(n * 0.10)], hi = sorted[Math.floor(n * 0.95)];
  if (hi - lo < 6) return out;                         // flat audio: no usable speech signal
  const thr = lo + (hi - lo) / 3;
  for (let i = 0; i < n; i++) out[i] = db[i] > thr ? 1 : 0;
  // fill gaps shorter than 150 ms, drop blips shorter than 50 ms
  fillRuns(out, 0, 15);
  fillRuns(out, 1, 5);
  return out;
}
function fillRuns(a, val, maxLen) {
  for (let i = 0; i < a.length;) {
    if (a[i] !== val) { i++; continue; }
    let j = i; while (j < a.length && a[j] === val) j++;
    if (j - i < maxLen && i > 0 && j < a.length) a.fill(1 - val, i, j);
    i = j;
  }
}

function cueSignal(cues, scale, length) {
  const s = new Float32Array(length);
  for (const c of cues) {
    const a = Math.max(0, Math.round(c.start * scale * FPS)), b = Math.min(length, Math.round(c.end * scale * FPS));
    if (b > a) s.fill(1, a, b);
  }
  return s;
}

function centered(a) {
  let mean = 0; for (let i = 0; i < a.length; i++) mean += a[i];
  mean /= a.length || 1;
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] - mean;
  return o;
}

// Best shift d (frames) maximising sum_t sub[t] * audio[t + d], |d| <= MAX_OFFSET_S. Returns the
// peak and a z-score of that peak against all lags (how much it stands out).
function bestLag(audio, sub, fft, audioSpec, size) {
  const inp = new Float32Array(size); inp.set(sub);
  const spec = fft.createComplexArray(), prod = fft.createComplexArray(), full = fft.createComplexArray();
  fft.realTransform(spec, inp); fft.completeSpectrum(spec);
  for (let k = 0; k < size; k++) {                    // audioSpec * conj(spec)
    const ar = audioSpec[2 * k], ai = audioSpec[2 * k + 1], br = spec[2 * k], bi = -spec[2 * k + 1];
    prod[2 * k] = ar * br - ai * bi; prod[2 * k + 1] = ar * bi + ai * br;
  }
  fft.inverseTransform(full, prod);
  const maxLag = MAX_OFFSET_S * FPS;
  let best = -Infinity, bestD = 0, sum = 0, sq = 0, cnt = 0;
  for (let d = -maxLag; d <= maxLag; d++) {
    const v = full[2 * ((d + size) % size)];
    sum += v; sq += v * v; cnt++;
    if (v > best) { best = v; bestD = d; }
  }
  const mean = sum / cnt, sd = Math.sqrt(Math.max(1e-12, sq / cnt - mean * mean));
  return { lag: bestD, score: best, z: (best - mean) / sd };
}

const tick = () => new Promise((r) => setImmediate(r)); // FFTs are synchronous; let the UI breathe

// speech: Float32Array of 0/1 audio frames. -> { offset (s), scale, z } or null if nothing found.
async function estimate(speech, cues) {
  if (!cues.length || !speech.length) return null;
  const lastEnd = cues.reduce((m, c) => Math.max(m, c.end), 0);
  const subLen = Math.ceil(lastEnd * 1.05 * FPS) + 1;
  let size = 1; while (size < speech.length + subLen + MAX_OFFSET_S * FPS * 2) size <<= 1;
  const fft = new FFT(size);
  const audioSpec = fft.createComplexArray();
  const a = new Float32Array(size); a.set(centered(speech));
  fft.realTransform(audioSpec, a); fft.completeSpectrum(audioSpec);
  let best = null;
  for (const scale of RATIOS) {
    await tick();
    const r = bestLag(audioSpec, centered(cueSignal(cues, scale, subLen)), fft, audioSpec, size);
    if (!best || r.score > best.score) best = { ...r, scale };
  }
  return { offset: best.lag / FPS, scale: best.scale, z: best.z };
}

// ---- audio ----
// Stream 16 kHz mono s16le out of ffmpeg and reduce to per-frame dB without buffering the PCM.
function audioSpeech(ffmpeg, file, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpeg, ['-v', 'error', '-i', file, '-map', '0:a:0', '-vn', '-sn', '-ac', '1', '-ar', String(RATE), '-f', 's16le', '-'],
      { stdio: ['ignore', 'pipe', 'ignore'] });
    const db = [];
    let carry = Buffer.alloc(0), acc = 0, n = 0;
    const timer = setTimeout(() => { p.kill(); reject(new Error('audio analysis timed out')); }, opts.timeout || 600000);
    p.stdout.on('data', (chunk) => {
      const buf = carry.length ? Buffer.concat([carry, chunk]) : chunk;
      const usable = buf.length - (buf.length % 2);
      for (let i = 0; i < usable; i += 2) {
        const v = buf.readInt16LE(i) / 32768;
        acc += v * v;
        if (++n === FRAME) { db.push(10 * Math.log10(acc / FRAME + 1e-10)); acc = 0; n = 0; }
      }
      carry = buf.subarray(usable);
    });
    p.on('error', (e) => { clearTimeout(timer); reject(e); });
    p.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0 && !db.length) return reject(new Error('ffmpeg failed to decode audio'));
      resolve(speechFromEnergy(db));
    });
  });
}

const MIN_Z = 4;            // peak must stand out this many sigma over the other lags, else leave the subs alone

// Sync SRT text to a media file. -> { srt, offset, scale, z } or throws. Returns null when the
// correlation is too weak to trust (caller should keep the original subtitles).
async function syncSrt({ ffmpeg, media, srtText }) {
  const cues = parseSrt(srtText);
  if (!cues.length) throw new Error('no subtitle cues found');
  const speech = await audioSpeech(ffmpeg, media);
  const est = await estimate(speech, cues);
  if (!est || est.z < MIN_Z) return null;
  return { srt: formatSrt(shiftCues(cues, est.scale, est.offset)), offset: est.offset, scale: est.scale, z: est.z };
}

module.exports = { parseSrt, formatSrt, shiftCues, speechFromEnergy, estimate, syncSrt, FPS };
