'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { parseSrt, formatSrt, shiftCues, estimate, FPS } = require('../src/main/subsync');

// Deterministic pseudo-random speech bursts, so the signal has a unique alignment.
function makeCues(n) {
  let seed = 7, t = 2; const cues = [];
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = 0; i < n; i++) { const d = 1 + rnd() * 3; cues.push({ start: t, end: t + d, text: 'line ' + i }); t += d + 0.5 + rnd() * 4; }
  return cues;
}
function speechFrom(cues, total) {
  const a = new Float32Array(Math.ceil(total * FPS));
  for (const c of cues) a.fill(1, Math.round(c.start * FPS), Math.round(c.end * FPS));
  return a;
}

test('srt round-trips', () => {
  const txt = '\uFEFF1\r\n00:00:01,500 --> 00:00:03,000\r\nHi\r\nthere\r\n\r\n2\r\n00:01:00,000 --> 00:01:02,250\r\nBye\r\n';
  const cues = parseSrt(txt);
  assert.deepStrictEqual(cues.map((c) => c.text), ['Hi\nthere', 'Bye']);
  assert.deepStrictEqual(parseSrt(formatSrt(cues)), cues);
});

test('shiftCues drops cues pushed before zero', () => {
  const out = shiftCues([{ start: 0, end: 1, text: 'a' }, { start: 5, end: 6, text: 'b' }], 1, -3);
  assert.deepStrictEqual(out, [{ start: 2 - 2, end: 3, text: 'b' }].map((c) => ({ ...c, start: 2 })));
});

test('estimate recovers a constant offset', async () => {
  const cues = makeCues(60);
  const audio = speechFrom(shiftCues(cues, 1, 7.3), 400);
  const r = await estimate(audio, cues);
  assert.ok(Math.abs(r.offset - 7.3) < 0.02, 'offset ' + r.offset);
  assert.strictEqual(r.scale, 1);
  assert.ok(r.z > 4);
});

test('estimate recovers a negative offset and a 25/23.976 framerate mismatch', async () => {
  const cues = makeCues(80);
  const r1 = await estimate(speechFrom(shiftCues(cues, 1, -4.5), 500), cues);
  assert.ok(Math.abs(r1.offset + 4.5) < 0.02, 'offset ' + r1.offset);
  const r2 = await estimate(speechFrom(shiftCues(cues, 25 / 23.976, 1.2), 500), cues);
  assert.ok(Math.abs(r2.scale - 25 / 23.976) < 1e-9);
  assert.ok(Math.abs(r2.offset - 1.2) < 0.03, 'offset ' + r2.offset);
});
