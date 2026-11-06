// Choose the best photo inside a hold. Port of pose_extractor/select.py.
// `scoreHold` returns the score of every frame (the app offers the runner-ups as choices);
// `pickBestFrame` returns the best one, exactly like the Python version.

import { ABS_AND_THIGHS, BACK_LAT_SPREAD, FRONT_LAT_SPREAD, LEGS_LEVEL_MAX } from './classic.js';

const MIN_WAIST_RANGE = 0.02;
const MIN_HEAD_RANGE = 0.12;
const MIN_STRENGTH_RANGE = 0.04;

// Same keys and order as WEIGHTS in select.py ('' stands for every other pose).
const WEIGHTS = {
  [ABS_AND_THIGHS]: { vacuum: 0.5, expression: 0.0, head: 0.0, sharp: 0.2, still: 0.15, center: 0.15 },
  [FRONT_LAT_SPREAD]: { vacuum: 0.0, expression: 0.5, head: 0.1, sharp: 0.2, still: 0.1, center: 0.1 },
  '': { vacuum: 0.0, expression: 0.4, head: 0.0, sharp: 0.35, still: 0.1, center: 0.15 },
};

const isNum = (v) => v !== null && v !== undefined;

/** Median filter, window 3, keeping missing values missing (select.py flavour: upper middle). */
function median3(values) {
  const out = values.slice();
  for (let i = 1; i < values.length - 1; i++) {
    const w = values.slice(i - 1, i + 2).filter(isNum).sort((a, b) => a - b);
    if (isNum(values[i]) && w.length) out[i] = w[w.length >> 1];
  }
  return out;
}

function runs(indices) {
  const out = [];
  for (const i of [...new Set(indices)].sort((a, b) => a - b)) {
    if (out.length && i === out[out.length - 1][out[out.length - 1].length - 1] + 1) out[out.length - 1].push(i);
    else out.push([i]);
  }
  return out;
}

function smoothed(indices, value) {
  const out = new Map();
  for (const run of runs(indices)) {
    const vals = median3(run.map(value));
    run.forEach((i, k) => out.set(i, vals[k]));
  }
  return out;
}

/** numpy.percentile with linear interpolation. */
export function percentile(values, p) {
  const v = [...values].sort((a, b) => a - b);
  const pos = (p / 100) * (v.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

function ranker(poolValues, lowerIsBetter, minRange) {
  const present = poolValues.filter(isNum);
  if (present.length < 3) return () => 0.5;
  const lo = percentile(present, 5), hi = percentile(present, 95);
  if (hi - lo < minRange) return () => 0.5;
  return (v) => {
    if (!isNum(v)) return 0.5;
    const t = Math.min(Math.max((v - lo) / (hi - lo), 0), 1);
    return lowerIsBetter ? 1 - t : t;
  };
}

function legsFactor(s) {
  if (!s.features) return 1.0;
  const excess = s.features.legsDy - LEGS_LEVEL_MAX;
  return excess <= 0 ? 1.0 : Math.max(0.2, 1.0 - excess / 0.2);
}

function expression(s, label) {
  if (!s.features) return null;
  return label === FRONT_LAT_SPREAD ? s.features.elbowLat : s.strength;
}

function waists(samples, indices) {
  const torsos = indices.map((i) => samples[i].torsoPx).filter((v) => v);
  const ref = torsos.length ? percentile(torsos, 50) : null;
  return smoothed(indices, (i) => {
    const s = samples[i];
    if (!isNum(s.waist)) return null;
    return ref && s.torsoPx ? (s.waist * s.torsoPx) / ref : s.waist;
  });
}

/** [{i, score}] for every frame of the hold that has a pose. Options: refSharp, refStrength, pool, motionThresh. */
export function scoreHold(samples, hold, { refSharp = null, refStrength = null, pool = null, motionThresh = 0.5 } = {}) {
  const indices = [];
  for (let i = hold.start; i <= hold.end; i++) indices.push(i);
  if (hold.label === BACK_LAT_SPREAD) return scoreBackLatSpread(samples, indices, refSharp, refStrength);

  const poolIdx = pool ? [...new Set([...pool, ...indices])].sort((a, b) => a - b) : indices;
  const label = hold.label in WEIGHTS ? hold.label : '';
  const w = WEIGHTS[label];
  const maxSharp = refSharp || Math.max(0, ...indices.map((i) => samples[i].sharpness)) || 1.0;
  const midIdx = (hold.start + hold.end) / 2;
  const half = Math.max((hold.end - hold.start) / 2, 1e-9);

  const expr = smoothed(poolIdx, (i) => expression(samples[i], label));
  const rankExpr = ranker([...expr.values()], false, MIN_STRENGTH_RANGE);
  const waist = w.vacuum ? waists(samples, poolIdx) : new Map();
  const rankWaist = ranker([...waist.values()], true, MIN_WAIST_RANGE);
  const head = w.head ? smoothed(poolIdx, (i) => samples[i].headDrop) : new Map();
  const rankHead = ranker([...head.values()], true, MIN_HEAD_RANGE);

  const out = [];
  for (const i of indices) {
    const s = samples[i];
    if (!s.pose) continue;
    const terms = {
      vacuum: rankWaist(waist.get(i)),
      expression: rankExpr(expr.get(i)),
      head: rankHead(head.get(i)),
      sharp: Math.min(s.sharpness / maxSharp, 1.0),
      still: isNum(s.speed) ? Math.max(0, 1 - s.speed / motionThresh) : 0.5,
      center: Math.max(0, 1 - Math.abs(i - midIdx) / half),
    };
    let sum = 0;
    for (const k of Object.keys(w)) sum += w[k] * terms[k];
    let score = s.completeness * sum;
    if (label === ABS_AND_THIGHS) score *= legsFactor(s);
    out.push({ i, score });
  }
  return out;
}

/** The original scoring, kept for the back lat spread. */
function scoreBackLatSpread(samples, indices, refSharp, refStrength) {
  const maxSharp = refSharp || Math.max(0, ...indices.map((i) => samples[i].sharpness)) || 1.0;
  const maxStrength = refStrength || Math.max(0, ...indices.map((i) => samples[i].strength)) || 1.0;
  const midIdx = (indices[0] + indices[indices.length - 1]) / 2;
  const half = Math.max((indices[indices.length - 1] - indices[0]) / 2, 1e-9);
  const out = [];
  for (const i of indices) {
    const s = samples[i];
    if (!s.pose) continue;
    const centrality = Math.max(0, 1 - Math.abs(i - midIdx) / half);
    const sharp = Math.min(s.sharpness / maxSharp, 1.0);
    const str = Math.min(s.strength / maxStrength, 1.0);
    out.push({ i, score: s.completeness * (0.5 * str + 0.35 * sharp + 0.15 * centrality) });
  }
  return out;
}

/** [index, score] of the best frame (first one on ties), or [-1, 0]. */
export function pickBestFrame(samples, hold, opts) {
  let best = [-1, 0.0];
  for (const { i, score } of scoreHold(samples, hold, opts)) {
    if (best[0] < 0 || score > best[1]) best = [i, score];
  }
  return best;
}
