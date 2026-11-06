// Find the moments where the athlete holds a pose. Port of pose_extractor/holds.py.

import { poseDistance } from './geometry.js';

/** Speed (torso lengths / s) between each sample and the previous one; NaN where unknown. */
export function poseSpeeds(poses, times, cuts, topK = 3) {
  const n = poses.length;
  const speeds = new Array(n).fill(NaN);
  for (let i = 1; i < n; i++) {
    if (cuts[i] || !poses[i] || !poses[i - 1]) continue;
    const dt = times[i] - times[i - 1];
    if (dt <= 0) continue;
    const d = poseDistance(poses[i - 1], poses[i], topK);
    if (!Number.isNaN(d)) speeds[i] = d / dt;
  }
  return speeds;
}

function median(values) {
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Window-3 median that removes one-sample spikes and keeps NaNs (numpy version). */
function median3(values) {
  const out = values.slice();
  for (let i = 1; i < values.length - 1; i++) {
    if (Number.isNaN(values[i])) continue;
    out[i] = median(values.slice(i - 1, i + 2).filter((v) => !Number.isNaN(v)));
  }
  return out;
}

export function findHolds(speeds, times, motionThresh, minDuration) {
  const smooth = median3(speeds);
  const low = smooth.map((v) => !Number.isNaN(v) && v < motionThresh);
  const holds = [];
  const n = smooth.length;
  let i = 1;
  while (i < n) {
    if (!low[i]) { i++; continue; }
    let j = i;
    while (j + 1 < n && low[j + 1]) j++;
    const start = i - 1, end = j;
    if (times[end] - times[start] >= minDuration) holds.push({ start, end, label: null });
    i = j + 1;
  }
  return holds;
}

export function splitByLabel(holds, labels, times, minDuration) {
  const out = [];
  for (const hold of holds) {
    const seq = labels.slice(hold.start, hold.end + 1);
    for (let i = 1; i < seq.length - 1; i++) {
      if (seq[i] !== seq[i - 1] && seq[i - 1] === seq[i + 1]) seq[i] = seq[i - 1];
    }
    let run = 0;
    for (let k = 1; k <= seq.length; k++) {
      if (k === seq.length || seq[k] !== seq[run]) {
        const start = hold.start + run, end = hold.start + k - 1;
        if (times[end] - times[start] >= minDuration) out.push({ start, end, label: seq[run] });
        run = k;
      }
    }
  }
  return out;
}

export function splitOnDrift(holds, poses, times, driftThresh, minDuration) {
  const out = [];
  const emit = (start, end) => {
    if (times[end] - times[start] >= minDuration) out.push({ start, end, label: null });
  };
  for (const hold of holds) {
    let segStart = hold.start;
    let ref = poses[segStart];
    for (let k = hold.start + 1; k <= hold.end; k++) {
      const d = ref && poses[k] ? poseDistance(ref, poses[k]) : NaN;
      if (Number.isNaN(d) || d > driftThresh) {
        emit(segStart, k - 1);
        segStart = k;
        ref = poses[k];
      }
    }
    emit(segStart, hold.end);
  }
  return out;
}
