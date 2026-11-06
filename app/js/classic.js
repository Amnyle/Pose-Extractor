// The six classic bodybuilding poses from body keypoints. Port of pose_extractor/classic.py
// (see that file for how the features and thresholds were calibrated).

import { MIN_TORSO_PX, mid, torsoLength } from './geometry.js';

export const FRONT_DOUBLE_BICEPS = 'front_double_biceps';
export const BACK_DOUBLE_BICEPS = 'back_double_biceps';
export const FRONT_LAT_SPREAD = 'front_lat_spread';
export const BACK_LAT_SPREAD = 'back_lat_spread';
export const ABS_AND_THIGHS = 'abs_and_thighs';
export const SIDE_CHEST = 'side_chest';
export const OTHER = 'other';

export const CLASSIC_POSES = [
  FRONT_DOUBLE_BICEPS, BACK_DOUBLE_BICEPS, FRONT_LAT_SPREAD, BACK_LAT_SPREAD, ABS_AND_THIGHS, SIDE_CHEST,
];

export const DEFAULT_ARM_RAISE_MIN = 0.15;
const HIGH_RAISE = 1.0;
const DOUBLE_BICEPS_MIN_ELBOW_LAT = 0.7;
const SIDE_MAX_SHOULDER_W = 0.55;
const SIDE_MAX_ELBOW_LAT = 0.55;
const LAT_MIN_ELBOW_LAT = 0.6;
const TORSO_MIN_VIS = 0.5;
const ARM_MIN_VIS = 0.3;
export const LEGS_LEVEL_MAX = 0.12;
const WAIST_ROWS = [0.15, 0.6];

/** Python's round(): halves go to the even neighbour. */
export function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** {raise, elbowLat, shoulderW, facing, legsDy} or null. */
export function armFeatures(kp, armMinVis = ARM_MIN_VIS) {
  if (!kp || kp.length < 29) return null;
  if ([11, 12, 23, 24].some((i) => kp[i][2] < TORSO_MIN_VIS)) return null;
  const shoulders = mid(kp[11], kp[12]);
  const hips = mid(kp[23], kp[24]);
  const torso = torsoLength(kp);
  if (!Number.isFinite(torso) || torso < MIN_TORSO_PX) return null;
  const wrists = [15, 16].filter((i) => kp[i][2] >= armMinVis);
  const elbows = [13, 14].filter((i) => kp[i][2] >= armMinVis);
  if (!wrists.length || !elbows.length) return null;
  const raise = Math.min(...wrists.map((i) => (hips[1] - kp[i][1]) / torso));
  const elbowLat = elbows.reduce((s, i) => s + Math.abs(kp[i][0] - shoulders[0]) / torso, 0) / elbows.length;
  const dx = (kp[11][0] - kp[12][0]) / torso;
  const legsDy = Math.min(kp[27][2], kp[28][2]) >= armMinVis ? Math.abs(kp[27][1] - kp[28][1]) / torso : 0;
  return { raise, elbowLat, shoulderW: Math.abs(dx), facing: dx >= 0 ? 1 : -1, legsDy };
}

/**
 * Narrowest waist width in torso lengths, measured on the person mask.
 * mask: {width, height, data} with data a Float32Array (row-major, values above 0.5 are the body).
 */
export function waistRatio(mask, kp) {
  const hips = mid(kp[23], kp[24]);
  const torso = torsoLength(kp);
  if (!Number.isFinite(torso) || torso < MIN_TORSO_PX) return null;
  const { width, height, data } = mask;
  const x = pyRound(Math.min(Math.max(hips[0], 0), width - 1));
  const widths = [];
  for (let k = 0; k < 10; k++) {
    // same rows as numpy.linspace(0.15, 0.60, 10)
    const f = k === 9 ? WAIST_ROWS[1] : WAIST_ROWS[0] + k * ((WAIST_ROWS[1] - WAIST_ROWS[0]) / 9);
    const y = pyRound(hips[1] - f * torso);
    if (y < 0 || y >= height) continue;
    const row = y * width;
    if (!(data[row + x] > 0.5)) continue;
    let l = x;
    while (l > 0 && data[row + l - 1] > 0.5) l--;
    let r = x;
    while (r < width - 1 && data[row + r + 1] > 0.5) r++;
    widths.push((r - l + 1) / torso);
  }
  return widths.length >= 5 ? Math.min(...widths) : null;
}

/** How far the nose sits below the ears, in ear-to-ear distances. Larger = head tilted down. */
export function headDrop(kp, minVis = ARM_MIN_VIS) {
  if (kp.length < 9 || Math.min(kp[0][2], kp[7][2], kp[8][2]) < minVis) return null;
  const earDist = Math.hypot(kp[7][0] - kp[8][0], kp[7][1] - kp[8][1]);
  if (earDist < 4.0) return null;
  return (kp[0][1] - (kp[7][1] + kp[8][1]) / 2) / earDist;
}

export function classify(f, armRaiseMin = DEFAULT_ARM_RAISE_MIN) {
  if (!f || f.raise < armRaiseMin) return null;
  const side = f.facing > 0 ? 'front' : 'back';
  if (f.raise >= HIGH_RAISE) {
    if (f.elbowLat >= DOUBLE_BICEPS_MIN_ELBOW_LAT) return `${side}_double_biceps`;
    return side === 'front' ? ABS_AND_THIGHS : OTHER;
  }
  if (f.shoulderW < SIDE_MAX_SHOULDER_W && f.elbowLat < SIDE_MAX_ELBOW_LAT) return SIDE_CHEST;
  if (f.elbowLat >= LAT_MIN_ELBOW_LAT) return `${side}_lat_spread`;
  return OTHER;
}

export function strength(f) {
  return f.elbowLat + 0.25 * Math.min(Math.max(f.raise, 0), 1.5);
}
