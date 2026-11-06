// From analysed frames to labelled holds, best frames and the photo choices offered to the user.
// Port of build_sample / select_poses in pose_extractor/pipeline.py, plus `buildChoices` for the app.

import {
  ABS_AND_THIGHS, BACK_LAT_SPREAD, DEFAULT_ARM_RAISE_MIN, FRONT_LAT_SPREAD, OTHER,
  armFeatures, classify, headDrop, strength, waistRatio,
} from './classic.js';
import { completeness, normalizePose, poseDistance, torsoLength } from './geometry.js';
import { findHolds, poseSpeeds, splitByLabel, splitOnDrift } from './holds.js';
import { REFERENCES } from './references.js';
import { percentile, scoreHold } from './select.js';

export const FULL_STRENGTH = 1.3;

export const DEFAULT_CONFIG = {
  sampleFps: 6,
  analysisMaxSide: 960,
  cutThreshold: 0.5,
  motionThresh: 0.5,
  minHold: 0.8,
  driftThresh: 0.3,
  minVisibility: 0.5,
  armRaiseMin: DEFAULT_ARM_RAISE_MIN,
};

/** Everything measured on one frame. kp: 33 [x, y, visibility] in pixels, or null. mask: see waistRatio. */
export function buildSample({ index, time, kp, width, height, sharpness, cut, mask = null, waist }, cfg = DEFAULT_CONFIG) {
  const base = {
    index, time, pose: null, completeness: 0, sharpness, cut, features: null, label: null,
    strength: 0, waist: null, headDrop: null, torsoPx: null, speed: null,
  };
  if (!kp) return base;
  const pose = normalizePose(kp, cfg.minVisibility);
  const features = pose ? armFeatures(kp) : null;
  const label = classify(features, cfg.armRaiseMin);
  const torso = torsoLength(kp);
  let w = null;
  if (label === ABS_AND_THIGHS) {
    // a mask from the pose model may have its own size (sx, sy): measure in its pixels
    const mkp = mask && mask.sx ? kp.map(([x, y, v]) => [x * mask.sx, y * mask.sy, v]) : kp;
    w = waist !== undefined ? waist : mask ? waistRatio(mask, mkp) : null;
  }
  return {
    ...base,
    pose,
    completeness: completeness(kp, width, height, cfg.minVisibility),
    features,
    label,
    strength: features ? strength(features) : 0,
    waist: w,
    headDrop: label === FRONT_LAT_SPREAD ? headDrop(kp) : null,
    torsoPx: Number.isFinite(torso) && torso > 0 ? torso : null,
  };
}

/** Holds (with labels) and the frame scores of every candidate hold. Fills in each sample's speed. */
export function analyseHolds(samples, cfg = DEFAULT_CONFIG) {
  const poses = samples.map((s) => s.pose);
  const times = samples.map((s) => s.time);
  const speeds = poseSpeeds(poses, times, samples.map((s) => s.cut));
  samples.forEach((s, i) => {
    const around = speeds.slice(i, i + 2).filter((v) => !Number.isNaN(v));
    s.speed = around.length ? Math.max(...around) : null;
  });
  let holds = findHolds(speeds, times, cfg.motionThresh, cfg.minHold);
  holds = splitOnDrift(holds, poses, times, cfg.driftThresh, cfg.minHold);
  holds = splitByLabel(holds, samples.map((s) => s.label), times, cfg.minHold);

  const sharp = samples.filter((s) => s.pose).map((s) => s.sharpness);
  const refSharp = sharp.length ? percentile(sharp, 90) : null;
  const pools = {};
  for (const h of holds) {
    if (h.label === null || h.label === OTHER || h.label === BACK_LAT_SPREAD) continue;
    (pools[h.label] ||= []).push(...range(h.start, h.end));
  }
  for (const h of holds) {
    if (h.label === null) continue;
    h.frames = scoreHold(samples, h, {
      refSharp, refStrength: FULL_STRENGTH, pool: pools[h.label] || null, motionThresh: cfg.motionThresh,
    });
  }
  return { holds, refSharp };
}

/**
 * Named poses that already have a finished hold: held for at least `minHold`, and over (the athlete
 * has moved on for at least `settle` seconds), so the hold's best frame is known.
 * Used to stop the analysis early once every pose the user expects has been seen.
 */
export function finishedPoses(samples, cfg = DEFAULT_CONFIG, settle = 0.5) {
  const done = new Set();
  if (samples.length < 3) return done;
  const poses = samples.map((s) => s.pose);
  const times = samples.map((s) => s.time);
  const speeds = poseSpeeds(poses, times, samples.map((s) => s.cut));
  let holds = findHolds(speeds, times, cfg.motionThresh, cfg.minHold);
  holds = splitOnDrift(holds, poses, times, cfg.driftThresh, cfg.minHold);
  holds = splitByLabel(holds, samples.map((s) => s.label), times, cfg.minHold);
  const now = times[times.length - 1];
  for (const h of holds) {
    if (h.label && h.label !== OTHER && now - times[h.end] >= settle) done.add(h.label);
  }
  return done;
}

/** One photo per named pose: the best frame over all its takes (same choice as the Python tool). */
export function bestPerPose(samples, holds) {
  const best = {};
  for (const h of holds) {
    if (!h.label || h.label === OTHER) continue;
    for (const f of h.frames) {
      if (!best[h.label] || f.score > best[h.label].score) best[h.label] = { ...f, hold: h };
    }
  }
  return best;
}

const range = (a, b) => Array.from({ length: b - a + 1 }, (_, k) => a + k);

/**
 * Photo choices for every pose the user says is in the video.
 * Returns {pose: {status: 'found' | 'guess' | 'missing', unsure, choices: [{i, time, score}]}}.
 *   found: recognised; choices[0] is the tool's pick, the rest are the next best distinct moments.
 *   guess: not recognised; the choices are held moments that match no selected pose, for the user to pick.
 * `unsure` is set when another take, or another moment at least 1 s away, is nearly as good as the
 * pick (see `doubtful`), and always when the pose was not recognised.
 */
export function buildChoices(samples, holds, selected, cfg = DEFAULT_CONFIG, { maxChoices = 4, minGap = 0.6, doubt = 0.9 } = {}) {
  const spread = (frames) => {
    const sorted = [...frames].sort((a, b) => b.score - a.score || a.i - b.i);
    const out = [];
    for (const f of sorted) {
      if (out.length >= maxChoices) break;
      if (out.every((c) => Math.abs(samples[c.i].time - samples[f.i].time) >= minGap)) out.push(f);
    }
    return out.map((f) => ({ i: f.i, time: samples[f.i].time, score: f.score, take: f.take }));
  };
  // Real doubt: a nearly-as-good moment from another take, or far from the pick in the same take.
  // Neighbouring frames of one steady hold always score alike, and either of them is fine.
  const doubtful = (choices) => choices.slice(1).some((c) =>
    c.score >= doubt * choices[0].score && (c.take !== choices[0].take || Math.abs(c.time - choices[0].time) >= 1.0));

  const result = {};
  const unmatched = holds.filter((h) => h.label === null || !selected.includes(h.label));
  for (const pose of selected) {
    const frames = holds.filter((h) => h.label === pose).flatMap((h, take) => h.frames.map((f) => ({ ...f, take })));
    if (frames.length) {
      const choices = spread(frames);
      const unsure = doubtful(choices);
      result[pose] = { status: 'found', unsure, choices };
      continue;
    }
    // Not recognised: offer the held moments that were not given to any selected pose, those that
    // look most like this pose first (half photo quality, half likeness to the reference pose).
    // Relaxed holds (arms down) come last.
    const ref = REFERENCES[pose];
    const guesses = [];
    for (const h of unmatched) {
      const scored = scoreHold(samples, { ...h, label: OTHER }, { motionThresh: cfg.motionThresh });
      const penalty = h.label === null ? 0.5 : 1.0;
      for (const f of scored) {
        const d = ref ? poseDistance(samples[f.i].pose, ref) : NaN;
        const likeness = Number.isNaN(d) ? 0 : Math.max(0, 1 - d);
        guesses.push({ i: f.i, score: penalty * (0.5 * f.score + 0.5 * likeness) });
      }
    }
    const choices = spread(guesses);
    result[pose] = { status: choices.length ? 'guess' : 'missing', unsure: true, choices };
  }
  return result;
}
