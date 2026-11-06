// The app's JavaScript must make the same decisions as the Python tool.
// Run: node --test app/tests     (regenerate the reference with: python app/tests/make_golden.py)

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { CLASSIC_POSES, armFeatures, classify } from '../js/classic.js';
import { analyseHolds, bestPerPose, buildChoices, buildSample } from '../js/pipeline.js';

const golden = JSON.parse(readFileSync(new URL('./golden.json', import.meta.url)));
const userFrames = JSON.parse(readFileSync(new URL('../../tests/data/user_frames.json', import.meta.url)));

const close = (a, b, tol = 1e-9) => (a === null ? b === null : Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)));

function mask(rect, width, height) {
  const data = new Float32Array(width * height);
  const [x0, x1, y0, y1] = rect;
  for (let y = y0; y < y1; y++) data.fill(1, y * width + x0, y * width + x1);
  return { width, height, data };
}

function analyse(input) {
  const samples = input.map((s) => buildSample({ ...s, mask: s.maskRect ? mask(s.maskRect, s.width, s.height) : null }));
  const { holds } = analyseHolds(samples);
  return { samples, holds };
}

test('the real calibration frames get the same names as in Python', () => {
  const expected = {
    pose_01: 'abs_and_thighs', pose_04: 'front_double_biceps', pose_08: 'side_chest',
    pose_12: 'back_double_biceps', pose_13: 'back_lat_spread', pose_14: 'front_lat_spread',
  };
  for (const [name, kp] of Object.entries(userFrames)) assert.equal(classify(armFeatures(kp)), expected[name] ?? null, name);
});

for (const g of golden) {
  test(`routine ${g.seed}: same labels, measures, holds and picks as Python`, () => {
    const { samples, holds } = analyse(g.input);
    g.expected.samples.forEach((e, i) => {
      const s = samples[i];
      assert.equal(s.label, e.label, `label of sample ${i}`);
      assert.ok(close(s.strength, e.strength), `strength of sample ${i}`);
      assert.ok(close(s.speed, e.speed), `speed of sample ${i}: ${s.speed} vs ${e.speed}`);
      assert.ok(close(s.waist, e.waist), `waist of sample ${i}: ${s.waist} vs ${e.waist}`);
      assert.ok(close(s.headDrop, e.headDrop), `head of sample ${i}`);
    });
    assert.deepEqual(holds.map((h) => [h.start, h.end, h.label]), g.expected.holds);
    holds.forEach((h, k) => {
      const e = g.expected.picks[k];
      if (e === null) return;
      const best = h.frames.reduce((b, f) => (b === null || f.score > b.score ? f : b), null);
      assert.equal(best.i, e[0], `pick of hold ${k} (${h.label})`);
      assert.ok(close(best.score, e[1]), `score of hold ${k}`);
    });
    const best = bestPerPose(samples, holds);
    for (const r of g.expected.results) {
      assert.ok(close(samples[best[r.label].i].time, r.time), `time of ${r.label}`);
      assert.ok(close(best[r.label].score, r.score), `score of ${r.label}`);
    }
  });
}

test('choices: the first choice is the tool pick, the others are distinct moments', () => {
  const { samples, holds } = analyse(golden[0].input);
  const choices = buildChoices(samples, holds, CLASSIC_POSES);
  const best = bestPerPose(samples, holds);
  for (const pose of CLASSIC_POSES) {
    const c = choices[pose];
    assert.equal(c.status, 'found', pose);
    assert.equal(c.choices[0].i, best[pose].i, pose);
    assert.ok(c.choices.length >= 2 && c.choices.length <= 4, pose);
    for (let a = 0; a < c.choices.length; a++)
      for (let b = a + 1; b < c.choices.length; b++)
        assert.ok(Math.abs(c.choices[a].time - c.choices[b].time) >= 0.6 - 1e-9, pose);
  }
});

test('choices: a pose the user says is there but was not recognised gets guesses from unclaimed holds', () => {
  const { samples, holds } = analyse(golden[0].input);
  // pretend the athlete's side chest was never recognised: drop its holds
  const without = holds.filter((h) => h.label !== 'side_chest');
  const c = buildChoices(samples, without, ['abs_and_thighs', 'side_chest'])['side_chest'];
  assert.equal(c.status, 'guess');
  assert.equal(c.unsure, true);
  assert.ok(c.choices.length > 0);
  const abs = new Set(without.filter((h) => h.label === 'abs_and_thighs').flatMap((h) => h.frames.map((f) => f.i)));
  assert.ok(c.choices.every((x) => !abs.has(x.i)), 'never offers frames of a recognised selected pose');
});

test('choices: nothing held at all means missing', () => {
  const c = buildChoices([], [], ['side_chest']);
  assert.deepEqual(c.side_chest, { status: 'missing', unsure: true, choices: [] });
});

test('choices: guesses for an unrecognised pose start with the moment that looks most like it', () => {
  const { samples, holds } = analyse(golden[0].input);
  // pretend side chest and back lat spread were not recognised (they come out as "other")
  const blind = holds.map((h) => (['side_chest', 'back_lat_spread'].includes(h.label) ? { ...h, label: 'other' } : h));
  const frames = (label) => new Set(holds.filter((h) => h.label === label).flatMap((h) => h.frames.map((f) => f.i)));
  const c = buildChoices(samples, blind, CLASSIC_POSES);
  assert.equal(c.side_chest.status, 'guess');
  assert.ok(frames('side_chest').has(c.side_chest.choices[0].i), 'side chest guess');
  assert.ok(frames('back_lat_spread').has(c.back_lat_spread.choices[0].i), 'back lat spread guess');
});

test('choices: near-equal neighbours in one steady hold are not doubt, another equally good take is', () => {
  const pose = [[0, -1], [0.3, -1], [-0.3, -1], [0.6, -1], [-0.6, -1], [0.5, -1.4], [-0.5, -1.4], [0.15, 0], [-0.15, 0], [0.15, 0.5], [-0.15, 0.5], [0.15, 1], [-0.15, 1]];
  const samples = Array.from({ length: 30 }, (_, i) => ({ time: i / 6, pose }));
  const hold = (start, end, scores) => ({ start, end, label: 'side_chest', frames: scores.map((score, k) => ({ i: start + k, score })) });
  const steady = buildChoices(samples, [hold(0, 7, [0.70, 0.79, 0.80, 0.80, 0.79, 0.78, 0.71, 0.60])], ['side_chest']).side_chest;
  assert.equal(steady.unsure, false);
  const twoTakes = buildChoices(samples, [hold(0, 7, [0.7, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6]), hold(15, 22, [0.6, 0.79, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])], ['side_chest']).side_chest;
  assert.equal(twoTakes.unsure, true);
});
