// Pass 1 in the browser: sample the video, detect the pose on each frame, measure it.

import { analyseHolds, buildChoices, buildSample, DEFAULT_CONFIG, finishedPoses } from './pipeline.js';
import { CutDetector, analysisSize, makeCanvas, personSharpness, seek } from './video.js';

/**
 * Analyse `video` and return {samples, holds, choices, stoppedAt}.
 * onProgress({done, total, time, canvas, kp, label}) is called after every frame.
 * With `stopWhenFound`, the analysis stops once every selected pose has a finished hold
 * (`stoppedAt` is then the time it stopped, otherwise null): later takes are not compared.
 * Throws an AbortError if `signal` is aborted.
 */
export async function analyse(video, estimator, selected, { cfg = DEFAULT_CONFIG, onProgress = () => {}, signal, stopWhenFound = false } = {}) {
  const [width, height] = analysisSize(video, cfg.analysisMaxSide);
  const canvas = makeCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const cuts = new CutDetector(cfg.cutThreshold);
  const step = 1 / cfg.sampleFps;
  const total = Math.max(1, Math.floor(video.duration * cfg.sampleFps) + 1);

  const samples = [];
  let stoppedAt = null;
  const checkEvery = Math.max(1, Math.round(cfg.sampleFps));   // once per second of video
  for (let index = 0; index < total; index++) {
    if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');
    const time = index * step;
    if (time > video.duration) break;
    await seek(video, time);
    ctx.drawImage(video, 0, 0, width, height);
    const cut = cuts.update(canvas);
    const { kp, mask, release } = estimator.detect(canvas, width, height, time * 1000);
    try {
      const sharpness = personSharpness(canvas, width, height, kp, cfg.minVisibility);
      samples.push(buildSample({ index, time, kp, width, height, sharpness, cut, mask }, cfg));
    } finally {
      release();
    }
    onProgress({ done: index + 1, total, time, canvas, kp, label: samples[samples.length - 1].label });
    if (stopWhenFound && index % checkEvery === 0 && index + 1 < total) {
      const done = finishedPoses(samples, cfg);
      if (selected.every((p) => done.has(p))) {
        stoppedAt = time;
        break;
      }
    }
  }

  const { holds } = analyseHolds(samples, cfg);
  const choices = buildChoices(samples, holds, selected, cfg);
  return { samples, holds, choices, stoppedAt };
}
