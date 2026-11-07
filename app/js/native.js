// Inside the Android app the video work is native (PosesPlugin.java, AnalysisService.java): faster,
// and it goes on in the background. The pose logic stays here, the same as in the browser.

import { analyseHolds, buildChoices, buildSample, DEFAULT_CONFIG } from './pipeline.js';

export const isNative = window.Capacitor?.isNativePlatform?.() ?? false;

const call = (method, options = {}) => window.Capacitor.nativePromise('Poses', method, options);
const listen = (event, callback) => window.Capacitor.addListener('Poses', event, callback);

function base64ToBlob(data, type = 'image/jpeg') {
  const bin = atob(data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

/** A video known to the native side, with the same shape as the browser's (see app.js). */
export function nativeVideo(info) {
  return {
    native: true,
    uri: info.uri,
    name: info.name,
    duration: info.duration,
    width: info.width,
    height: info.height,
    async grab(time, maxSide = Infinity, quality = 0.95) {
      const { data } = await call('frame', {
        uri: info.uri, time, maxSide: Number.isFinite(maxSide) ? maxSide : 0, quality: Math.round(quality * 100),
      });
      return base64ToBlob(data);
    },
  };
}

/** Open Android's video picker. Resolves to a video, or null if the user backed out. */
export async function pickVideo() {
  try {
    return nativeVideo(await call('pickVideo'));
  } catch (err) {
    if (err?.code === 'CANCELLED') return null;
    throw err;
  }
}

/** Turn the stored native result into poses and photo choices with the app's logic. */
export function resultToChoices(result, selected, cfg = DEFAULT_CONFIG) {
  const samples = result.samples.map((s) => buildSample({ ...s, kp: s.kp ?? null, waist: s.waist ?? null }, cfg));
  const { holds } = analyseHolds(samples, cfg);
  return { samples, holds, choices: buildChoices(samples, holds, selected, cfg) };
}

/**
 * Run the analysis in the background service. onProgress({done, total, time, duration, label,
 * preview (base64 JPEG), previewScale, kp}). Resolves to the stored result {status, samples, stoppedAt, ...}.
 * Aborting `signal` cancels it.
 */
export async function runAnalysis(video, selected, { cfg = DEFAULT_CONFIG, stopWhenFound = false, onProgress = () => {}, signal } = {}) {
  await call('startAnalysis', {
    uri: video.uri, sampleFps: cfg.sampleFps, maxSide: cfg.analysisMaxSide, stopWhenFound, selected,
  });
  return followAnalysis({ onProgress, signal });
}

/** Wait for the running (or just finished) analysis. */
export function followAnalysis({ onProgress = () => {}, signal } = {}) {
  return new Promise((resolve, reject) => {
    const progress = listen('progress', (p) => onProgress(p));
    let finished = null;
    const done = async () => {
      progress.remove();
      finished?.remove();
      const result = await call('getAnalysis');
      if (result.status === 'done') resolve(result);
      else if (result.status === 'cancelled') reject(new DOMException('Analysis cancelled', 'AbortError'));
      else reject(new Error(result.error || 'The analysis failed'));
    };
    finished = listen('finished', done);
    signal?.addEventListener('abort', () => call('cancelAnalysis'), { once: true });
    // it may have finished before we started listening
    call('getAnalysis').then((r) => { if (r.status !== 'running' && r.status !== 'idle') done(); });
  });
}

export const getAnalysis = () => call('getAnalysis');
export const markAnalysisSeen = () => call('markAnalysisSeen');

export const saveFrames = (video, items) => call('saveFrames', { uri: video.uri, items });

// ---- updates ----------------------------------------------------------------------------------

const RELEASE_API = 'https://api.github.com/repos/Amnyle/Pose-Extractor/releases/tags/android-latest';
export const RELEASE_PAGE = 'https://github.com/Amnyle/Pose-Extractor/releases/tag/android-latest';

/** {version, url} of a newer APK than this app, or null. Never throws. */
export async function checkForUpdate() {
  try {
    const [{ versionCode }, release] = await Promise.all([
      call('appInfo'),
      fetch(RELEASE_API, { headers: { Accept: 'application/vnd.github+json' } }).then((r) => (r.ok ? r.json() : null)),
    ]);
    const latest = Number(/\b1\.0\.(\d+)\b/.exec(release?.name ?? '')?.[1]);
    const apk = release?.assets?.find((a) => a.name.endsWith('.apk'));
    if (!apk || !latest || latest <= Number(versionCode)) return null;
    return { version: `1.0.${latest}`, url: apk.browser_download_url };
  } catch {
    return null;
  }
}

/** Download and open the installer. onProgress(fraction). Rejects with err.code SIGNATURE / PERMISSION. */
export async function installUpdate(url, onProgress = () => {}) {
  const sub = listen('updateProgress', (p) => onProgress(p.fraction));
  try {
    await call('installUpdate', { url });
  } finally {
    sub.remove();
  }
}

export const openUrl = (url) => call('openUrl', { url });
