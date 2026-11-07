// The app: import a video, say which poses are in it, get the best photo of each, choose between candidates.

import { CLASSIC_POSES, OTHER } from './classic.js';
import { PoseEstimator } from './estimator.js';
import {
  RELEASE_PAGE, checkForUpdate, followAnalysis, getAnalysis, isNative, markAnalysisSeen,
  nativeVideo, openUrl, pickVideo, resultToChoices, runAnalysis, saveFrames,
} from './native.js';
import { DEFAULT_CONFIG } from './pipeline.js';
import { PICTOGRAMS } from './pictograms.js';
import { analyse } from './runner.js';
import { grabFrame, loadVideo } from './video.js';

const NAMES = {
  front_double_biceps: ['Double biceps', 'front'],
  back_double_biceps: ['Double biceps', 'back'],
  front_lat_spread: ['Lat spread', 'front'],
  back_lat_spread: ['Lat spread', 'back'],
  abs_and_thighs: ['Abs & thighs', ''],
  side_chest: ['Side chest', ''],
};
const fullName = (p) => (NAMES[p][1] ? `${NAMES[p][1][0].toUpperCase()}${NAMES[p][1].slice(1)} ${NAMES[p][0].toLowerCase()}` : NAMES[p][0]);
const SPEED_HINTS = {
  4: 'Quickest. Fine for long, steady poses.',
  6: 'Recommended. About 6 frames looked at per second of video.',
  10: 'Slowest, finest timing. Best for short, quick poses.',
};
const BONES = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];

const $ = (id) => document.getElementById(id);

// Inside the Android app, the video work runs natively and in the background (native.js).
const native = isNative;
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
};

const state = {
  file: null,
  video: null,
  selected: new Set(store.get('poses', CLASSIC_POSES).filter((p) => CLASSIC_POSES.includes(p))),
  fps: store.get('fps', 6),
  stopWhenFound: store.get('stopWhenFound', true),
  abort: null,
  estimator: null,
  results: null,   // [{pose, status, unsure, choices: [{time, thumb}], picked: index or -1}]
  thumbs: [],
};

// ---- setup view -------------------------------------------------------------------------------

function pictogram(pose) {
  const p = PICTOGRAMS[pose];
  const lines = BONES.map(([a, b]) => `<line class="bone" x1="${p[a][0]}" y1="${p[a][1]}" x2="${p[b][0]}" y2="${p[b][1]}"/>`).join('');
  return `<svg viewBox="0 0 100 100" aria-hidden="true">${lines}<circle class="head" cx="${p[0][0]}" cy="${p[0][1]}" r="6"/></svg>`;
}

function renderPoseGrid() {
  $('pose-grid').innerHTML = CLASSIC_POSES.map((pose) => `
    <button type="button" class="pose-toggle" data-pose="${pose}" aria-pressed="${state.selected.has(pose)}">
      <span class="check" aria-hidden="true"></span>
      ${pictogram(pose)}
      <span>${NAMES[pose][0]}${NAMES[pose][1] ? `<br><span class="side">${NAMES[pose][1]}</span>` : ''}</span>
    </button>`).join('');
  $('toggle-all').textContent = state.selected.size === CLASSIC_POSES.length ? 'Clear' : 'Select all';
  updateStart();
}

function renderSpeed() {
  for (const b of $('speed').querySelectorAll('button')) b.setAttribute('aria-checked', String(Number(b.dataset.fps) === state.fps));
  $('speed-hint').textContent = SPEED_HINTS[state.fps];
}

function updateStart() {
  $('start').disabled = !state.video || state.selected.size === 0;
  $('start').textContent = state.selected.size
    ? `Find ${state.selected.size === 1 ? 'this pose' : `these ${state.selected.size} poses`}`
    : 'Choose at least one pose';
}

/** A video in the browser, with the same shape as nativeVideo() in native.js. */
function webVideo(element, name) {
  return {
    native: false,
    element,
    name,
    duration: element.duration,
    width: element.videoWidth,
    height: element.videoHeight,
    grab: (time, maxSide = Infinity, quality = 0.95) => grabFrame(element, time, maxSide, quality),
  };
}

async function chooseFile(file) {
  if (!file) return;
  $('video-error').hidden = true;
  try {
    const element = await loadVideo(file);
    if (state.video?.element) URL.revokeObjectURL(state.video.element.src);
    await useVideo(webVideo(element, file.name));
  } catch (err) {
    showError(err.message || 'Could not open this video.');
  }
  updateStart();
}

async function chooseNativeVideo() {
  $('video-error').hidden = true;
  try {
    const video = await pickVideo();
    if (video) await useVideo(video);
  } catch (err) {
    showError(err.message || 'Could not open this video.');
  }
  updateStart();
}

async function useVideo(video) {
  state.video = video;
  const thumb = await video.grab(Math.min(1, video.duration / 2), 240, 0.8);
  $('video-thumb').src = URL.createObjectURL(thumb);
  $('video-name').textContent = video.name;
  $('video-info').textContent = `${formatDuration(video.duration)} · ${video.width}×${video.height}`;
  $('dropzone').classList.add('has-video');
}

function showError(message) {
  $('video-error').textContent = message;
  $('video-error').hidden = false;
}

const formatDuration = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

// ---- analysing view ---------------------------------------------------------------------------

function showAnalysing() {
  document.body.dataset.view = 'analysing';
  window.scrollTo(0, 0);
  state.seen = new Set();
  $('seen').innerHTML = '';
  $('bar').style.width = '0%';
  $('live-label').textContent = '';
  $('status').textContent = 'Starting…';
  $('analysing-hint').textContent = native
    ? 'You can leave the app or lock the phone: it carries on, and a notification tells you when it is done.'
    : 'Keep the app open until it finishes.';
  history.pushState({ view: 'analysing' }, '');
}

function showProgress({ done, total, time, label }) {
  const pct = Math.round((100 * done) / Math.max(1, total));
  $('bar').style.width = `${pct}%`;
  $('progress').setAttribute('aria-valuenow', String(pct));
  $('status').textContent = `Analysing ${time.toFixed(1)} s of ${state.video.duration.toFixed(1)} s`;
  $('live-label').textContent = label && label !== OTHER ? fullName(label) : '';
  if (label && label !== OTHER && !state.seen.has(label)) {
    state.seen.add(label);
    $('seen').insertAdjacentHTML('beforeend', `<span class="chip">✓ ${fullName(label)}</span>`);
  }
}

async function start() {
  const selected = CLASSIC_POSES.filter((p) => state.selected.has(p));
  showAnalysing();
  state.abort = new AbortController();
  if (native) {
    // remembered so the app can pick the analysis up again if Android closes it meanwhile
    store.set('job', { video: { uri: state.video.uri, name: state.video.name, duration: state.video.duration, width: state.video.width, height: state.video.height }, selected, fps: state.fps });
    try {
      const cfg = { ...DEFAULT_CONFIG, sampleFps: state.fps };
      const result = await runAnalysis(state.video, selected, {
        cfg, stopWhenFound: state.stopWhenFound, signal: state.abort.signal, onProgress: showNativeProgress,
      });
      await showNativeResult(result, selected, cfg);
    } catch (err) {
      analysisFailed(err);
    } finally {
      state.abort = null;
    }
    return;
  }
  const wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
  try {
    state.estimator ??= await (window.__createEstimator?.() ?? PoseEstimator.create((s) => { $('status').textContent = s; }));
    const cfg = { ...DEFAULT_CONFIG, sampleFps: state.fps };
    const preview = $('preview');
    const { choices, stoppedAt } = await analyse(state.video.element, state.estimator, selected, {
      cfg,
      stopWhenFound: state.stopWhenFound,
      signal: state.abort.signal,
      onProgress: (p) => {
        drawPreview(preview, p.canvas, p.kp);
        showProgress(p);
      },
    });
    $('status').textContent = 'Preparing the photos…';
    state.stoppedAt = stoppedAt;
    await showResults(selected, choices);
  } catch (err) {
    analysisFailed(err);
  } finally {
    wakeLock?.release().catch(() => {});
    state.abort = null;
  }
}

function analysisFailed(err) {
  showSetup();
  if (err.name !== 'AbortError') {
    console.error(err);
    showError(`Analysis failed: ${err.message || err}`);
  }
}

// ---- native analysis (Android app) -------------------------------------------------------------

function showNativeProgress(p) {
  showProgress(p);
  if (!p.preview) return;
  const img = new Image();
  img.onload = () => {
    const canvas = $('preview');
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
    const s = p.previewScale || 1;
    if (p.kp) drawPreview(canvas, canvas, p.kp.map(([x, y, v]) => [x * s, y * s, v]));
  };
  img.src = `data:image/jpeg;base64,${p.preview}`;
}

async function showNativeResult(result, selected, cfg) {
  $('status').textContent = 'Preparing the photos…';
  const { choices } = resultToChoices(result, selected, cfg);
  state.stoppedAt = result.stoppedAt;
  await showResults(selected, choices);
  markAnalysisSeen().catch(() => {});
}

/** When the app opens: pick up an analysis that is still running, or finished while it was closed. */
async function resumeNative() {
  const job = store.get('job', null);
  if (!job) return;
  const result = await getAnalysis().catch(() => null);
  if (!result || result.status === 'idle' || result.uri !== job.video.uri) return;
  if (result.status !== 'running' && (result.seen || result.status !== 'done')) return;
  state.video = nativeVideo(job.video);
  state.fps = job.fps;
  const cfg = { ...DEFAULT_CONFIG, sampleFps: job.fps };
  showAnalysing();
  state.abort = new AbortController();
  try {
    const done = result.status === 'running'
      ? await followAnalysis({ onProgress: showNativeProgress, signal: state.abort.signal })
      : result;
    await showNativeResult(done, job.selected, cfg);
  } catch (err) {
    analysisFailed(err);
  } finally {
    state.abort = null;
  }
}

function drawPreview(target, source, kp) {
  if (target.width !== source.width || target.height !== source.height) {
    target.width = source.width;
    target.height = source.height;
  }
  const ctx = target.getContext('2d');
  ctx.drawImage(source, 0, 0);
  if (!kp) return;
  const J = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#ff7a33';
  ctx.lineWidth = Math.max(3, source.width / 160);
  ctx.lineCap = 'round';
  ctx.strokeStyle = accent;
  ctx.fillStyle = '#fff';
  for (const [a, b] of BONES) {
    const p = kp[J[a]], q = kp[J[b]];
    if (p[2] < 0.3 || q[2] < 0.3) continue;
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.stroke();
  }
  for (const j of J) {
    if (kp[j][2] < 0.3) continue;
    ctx.beginPath();
    ctx.arc(kp[j][0], kp[j][1], ctx.lineWidth * 0.9, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---- results view -----------------------------------------------------------------------------

async function showResults(selected, choices) {
  for (const url of state.thumbs) URL.revokeObjectURL(url);
  state.thumbs = [];
  const results = [];
  for (const pose of selected) {
    const c = choices[pose];
    const items = [];
    for (const ch of c.choices) {
      const blob = await state.video.grab(ch.time, 720, 0.85);
      const url = URL.createObjectURL(blob);
      state.thumbs.push(url);
      items.push({ time: ch.time, thumb: url });
    }
    // a pose that was recognised starts with the tool's pick; a guess starts with nothing picked
    results.push({ pose, status: c.status, unsure: c.unsure, choices: items, picked: c.status === 'found' ? 0 : -1 });
  }
  const firstTime = (r) => (r.status === 'found' ? r.choices[0].time : Infinity);
  results.sort((a, b) => firstTime(a) - firstTime(b));
  state.results = results;
  renderResults();
  document.body.dataset.view = 'results';
  history.replaceState({ view: 'results' }, '');
  window.scrollTo(0, 0);
}

function showSetup() {
  document.body.dataset.view = 'setup';
  if (history.state?.view) history.back();   // drop the entry pushed by start(): back then exits
  window.scrollTo(0, 0);
}

function renderResults() {
  const ar = `${state.video.width} / ${state.video.height}`;
  const found = state.results.filter((r) => r.status === 'found').length;
  const toCheck = state.results.filter((r) => r.status !== 'missing' && r.unsure).length;
  $('summary').textContent = `${found} of ${state.results.length} pose${state.results.length > 1 ? 's' : ''} found`
    + (toCheck ? ` · ${toCheck} to check` : '')
    + (state.stoppedAt !== null && state.stoppedAt !== undefined
      ? ` · stopped at ${formatDuration(state.stoppedAt)} of ${formatDuration(state.video.duration)}, all poses found`
      : '');

  $('results').innerHTML = state.results.map((r, k) => {
    const badge = r.status === 'found'
      ? (r.unsure ? '<span class="badge check">Check</span>' : '<span class="badge found">Found</span>')
      : r.status === 'guess' ? '<span class="badge check">Pick one</span>' : '<span class="badge missing">Not found</span>';
    const note = r.status === 'found' && r.unsure
      ? '<p class="note">Several moments look equally good. The best guess is selected; tap another to use it instead.</p>'
      : r.status === 'guess'
        ? '<p class="note">Not recognised automatically. If one of these held moments is this pose, tap it.</p>'
        : r.status === 'missing' ? '<p class="note">No held pose was left to suggest. Try “Thorough” detail, or check the pose is held for about a second.</p>' : '';
    const picked = r.picked >= 0 ? r.choices[r.picked] : null;
    const main = picked
      ? `<button type="button" class="main-photo" data-view-photo="${k}" style="--ar:${ar}"><img src="${picked.thumb}" alt="${fullName(r.pose)} at ${picked.time.toFixed(1)} s"></button>`
      : r.choices.length ? '<div class="main-photo skipped">No photo selected for this pose</div>' : '';
    const strip = r.choices.length ? `<div class="choices" role="group" aria-label="Moments to choose from">
        ${r.choices.map((c, i) => `
          <button type="button" class="choice" data-card="${k}" data-choice="${i}" aria-pressed="${i === r.picked}">
            ${i === 0 && r.status === 'found' ? '<span class="tag">Best</span>' : ''}
            <img src="${c.thumb}" alt="" loading="lazy"><span>${c.time.toFixed(1)} s</span>
          </button>`).join('')}
        ${r.status === 'guess' ? `<button type="button" class="choice skip" data-card="${k}" data-choice="-1" aria-pressed="${r.picked === -1}"><span class="box">None</span><span>skip</span></button>` : ''}
      </div>` : '';
    return `<section class="card pose-card">
        <div class="card-head"><h2>${fullName(r.pose)}</h2>${badge}</div>
        ${note}${main}${strip}
      </section>`;
  }).join('');
  const count = state.results.filter((r) => r.picked >= 0).length;
  $('save-all').textContent = count ? `Save ${count} photo${count > 1 ? 's' : ''}` : 'Nothing to save';
  $('save-all').disabled = count === 0;
}

async function saveAll() {
  const picks = state.results.filter((r) => r.picked >= 0);
  if (!picks.length) return;
  const button = $('save-all');
  button.disabled = true;
  button.textContent = 'Preparing…';
  try {
    const name = (r, c, k) => `${String(k + 1).padStart(2, '0')}_${r.pose}_${c.time.toFixed(2)}s.jpg`;
    if (native) {
      const items = picks.map((r, k) => ({ time: r.choices[r.picked].time, fileName: name(r, r.choices[r.picked], k) }));
      await saveFrames(state.video, items);
      toast(`${items.length} photo${items.length > 1 ? 's' : ''} saved to Pictures/Poses`);
      return;
    }
    const files = [];
    for (const [k, r] of picks.entries()) {
      const c = r.choices[r.picked];
      const blob = await state.video.grab(c.time, Infinity, 0.95);
      files.push(new File([blob], name(r, c, k), { type: 'image/jpeg' }));
    }
    if (navigator.canShare?.({ files })) {
      try {
        await navigator.share({ files, title: 'Poses' });
        toast('Shared');
      } catch (err) {
        if (err.name !== 'AbortError') downloadAll(files);
      }
    } else {
      downloadAll(files);
    }
  } catch (err) {
    toast(`Could not save: ${err.message || err}`);
  } finally {
    renderResults();
  }
}

// ---- updates (Android app) --------------------------------------------------------------------

async function offerUpdate() {
  const update = await checkForUpdate();
  if (!update) return;
  $('update-text').textContent = `A new version (${update.version}) is available. Download it from the release page, then open it to install.`;
  $('update').hidden = false;
  $('update-install').onclick = () => openUrl(RELEASE_PAGE);
  $('update-later').onclick = () => { $('update').hidden = true; };
}

function downloadAll(files) {
  for (const f of files) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(f);
    a.download = f.name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  }
  toast(`${files.length} photo${files.length > 1 ? 's' : ''} downloaded`);
}

let toastTimer;
function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
}

// ---- wiring -----------------------------------------------------------------------------------

$('file').addEventListener('change', (e) => chooseFile(e.target.files[0]));
if (native) {
  // Android's own picker gives the native side lasting access to the video
  $('dropzone').addEventListener('click', (e) => {
    e.preventDefault();
    chooseNativeVideo();
  });
}
$('pose-grid').addEventListener('click', (e) => {
  const b = e.target.closest('[data-pose]');
  if (!b) return;
  const pose = b.dataset.pose;
  state.selected.has(pose) ? state.selected.delete(pose) : state.selected.add(pose);
  store.set('poses', [...state.selected]);
  renderPoseGrid();
});
$('toggle-all').addEventListener('click', () => {
  state.selected = state.selected.size === CLASSIC_POSES.length ? new Set() : new Set(CLASSIC_POSES);
  store.set('poses', [...state.selected]);
  renderPoseGrid();
});
$('speed').addEventListener('click', (e) => {
  const b = e.target.closest('[data-fps]');
  if (!b) return;
  state.fps = Number(b.dataset.fps);
  store.set('fps', state.fps);
  renderSpeed();
});
$('stop-when-found').checked = state.stopWhenFound;
$('stop-when-found').addEventListener('change', (e) => {
  state.stopWhenFound = e.target.checked;
  store.set('stopWhenFound', state.stopWhenFound);
});
$('start').addEventListener('click', start);
$('cancel').addEventListener('click', () => state.abort?.abort());
$('results').addEventListener('click', (e) => {
  const choice = e.target.closest('[data-choice]');
  if (choice) {
    state.results[Number(choice.dataset.card)].picked = Number(choice.dataset.choice);
    renderResults();
    return;
  }
  const photo = e.target.closest('[data-view-photo]');
  if (photo) {
    $('viewer-img').src = photo.querySelector('img').src;
    $('viewer').showModal();
  }
});
$('save-all').addEventListener('click', saveAll);
$('again').addEventListener('click', showSetup);
// Back button (phone or browser): from the results or during an analysis, return to the start.
window.addEventListener('popstate', () => {
  if (document.body.dataset.view === 'analysing') state.abort?.abort();
  else if (document.body.dataset.view === 'results') showSetup();
});

renderPoseGrid();
renderSpeed();
if (native) {
  resumeNative();
  offerUpdate();
}

if (!native && 'serviceWorker' in navigator && window.isSecureContext) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
