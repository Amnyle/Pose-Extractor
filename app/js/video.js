// Browser video helpers: seeking, frame grabbing, scene cuts, sharpness. Mirrors pose_extractor/video.py.

/** Load a File into a muted, inline <video> ready for seeking. */
export async function loadVideo(file) {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = URL.createObjectURL(file);
  await new Promise((resolve, reject) => {
    // metadata, not 'loadeddata': iOS Safari may not decode a frame before the first seek
    video.addEventListener('loadedmetadata', resolve, { once: true });
    video.addEventListener('error', () => reject(new Error('This video format cannot be read on this device.')), { once: true });
  });
  if (!video.videoWidth || !Number.isFinite(video.duration)) throw new Error('This video has no readable picture.');
  await seek(video, 0);
  return video;
}

/** Seek and wait until the frame at `time` can be drawn. */
export function seek(video, time) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      // the 'seeked' frame may not be painted yet in some browsers: wait for it when possible
      if (video.requestVideoFrameCallback) {
        let settled = false;
        const go = () => { if (!settled) { settled = true; resolve(); } };
        video.requestVideoFrameCallback(go);
        setTimeout(go, 120);
      } else resolve();
    };
    const timer = setTimeout(finish, 4000);
    video.addEventListener('seeked', finish, { once: true });
    video.currentTime = Math.min(time, Math.max(0, video.duration - 1e-3));
  });
}

export function analysisSize(video, maxSide) {
  const w = video.videoWidth, h = video.videoHeight;
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

export function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Flags a scene cut when the colour histogram (hue x saturation) changes abruptly. */
export class CutDetector {
  constructor(threshold = 0.5) {
    this.threshold = threshold;
    this.prev = null;
    this.canvas = makeCanvas(160, 90);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  update(source) {
    this.ctx.drawImage(source, 0, 0, 160, 90);
    const px = this.ctx.getImageData(0, 0, 160, 90).data;
    const hist = new Float64Array(16 * 8);
    for (let k = 0; k < px.length; k += 4) {
      const [h, s] = hueSat(px[k], px[k + 1], px[k + 2]);
      hist[Math.min(15, Math.floor(h / (180 / 16))) * 8 + Math.min(7, Math.floor(s / 32))]++;
    }
    let cut = false;
    if (this.prev) cut = correlation(this.prev, hist) < this.threshold;
    this.prev = hist;
    return cut;
  }
}

/** OpenCV-style hue (0..180) and saturation (0..255). */
function hueSat(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  const s = max === 0 ? 0 : (255 * d) / max;
  let h = 0;
  if (d > 0) {
    if (max === r) h = (60 * (g - b)) / d;
    else if (max === g) h = 120 + (60 * (b - r)) / d;
    else h = 240 + (60 * (r - g)) / d;
    if (h < 0) h += 360;
  }
  return [h / 2, s];
}

function correlation(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da * db > 0 ? num / Math.sqrt(da * db) : 1;
}

const sharpCanvas = { c: null, ctx: null };

/** Laplacian variance of the area around the athlete, resized to 256 px high (whole frame without keypoints). */
export function personSharpness(source, width, height, kp, minVis = 0.5) {
  let [x0, y0, x1, y1] = [0, 0, width, height];
  if (kp) {
    const pts = kp.filter((p) => p[2] >= minVis);
    if (pts.length >= 4) {
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      let a = Math.min(...xs), b = Math.min(...ys), c = Math.max(...xs), d = Math.max(...ys);
      const mx = 0.15 * (c - a), my = 0.15 * (d - b);
      a = Math.trunc(Math.max(0, a - mx)); c = Math.trunc(Math.min(width, c + mx));
      b = Math.trunc(Math.max(0, b - my)); d = Math.trunc(Math.min(height, d + my));
      if (c - a >= 16 && d - b >= 16) [x0, y0, x1, y1] = [a, b, c, d];
    }
  }
  const h = 256, w = Math.max(1, Math.trunc(((x1 - x0) * 256) / (y1 - y0)));
  if (!sharpCanvas.c) {
    sharpCanvas.c = makeCanvas(w, h);
    sharpCanvas.ctx = sharpCanvas.c.getContext('2d', { willReadFrequently: true });
  }
  const { c, ctx } = sharpCanvas;
  if (c.width !== w) c.width = w;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const gray = new Float32Array(w * h);
  for (let i = 0, k = 0; i < gray.length; i++, k += 4) gray[i] = 0.299 * px[k] + 0.587 * px[k + 1] + 0.114 * px[k + 2];
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
      sum += v; sum2 += v * v; n++;
    }
  }
  return n ? sum2 / n - (sum / n) ** 2 : 0;
}

/** A frame of the video as a JPEG blob, at full resolution or scaled to `maxSide`. */
export async function grabFrame(video, time, maxSide = Infinity, quality = 0.95) {
  await seek(video, time);
  const [w, h] = analysisSize(video, maxSide);
  const canvas = makeCanvas(w, h);
  canvas.getContext('2d').drawImage(video, 0, 0, w, h);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
}
