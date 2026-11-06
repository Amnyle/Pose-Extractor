// Pose geometry: normalisation and distances. Port of pose_extractor/geometry.py.
// Keypoints are arrays of 33 [x, y, visibility] (x, y in pixels).

export const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const TORSO_IDS = [11, 12, 23, 24];
export const MIN_TORSO_PX = 20.0;

export const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function torsoLength(kp) {
  return dist(mid(kp[11], kp[12]), mid(kp[23], kp[24]));
}

/** (13, 2) pose centred on the hips and divided by torso length; NaN for hidden joints; null if no torso. */
export function normalizePose(kp, minVis = 0.5) {
  if (!kp || kp.length < 29) return null;
  if (TORSO_IDS.some((i) => kp[i][2] < minVis)) return null;
  const hips = mid(kp[23], kp[24]);
  const torso = torsoLength(kp);
  if (!Number.isFinite(torso) || torso < MIN_TORSO_PX) return null;
  return JOINTS.map((j) =>
    kp[j][2] < minVis ? [NaN, NaN] : [(kp[j][0] - hips[0]) / torso, (kp[j][1] - hips[1]) / torso],
  );
}

/** Mean of the `topK` largest per-joint distances, in torso lengths; NaN if too few joints are shared. */
export function poseDistance(a, b, topK = 4, minCommon = 6) {
  const d = [];
  for (let k = 0; k < a.length; k++) {
    const v = Math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]);
    if (!Number.isNaN(v)) d.push(v);
  }
  if (d.length < minCommon) return NaN;
  d.sort((x, y) => x - y);
  const top = d.slice(-Math.min(topK, d.length));
  return top.reduce((s, v) => s + v, 0) / top.length;
}

/** Share of the key joints that are confidently detected and inside the frame. */
export function completeness(kp, width, height, minVis = 0.5) {
  let ok = 0;
  for (const j of JOINTS) {
    const [x, y, v] = kp[j];
    if (v >= minVis && x >= 0 && x < width && y >= 0 && y < height) ok++;
  }
  return ok / JOINTS.length;
}
