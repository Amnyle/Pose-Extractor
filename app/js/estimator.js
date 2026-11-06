// MediaPipe PoseLandmarker in the browser (runs on the phone, nothing is uploaded).

const BASE = new URL('../vendor/', import.meta.url).href;

export class PoseEstimator {
  static async create(onStatus = () => {}) {
    onStatus('Loading the pose model…');
    const { FilesetResolver, PoseLandmarker } = await import(`${BASE}tasks-vision/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${BASE}tasks-vision`);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: `${BASE}models/pose_landmarker_full.task`, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: true,
    });
    let landmarker;
    try {
      landmarker = await PoseLandmarker.createFromOptions(fileset, options('GPU'));
    } catch {
      landmarker = await PoseLandmarker.createFromOptions(fileset, options('CPU'));
    }
    return new PoseEstimator(landmarker);
  }

  constructor(landmarker) {
    this.landmarker = landmarker;
    this.lastTs = -1;
  }

  /**
   * Detect the pose on `image` (a canvas of `width` x `height`).
   * Returns {kp, mask} where kp is 33 [x, y, visibility] in pixels (or null), and mask a lazy
   * person silhouette scaled to the same pixels as kp, or null. Call `release()` on the result.
   */
  detect(image, width, height, timestampMs) {
    const ts = Math.max(Math.round(timestampMs), this.lastTs + 1);
    this.lastTs = ts;
    const result = this.landmarker.detectForVideo(image, ts);
    const lm = result.landmarks?.[0];
    const kp = lm ? lm.map((p) => [p.x * width, p.y * height, p.visibility ?? 0]) : null;
    const raw = result.segmentationMasks?.[0] ?? null;
    let data = null;
    const mask = raw && {
      width: raw.width,
      height: raw.height,
      sx: raw.width / width,
      sy: raw.height / height,
      get data() {
        data ??= raw.getAsFloat32Array();
        return data;
      },
    };
    return { kp, mask, release: () => result.segmentationMasks?.forEach((m) => m.close()) };
  }

  close() {
    this.landmarker.close();
  }
}
