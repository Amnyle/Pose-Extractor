package io.github.amnyle.poseextractor.logic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The part of the pose logic the background analysis needs to decide when every pose has been
 * seen: pose names and holds. A port of app/js/geometry.js, classic.js and holds.js (themselves a
 * port of the Python tool), checked against the same reference data in PoseLogicTest.
 * Keypoints are double[33][3]: x, y in pixels, visibility.
 */
public final class PoseLogic {
    private PoseLogic() {}

    public static final String OTHER = "other";
    public static final String ABS_AND_THIGHS = "abs_and_thighs";
    static final int[] JOINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28};
    static final double MIN_TORSO_PX = 20.0;

    public static final class Config {
        public double motionThresh = 0.5, minHold = 0.8, driftThresh = 0.3, minVisibility = 0.5, armRaiseMin = 0.15;
    }

    public static final class Features {
        public final double raise, elbowLat, shoulderW, legsDy;
        public final int facing;
        Features(double raise, double elbowLat, double shoulderW, int facing, double legsDy) {
            this.raise = raise; this.elbowLat = elbowLat; this.shoulderW = shoulderW; this.facing = facing; this.legsDy = legsDy;
        }
    }

    public static final class Hold {
        public final int start, end;
        public final String label;
        Hold(int start, int end, String label) { this.start = start; this.end = end; this.label = label; }
        @Override public String toString() { return "[" + start + "," + end + "," + label + "]"; }
    }

    /** What the hold logic needs from one analysed frame. */
    public static final class Frame {
        public final double time;
        public final boolean cut;
        public final double[][] pose;   // normalised (13 x 2) or null
        public final String label;      // pose name, OTHER, or null
        public Frame(double time, boolean cut, double[][] kp, Config cfg) {
            this.time = time;
            this.cut = cut;
            this.pose = kp == null ? null : normalizePose(kp, cfg.minVisibility);
            Features f = pose == null ? null : armFeatures(kp, 0.3);
            this.label = classify(f, cfg.armRaiseMin);
        }
    }

    static double[] mid(double[] a, double[] b) { return new double[] {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2}; }

    static double torsoLength(double[][] kp) {
        double[] s = mid(kp[11], kp[12]), h = mid(kp[23], kp[24]);
        return Math.hypot(s[0] - h[0], s[1] - h[1]);
    }

    public static double[][] normalizePose(double[][] kp, double minVis) {
        if (kp == null || kp.length < 29) return null;
        for (int i : new int[] {11, 12, 23, 24}) if (kp[i][2] < minVis) return null;
        double[] hips = mid(kp[23], kp[24]);
        double torso = torsoLength(kp);
        if (!Double.isFinite(torso) || torso < MIN_TORSO_PX) return null;
        double[][] out = new double[JOINTS.length][2];
        for (int k = 0; k < JOINTS.length; k++) {
            double[] p = kp[JOINTS[k]];
            if (p[2] < minVis) { out[k][0] = Double.NaN; out[k][1] = Double.NaN; }
            else { out[k][0] = (p[0] - hips[0]) / torso; out[k][1] = (p[1] - hips[1]) / torso; }
        }
        return out;
    }

    public static double poseDistance(double[][] a, double[][] b, int topK, int minCommon) {
        double[] d = new double[a.length];
        int n = 0;
        for (int k = 0; k < a.length; k++) {
            double v = Math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]);
            if (!Double.isNaN(v)) d[n++] = v;
        }
        if (n < minCommon) return Double.NaN;
        Arrays.sort(d, 0, n);
        int k = Math.min(topK, n);
        double sum = 0;
        for (int i = n - k; i < n; i++) sum += d[i];
        return sum / k;
    }

    public static Features armFeatures(double[][] kp, double armMinVis) {
        if (kp == null || kp.length < 29) return null;
        for (int i : new int[] {11, 12, 23, 24}) if (kp[i][2] < 0.5) return null;
        double[] shoulders = mid(kp[11], kp[12]), hips = mid(kp[23], kp[24]);
        double torso = torsoLength(kp);
        if (!Double.isFinite(torso) || torso < MIN_TORSO_PX) return null;
        double raise = Double.POSITIVE_INFINITY, elbowSum = 0;
        int wrists = 0, elbows = 0;
        for (int i : new int[] {15, 16}) if (kp[i][2] >= armMinVis) { raise = Math.min(raise, (hips[1] - kp[i][1]) / torso); wrists++; }
        for (int i : new int[] {13, 14}) if (kp[i][2] >= armMinVis) { elbowSum += Math.abs(kp[i][0] - shoulders[0]) / torso; elbows++; }
        if (wrists == 0 || elbows == 0) return null;
        double dx = (kp[11][0] - kp[12][0]) / torso;
        double legsDy = Math.min(kp[27][2], kp[28][2]) >= armMinVis ? Math.abs(kp[27][1] - kp[28][1]) / torso : 0;
        return new Features(raise, elbowSum / elbows, Math.abs(dx), dx >= 0 ? 1 : -1, legsDy);
    }

    public static String classify(Features f, double armRaiseMin) {
        if (f == null || f.raise < armRaiseMin) return null;
        String side = f.facing > 0 ? "front" : "back";
        if (f.raise >= 1.0) {
            if (f.elbowLat >= 0.7) return side + "_double_biceps";
            return side.equals("front") ? ABS_AND_THIGHS : OTHER;
        }
        if (f.shoulderW < 0.55 && f.elbowLat < 0.55) return "side_chest";
        if (f.elbowLat >= 0.6) return side + "_lat_spread";
        return OTHER;
    }

    public static double[] poseSpeeds(List<Frame> frames, int topK) {
        int n = frames.size();
        double[] speeds = new double[n];
        Arrays.fill(speeds, Double.NaN);
        for (int i = 1; i < n; i++) {
            Frame a = frames.get(i - 1), b = frames.get(i);
            if (b.cut || a.pose == null || b.pose == null) continue;
            double dt = b.time - a.time;
            if (dt <= 0) continue;
            double d = poseDistance(a.pose, b.pose, topK, 6);
            if (!Double.isNaN(d)) speeds[i] = d / dt;
        }
        return speeds;
    }

    static double[] median3(double[] v) {
        double[] out = v.clone();
        for (int i = 1; i < v.length - 1; i++) {
            if (Double.isNaN(v[i])) continue;
            double[] w = new double[3];
            int n = 0;
            for (int k = i - 1; k <= i + 1; k++) if (!Double.isNaN(v[k])) w[n++] = v[k];
            Arrays.sort(w, 0, n);
            out[i] = n % 2 == 1 ? w[n / 2] : (w[n / 2 - 1] + w[n / 2]) / 2;
        }
        return out;
    }

    public static List<Hold> findHolds(double[] speeds, double[] times, double motionThresh, double minDuration) {
        double[] smooth = median3(speeds);
        int n = smooth.length;
        boolean[] low = new boolean[n];
        for (int i = 0; i < n; i++) low[i] = !Double.isNaN(smooth[i]) && smooth[i] < motionThresh;
        List<Hold> holds = new ArrayList<>();
        int i = 1;
        while (i < n) {
            if (!low[i]) { i++; continue; }
            int j = i;
            while (j + 1 < n && low[j + 1]) j++;
            int start = i - 1, end = j;
            if (times[end] - times[start] >= minDuration) holds.add(new Hold(start, end, null));
            i = j + 1;
        }
        return holds;
    }

    public static List<Hold> splitOnDrift(List<Hold> holds, List<Frame> frames, double[] times, double driftThresh, double minDuration) {
        List<Hold> out = new ArrayList<>();
        for (Hold h : holds) {
            int segStart = h.start;
            double[][] ref = frames.get(segStart).pose;
            for (int k = h.start + 1; k <= h.end; k++) {
                double[][] p = frames.get(k).pose;
                double d = ref != null && p != null ? poseDistance(ref, p, 4, 6) : Double.NaN;
                if (Double.isNaN(d) || d > driftThresh) {
                    if (times[k - 1] - times[segStart] >= minDuration) out.add(new Hold(segStart, k - 1, null));
                    segStart = k;
                    ref = p;
                }
            }
            if (times[h.end] - times[segStart] >= minDuration) out.add(new Hold(segStart, h.end, null));
        }
        return out;
    }

    public static List<Hold> splitByLabel(List<Hold> holds, List<Frame> frames, double[] times, double minDuration) {
        List<Hold> out = new ArrayList<>();
        for (Hold h : holds) {
            int n = h.end - h.start + 1;
            String[] seq = new String[n];
            for (int k = 0; k < n; k++) seq[k] = frames.get(h.start + k).label;
            for (int i = 1; i < n - 1; i++) {
                if (!Objects.equals(seq[i], seq[i - 1]) && Objects.equals(seq[i - 1], seq[i + 1])) seq[i] = seq[i - 1];
            }
            int run = 0;
            for (int k = 1; k <= n; k++) {
                if (k == n || !Objects.equals(seq[k], seq[run])) {
                    int start = h.start + run, end = h.start + k - 1;
                    if (times[end] - times[start] >= minDuration) out.add(new Hold(start, end, seq[run]));
                    run = k;
                }
            }
        }
        return out;
    }

    /** All labelled holds of the frames so far (same steps as the app and the Python tool). */
    public static List<Hold> holds(List<Frame> frames, Config cfg) {
        double[] times = new double[frames.size()];
        for (int i = 0; i < times.length; i++) times[i] = frames.get(i).time;
        List<Hold> h = findHolds(poseSpeeds(frames, 3), times, cfg.motionThresh, cfg.minHold);
        h = splitOnDrift(h, frames, times, cfg.driftThresh, cfg.minHold);
        return splitByLabel(h, frames, times, cfg.minHold);
    }

    /** Named poses with a finished hold (over for at least `settle` s). See finishedPoses in pipeline.js. */
    public static Set<String> finishedPoses(List<Frame> frames, Config cfg, double settle) {
        Set<String> done = new HashSet<>();
        if (frames.size() < 3) return done;
        double now = frames.get(frames.size() - 1).time;
        for (Hold h : holds(frames, cfg)) {
            if (h.label != null && !h.label.equals(OTHER) && now - frames.get(h.end).time >= settle) done.add(h.label);
        }
        return done;
    }
}
