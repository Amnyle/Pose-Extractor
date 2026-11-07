package io.github.amnyle.poseextractor.logic;

/**
 * Per-frame measurements on raw pixels (ARGB ints, row-major), as in app/js/video.js and
 * classic.js: sharpness of the athlete, scene cuts, and the waist width on the person mask.
 */
public final class FrameMetrics {
    private FrameMetrics() {}

    /** Laplacian variance of the area around the athlete, resized to 256 px high (whole frame without keypoints). */
    public static double sharpness(int[] argb, int width, int height, double[][] kp, double minVis) {
        int x0 = 0, y0 = 0, x1 = width, y1 = height;
        if (kp != null) {
            double a = Double.MAX_VALUE, b = Double.MAX_VALUE, c = -Double.MAX_VALUE, d = -Double.MAX_VALUE;
            int n = 0;
            for (double[] p : kp) {
                if (p[2] < minVis) continue;
                a = Math.min(a, p[0]); b = Math.min(b, p[1]); c = Math.max(c, p[0]); d = Math.max(d, p[1]);
                n++;
            }
            if (n >= 4) {
                double mx = 0.15 * (c - a), my = 0.15 * (d - b);
                int ax = (int) Math.max(0, a - mx), cx = (int) Math.min(width, c + mx);
                int by = (int) Math.max(0, b - my), dy = (int) Math.min(height, d + my);
                if (cx - ax >= 16 && dy - by >= 16) { x0 = ax; y0 = by; x1 = cx; y1 = dy; }
            }
        }
        int h = 256, w = Math.max(1, (int) ((x1 - x0) * 256L / (y1 - y0)));
        float[] gray = new float[w * h];
        double sx = (double) (x1 - x0) / w, sy = (double) (y1 - y0) / h;
        for (int y = 0; y < h; y++) {
            // box filter over the source pixels that fall in this output pixel (like a smooth downscale)
            int ys = y0 + (int) (y * sy), ye = Math.max(ys + 1, Math.min(y1, y0 + (int) ((y + 1) * sy)));
            for (int x = 0; x < w; x++) {
                int xs = x0 + (int) (x * sx), xe = Math.max(xs + 1, Math.min(x1, x0 + (int) ((x + 1) * sx)));
                double sum = 0;
                int cnt = 0;
                for (int yy = ys; yy < ye; yy++) {
                    int row = yy * width;
                    for (int xx = xs; xx < xe; xx++) {
                        int p = argb[row + xx];
                        sum += 0.299 * ((p >> 16) & 255) + 0.587 * ((p >> 8) & 255) + 0.114 * (p & 255);
                        cnt++;
                    }
                }
                gray[y * w + x] = (float) (sum / cnt);
            }
        }
        double s = 0, s2 = 0;
        long n = 0;
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                double v = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
                s += v; s2 += v * v; n++;
            }
        }
        return n > 0 ? s2 / n - (s / n) * (s / n) : 0;
    }

    /** Flags a scene cut when the hue x saturation histogram changes abruptly. */
    public static final class CutDetector {
        private final double threshold;
        private double[] prev;
        public CutDetector(double threshold) { this.threshold = threshold; }

        public boolean update(int[] argb, int width, int height) {
            double[] hist = new double[16 * 8];
            for (int gy = 0; gy < 90; gy++) {
                int y = (int) ((gy + 0.5) * height / 90);
                for (int gx = 0; gx < 160; gx++) {
                    int p = argb[y * width + (int) ((gx + 0.5) * width / 160)];
                    int r = (p >> 16) & 255, g = (p >> 8) & 255, b = p & 255;
                    int max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b)), d = max - min;
                    double s = max == 0 ? 0 : 255.0 * d / max, hue = 0;
                    if (d > 0) {
                        if (max == r) hue = 60.0 * (g - b) / d;
                        else if (max == g) hue = 120 + 60.0 * (b - r) / d;
                        else hue = 240 + 60.0 * (r - g) / d;
                        if (hue < 0) hue += 360;
                    }
                    hist[Math.min(15, (int) (hue / 2 / (180.0 / 16))) * 8 + Math.min(7, (int) (s / 32))]++;
                }
            }
            boolean cut = prev != null && correlation(prev, hist) < threshold;
            prev = hist;
            return cut;
        }
    }

    static double correlation(double[] a, double[] b) {
        double ma = 0, mb = 0;
        for (int i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
        ma /= a.length; mb /= b.length;
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < a.length; i++) {
            num += (a[i] - ma) * (b[i] - mb);
            da += (a[i] - ma) * (a[i] - ma);
            db += (b[i] - mb) * (b[i] - mb);
        }
        return da * db > 0 ? num / Math.sqrt(da * db) : 1;
    }

    /** Python's round(): halves go to the even neighbour. */
    static long pyRound(double x) { return (long) Math.rint(x); }

    /**
     * Narrowest waist width in torso lengths on the person mask (mask values above 0.5 are the body),
     * or null. kp must be in the mask's pixels. Port of waistRatio in app/js/classic.js.
     */
    public static Double waistRatio(float[] mask, int width, int height, double[][] kp) {
        double[] hips = PoseLogic.mid(kp[23], kp[24]);
        double torso = PoseLogic.torsoLength(kp);
        if (!Double.isFinite(torso) || torso < PoseLogic.MIN_TORSO_PX) return null;
        int x = (int) pyRound(Math.min(Math.max(hips[0], 0), width - 1));
        double best = Double.MAX_VALUE;
        int count = 0;
        for (int k = 0; k < 10; k++) {
            double f = k == 9 ? 0.6 : 0.15 + k * ((0.6 - 0.15) / 9);
            long y = pyRound(hips[1] - f * torso);
            if (y < 0 || y >= height) continue;
            int row = (int) y * width;
            if (!(mask[row + x] > 0.5f)) continue;
            int l = x, r = x;
            while (l > 0 && mask[row + l - 1] > 0.5f) l--;
            while (r < width - 1 && mask[row + r + 1] > 0.5f) r++;
            best = Math.min(best, (r - l + 1) / torso);
            count++;
        }
        return count >= 5 ? best : null;
    }
}
