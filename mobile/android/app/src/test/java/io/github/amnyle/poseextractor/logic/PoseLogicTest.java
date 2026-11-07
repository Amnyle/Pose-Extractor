package io.github.amnyle.poseextractor.logic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * The background analysis must name poses, find holds and decide "all poses found" exactly like the
 * app and the Python tool. golden.txt comes from mobile/scripts/make-java-golden.mjs.
 */
public class PoseLogicTest {

    private static final class Routine {
        String name;
        final List<double[][]> kps = new ArrayList<>();
        final List<Double> times = new ArrayList<>();
        final List<Boolean> cuts = new ArrayList<>();
        final List<String> labels = new ArrayList<>();
        final List<String> waists = new ArrayList<>();
        final List<int[]> masks = new ArrayList<>();
        final List<String> holds = new ArrayList<>();
        final List<String[]> finished = new ArrayList<>();
        int width, height;
    }

    private static List<Routine> load() throws Exception {
        List<Routine> routines = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                PoseLogicTest.class.getClassLoader().getResourceAsStream("golden.txt"), StandardCharsets.UTF_8))) {
            Routine r = null;
            for (String line; (line = in.readLine()) != null; ) {
                String[] f = line.split(" ");
                switch (f[0]) {
                    case "routine": r = new Routine(); r.name = f[1]; routines.add(r); break;
                    case "frame":
                        r.times.add(Double.parseDouble(f[1]));
                        r.cuts.add(f[2].equals("1"));
                        r.labels.add(f[3].equals("-") ? null : f[3]);
                        r.waists.add(f[4]);
                        r.masks.add(f[5].equals("-") ? null : Arrays.stream(f[5].split(",")).mapToInt(Integer::parseInt).toArray());
                        r.width = Integer.parseInt(f[6]);
                        r.height = Integer.parseInt(f[7]);
                        double[] v = Arrays.stream(f[8].split(",")).mapToDouble(Double::parseDouble).toArray();
                        double[][] kp = new double[33][3];
                        for (int j = 0; j < 33; j++) kp[j] = new double[] {v[3 * j], v[3 * j + 1], v[3 * j + 2]};
                        r.kps.add(kp);
                        break;
                    case "hold": r.holds.add("[" + f[1] + "," + f[2] + "," + (f[3].equals("-") ? null : f[3]) + "]"); break;
                    case "finished": r.finished.add(new String[] {f[1], f[2]}); break;
                    default: throw new IllegalStateException(line);
                }
            }
        }
        return routines;
    }

    private static List<PoseLogic.Frame> frames(Routine r, int n) {
        PoseLogic.Config cfg = new PoseLogic.Config();
        List<PoseLogic.Frame> out = new ArrayList<>();
        for (int i = 0; i < n; i++) out.add(new PoseLogic.Frame(r.times.get(i), r.cuts.get(i), r.kps.get(i), cfg));
        return out;
    }

    @Test
    public void labelsAndHoldsMatchTheReference() throws Exception {
        for (Routine r : load()) {
            List<PoseLogic.Frame> frames = frames(r, r.kps.size());
            for (int i = 0; i < frames.size(); i++) assertEquals(r.name + " frame " + i, r.labels.get(i), frames.get(i).label);
            List<String> holds = new ArrayList<>();
            for (PoseLogic.Hold h : PoseLogic.holds(frames, new PoseLogic.Config())) holds.add(h.toString());
            assertEquals(r.name, r.holds, holds);
        }
    }

    @Test
    public void allPosesFoundIsDecidedAtTheSameMomentAsInTheApp() throws Exception {
        for (Routine r : load()) {
            for (String[] check : r.finished) {
                int n = Integer.parseInt(check[0]);
                Set<String> got = new TreeSet<>(PoseLogic.finishedPoses(frames(r, n), new PoseLogic.Config(), 0.5));
                String expected = check[1].equals("-") ? "" : check[1];
                assertEquals(r.name + " after " + n + " frames", expected, String.join(",", got));
            }
        }
    }

    @Test
    public void waistWidthMatchesThePythonTool() throws Exception {
        int checked = 0;
        for (Routine r : load()) {
            for (int i = 0; i < r.kps.size(); i++) {
                int[] rect = r.masks.get(i);
                if (rect == null || r.waists.get(i).equals("-")) continue;
                float[] mask = new float[r.width * r.height];
                for (int y = rect[2]; y < rect[3]; y++) Arrays.fill(mask, y * r.width + rect[0], y * r.width + rect[1], 1f);
                Double w = FrameMetrics.waistRatio(mask, r.width, r.height, r.kps.get(i));
                assertNotNull(w);
                assertEquals(r.name + " frame " + i, Double.parseDouble(r.waists.get(i)), w, 1e-9);
                checked++;
            }
        }
        assertEquals(true, checked > 50);
    }

    @Test
    public void sharpnessPrefersAFocusedAthleteAndCutsAreDetected() {
        int w = 200, h = 300;
        int[] sharp = new int[w * h], blur = new int[w * h], other = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int v = ((x / 4 + y / 4) % 2) * 255;               // fine checkerboard
                int b = 128 + (int) (40 * Math.sin(x / 15.0));      // smooth gradient
                sharp[y * w + x] = 0xff000000 | (v << 16) | (v << 8) | v;
                blur[y * w + x] = 0xff000000 | (b << 16) | (b << 8) | b;
                other[y * w + x] = 0xff000000 | (200 << 16) | (30 << 8) | 30;   // red scene
            }
        }
        assertEquals(true, FrameMetrics.sharpness(sharp, w, h, null, 0.5) > 10 * FrameMetrics.sharpness(blur, w, h, null, 0.5));
        FrameMetrics.CutDetector cuts = new FrameMetrics.CutDetector(0.5);
        assertEquals(false, cuts.update(blur, w, h));
        assertEquals(false, cuts.update(blur, w, h));
        assertEquals(true, cuts.update(other, w, h));
    }

    @Test
    public void relaxedArmsAreNotAPose() {
        assertNull(PoseLogic.classify(null, 0.15));
    }
}
