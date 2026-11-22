package io.github.amnyle.poseextractor.logic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/** The live camera takes photos once per pose take, never on relaxed or moving frames. */
public class LiveHoldDetectorTest {

    private static final Set<String> SIX = new HashSet<>(Arrays.asList(
        "front_double_biceps", "back_double_biceps", "front_lat_spread", "back_lat_spread", "abs_and_thighs", "side_chest"));

    private static final class Routine {
        final List<Double> times = new ArrayList<>();
        final List<double[][]> kps = new ArrayList<>();
        final List<int[]> holds = new ArrayList<>();
        final List<String> holdLabels = new ArrayList<>();
    }

    private static List<Routine> load() throws Exception {
        List<Routine> out = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(
                LiveHoldDetectorTest.class.getClassLoader().getResourceAsStream("golden.txt"), StandardCharsets.UTF_8))) {
            Routine r = null;
            for (String line; (line = in.readLine()) != null; ) {
                String[] f = line.split(" ");
                if (f[0].equals("routine")) { r = new Routine(); out.add(r); }
                else if (f[0].equals("frame")) {
                    r.times.add(Double.parseDouble(f[1]));
                    double[] v = Arrays.stream(f[8].split(",")).mapToDouble(Double::parseDouble).toArray();
                    double[][] kp = new double[33][3];
                    for (int j = 0; j < 33; j++) kp[j] = new double[] {v[3 * j], v[3 * j + 1], v[3 * j + 2]};
                    r.kps.add(kp);
                } else if (f[0].equals("hold")) {
                    r.holds.add(new int[] {Integer.parseInt(f[1]), Integer.parseInt(f[2])});
                    r.holdLabels.add(f[3]);
                }
            }
        }
        return out;
    }

    /** The named poses of the routine, in order, a pose split into two holds counted once. */
    private static List<String> takes(Routine r, Set<String> wanted) {
        List<String> out = new ArrayList<>();
        String last = null;
        int lastEnd = -10;
        for (int k = 0; k < r.holds.size(); k++) {
            String label = r.holdLabels.get(k);
            if (!wanted.contains(label)) continue;
            boolean continues = label.equals(last) && r.holds.get(k)[0] - lastEnd <= 2;
            if (!continues) out.add(label);
            last = label;
            lastEnd = r.holds.get(k)[1];
        }
        return out;
    }

    private static List<String> fires(Routine r, Set<String> wanted, List<Integer> at) {
        LiveHoldDetector d = new LiveHoldDetector(new LiveHoldDetector.Config(), wanted);
        List<String> out = new ArrayList<>();
        for (int i = 0; i < r.kps.size(); i++) {
            LiveHoldDetector.Update u = d.push(r.times.get(i), r.kps.get(i));
            if (u.fire) { out.add(u.label); at.add(i); }
        }
        return out;
    }

    @Test
    public void firesOncePerPoseTakeInsideTheHold() throws Exception {
        for (Routine r : load()) {
            List<Integer> at = new ArrayList<>();
            List<String> got = fires(r, SIX, at);
            assertEquals(takes(r, SIX), got);
            // every photo is taken while that pose is actually held
            for (int k = 0; k < at.size(); k++) {
                int i = at.get(k);
                boolean inside = false;
                for (int h = 0; h < r.holds.size(); h++) {
                    int[] span = r.holds.get(h);
                    if (span[0] <= i && i <= span[1] && r.holdLabels.get(h).equals(got.get(k))) inside = true;
                }
                assertTrue("photo " + k + " (" + got.get(k) + ") at frame " + i + " is inside its hold", inside);
            }
        }
    }

    @Test
    public void onlyTheWantedPosesAreTaken() throws Exception {
        Set<String> two = new HashSet<>(Arrays.asList("abs_and_thighs", "side_chest"));
        for (Routine r : load()) {
            List<String> got = fires(r, two, new ArrayList<>());
            assertEquals(takes(r, two), got);
        }
    }

    @Test
    public void holdingAPoseForALongTimeGivesOneBurst() {
        LiveHoldDetector d = new LiveHoldDetector(new LiveHoldDetector.Config(), SIX);
        double[][] kp = doubleBiceps(0);
        int fired = 0;
        for (int i = 0; i < 15 * 20; i++) {          // 20 s at 15 frames/s, tiny jitter
            if (d.push(i / 15.0, doubleBiceps(Math.sin(i) * 0.5)).fire) fired++;
        }
        assertEquals(1, fired);
        // relax for a second, pose again: a new burst
        for (int i = 0; i < 15; i++) assertFalse(d.push(20 + i / 15.0, relaxed()).fire);
        for (int i = 0; i < 30; i++) if (d.push(21 + i / 15.0, kp).fire) fired++;
        assertEquals(2, fired);
    }

    @Test
    public void movingThroughAPoseTakesNothing() {
        LiveHoldDetector d = new LiveHoldDetector(new LiveHoldDetector.Config(), SIX);
        for (int i = 0; i < 60; i++) {
            // arms swing through the double biceps position without stopping
            double[][] kp = doubleBiceps(0);
            double swing = 60 * Math.sin(i / 3.0);
            kp[13][0] += swing; kp[14][0] -= swing; kp[15][1] += swing; kp[16][1] += swing;
            assertFalse(d.push(i / 15.0, kp).fire);
        }
    }

    private static double[][] base() {
        double[][] kp = new double[33][];
        for (int j = 0; j < 33; j++) kp[j] = new double[] {500, 600, 0.0};
        set(kp, 0, 500, 330); set(kp, 7, 480, 335); set(kp, 8, 520, 335);
        set(kp, 11, 580, 420); set(kp, 12, 420, 420);    // facing the camera: left shoulder on the right
        set(kp, 23, 550, 720); set(kp, 24, 450, 720);
        set(kp, 25, 555, 900); set(kp, 26, 445, 900); set(kp, 27, 560, 1080); set(kp, 28, 440, 1080);
        return kp;
    }

    private static double[][] doubleBiceps(double jitter) {
        double[][] kp = base();
        set(kp, 13, 760 + jitter, 400); set(kp, 14, 240 - jitter, 400);
        set(kp, 15, 700, 260 + jitter); set(kp, 16, 300, 260 + jitter);
        return kp;
    }

    private static double[][] relaxed() {
        double[][] kp = base();
        set(kp, 13, 600, 580); set(kp, 14, 400, 580); set(kp, 15, 610, 740); set(kp, 16, 390, 740);
        return kp;
    }

    private static void set(double[][] kp, int j, double x, double y) { kp[j] = new double[] {x, y, 0.99}; }
}
