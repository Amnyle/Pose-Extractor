package io.github.amnyle.poseextractor.logic;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Decides, frame by frame from the live camera, when the athlete is holding a pose: the moment to
 * take the photos. Same ideas as the video analysis (PoseLogic: pose names, stillness measured on
 * the pose normalised by torso length), but online:
 *
 *  - a hold is a run of frames with the same pose name, nearly still (speed below motionThresh)
 *    and not drifting away from where it started; one odd frame (a joint jittering across a
 *    threshold) does not break it;
 *  - when a wanted pose has been held for holdSeconds, push() returns fire = true, once;
 *  - that pose is then not taken again until it has been let go for releaseSeconds (another pose,
 *    relaxed arms, nobody in view), so holding a pose long gives one burst, not a burst per second.
 */
public final class LiveHoldDetector {

    public static final class Config {
        public double holdSeconds = 0.7;
        public double releaseSeconds = 0.6;
        public double motionThresh = 0.5;   // torso lengths / second
        public double driftThresh = 0.3;    // torso lengths from the start of the hold
        public double minVisibility = 0.5;
        public double armRaiseMin = 0.15;
    }

    /** What the camera screen shows after each frame. */
    public static final class Update {
        public String label;          // pose seen in this frame (null: relaxed / nobody), OTHER for unnamed
        public double progress;       // 0..1 towards taking the photos (0 when not a wanted pose)
        public boolean fire;          // take the photos now
        public boolean alreadyTaken;  // a wanted pose, held, but already taken in this hold
    }

    private final Config cfg;
    private final Set<String> wanted;
    private final PoseLogic.Config logicCfg = new PoseLogic.Config();

    // current hold
    private String runLabel;
    private double runStart = Double.NaN;
    private double[][] runPose;
    private boolean runFired;
    // previous frame
    private double[][] prevPose;
    private double prevTime = Double.NaN;
    private final double[] speeds = {Double.NaN, Double.NaN, Double.NaN};
    private int speedIndex;
    private String pendingLabel;        // a different label seen once: maybe just jitter
    // poses taken and not yet let go: label -> time it was last seen
    private final Map<String, Double> blocked = new HashMap<>();

    public LiveHoldDetector(Config cfg, Set<String> wanted) {
        this.cfg = cfg;
        this.wanted = Collections.unmodifiableSet(wanted);
        logicCfg.minVisibility = cfg.minVisibility;
        logicCfg.armRaiseMin = cfg.armRaiseMin;
    }

    /** Forget everything (camera switched, paused...). */
    public void reset() {
        runLabel = null;
        runStart = Double.NaN;
        runPose = null;
        runFired = false;
        prevPose = null;
        prevTime = Double.NaN;
        speeds[0] = speeds[1] = speeds[2] = Double.NaN;
        pendingLabel = null;
        blocked.clear();
    }

    /** One camera frame at `time` seconds; kp = 33 x (x, y, visibility) in pixels, or null if nobody. */
    public Update push(double time, double[][] kp) {
        PoseLogic.Frame frame = new PoseLogic.Frame(time, false, kp, logicCfg);
        Update u = new Update();
        u.label = frame.label;

        // speed since the previous frame, smoothed over the last three (median)
        double speed = Double.NaN;
        if (prevPose != null && frame.pose != null && time > prevTime) {
            double d = PoseLogic.poseDistance(prevPose, frame.pose, 3, 6);
            if (!Double.isNaN(d)) speed = d / (time - prevTime);
        }
        speeds[speedIndex++ % 3] = speed;
        prevPose = frame.pose;
        prevTime = time;
        double smooth = median(speeds);

        // let go of blocked poses that have not been seen for releaseSeconds
        if (frame.label != null) blocked.computeIfPresent(frame.label, (k, v) -> time);
        blocked.entrySet().removeIf(e -> !e.getKey().equals(frame.label) && time - e.getValue() >= cfg.releaseSeconds);

        boolean still = !Double.isNaN(smooth) && smooth < cfg.motionThresh;
        boolean drifted = runPose != null && frame.pose != null
            && PoseLogic.poseDistance(runPose, frame.pose, 4, 6) > cfg.driftThresh;
        boolean sameLabel = java.util.Objects.equals(frame.label, runLabel);

        if (frame.pose == null || !still || drifted) {
            startRun(frame, time);
        } else if (!sameLabel) {
            // one frame with another name is ignored; two in a row start a new hold
            if (pendingLabel != null && pendingLabel.equals(frame.label == null ? "" : frame.label)) startRun(frame, time);
            else pendingLabel = frame.label == null ? "" : frame.label;
        } else {
            pendingLabel = null;
        }

        boolean wantedPose = runLabel != null && wanted.contains(runLabel);
        if (wantedPose && !Double.isNaN(runStart)) {
            if (runFired || blocked.containsKey(runLabel)) {
                u.alreadyTaken = true;
            } else {
                u.progress = Math.min(1.0, (time - runStart) / cfg.holdSeconds);
                if (time - runStart >= cfg.holdSeconds) {
                    u.fire = true;
                    runFired = true;
                    blocked.put(runLabel, time);
                }
            }
        }
        return u;
    }

    private void startRun(PoseLogic.Frame frame, double time) {
        runLabel = frame.label;
        runStart = frame.pose == null ? Double.NaN : time;
        runPose = frame.pose;
        runFired = false;
        pendingLabel = null;
    }

    private static double median(double[] v) {
        double[] w = new double[3];
        int n = 0;
        for (double x : v) if (!Double.isNaN(x)) w[n++] = x;
        if (n == 0) return Double.NaN;
        java.util.Arrays.sort(w, 0, n);
        return n == 2 ? (w[0] + w[1]) / 2 : w[n / 2];
    }
}
