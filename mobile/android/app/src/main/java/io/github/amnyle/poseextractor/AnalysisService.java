package io.github.amnyle.poseextractor;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Base64;
import android.util.Log;
import androidx.core.app.NotificationCompat;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import io.github.amnyle.poseextractor.logic.FrameMetrics;
import io.github.amnyle.poseextractor.logic.PoseLogic;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Analyses a video in the background (foreground service with a progress notification), so the
 * analysis goes on with the app closed or the screen off. Per sampled frame it records what the
 * app's pose logic needs (keypoints, sharpness, scene cut, waist width) into analysis.json; the
 * app turns that into poses and photo choices. With "stop when found", it stops once every
 * selected pose has a finished hold (PoseLogic.finishedPoses, same rule as the app).
 */
public class AnalysisService extends Service {

    static final String ACTION_START = "start", ACTION_CANCEL = "cancel";
    private static final String CHANNEL_PROGRESS = "analysis", CHANNEL_DONE = "analysis_done";
    private static final int NOTE_PROGRESS = 1, NOTE_DONE = 2;
    private static final String MODEL = "public/vendor/models/pose_landmarker_full.task";
    private static final String TAG = "PoseAnalysis";

    /** Live state shared with the plugin. */
    static final class State {
        volatile String status = "idle";   // idle, running, done, cancelled, failed
        volatile String error, uri;
        volatile int done, total;
        volatile double time, duration, stoppedAt = -1;
        volatile boolean cancel;
    }

    interface Listener {
        void onProgress(JSONObject progress);
        void onFinished(String status);
    }

    static final State STATE = new State();
    static volatile Listener listener;

    private Thread worker;
    private PowerManager.WakeLock wakeLock;

    static File resultFile(Context c) { return new File(c.getFilesDir(), "analysis.json"); }

    /** Small JPEG of every analysed frame, by sample index: the app's thumbnails, read instantly. */
    static File frameDir(Context c) { return new File(c.getCacheDir(), "frames"); }

    static File frameFile(Context c, int index) { return new File(frameDir(c), index + ".jpg"); }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_CANCEL.equals(intent.getAction())) {
            STATE.cancel = true;
            return START_NOT_STICKY;
        }
        if (intent == null || !ACTION_START.equals(intent.getAction()) || "running".equals(STATE.status)) return START_NOT_STICKY;

        createChannels();
        Notification note = progressNote(0, 0, "Starting…");
        if (Build.VERSION.SDK_INT >= 35) startForeground(NOTE_PROGRESS, note, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING);
        else if (Build.VERSION.SDK_INT >= 29) startForeground(NOTE_PROGRESS, note, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        else startForeground(NOTE_PROGRESS, note);

        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Poses:analysis");
        wakeLock.acquire(3 * 60 * 60 * 1000L);

        Uri uri = Uri.parse(intent.getStringExtra("uri"));
        double fps = intent.getDoubleExtra("sampleFps", 6);
        int maxSide = intent.getIntExtra("maxSide", 960);
        boolean stopWhenFound = intent.getBooleanExtra("stopWhenFound", false);
        String[] selected = intent.getStringArrayExtra("selected");
        STATE.status = "running";
        STATE.cancel = false;
        STATE.error = null;
        STATE.uri = uri.toString();
        STATE.done = 0;
        STATE.total = 0;
        STATE.stoppedAt = -1;
        resultFile(this).delete();
        File frames = frameDir(this);
        File[] old = frames.listFiles();
        if (old != null) for (File f : old) f.delete();
        frames.mkdirs();
        worker = new Thread(() -> run(uri, fps, maxSide, stopWhenFound, selected == null ? new String[0] : selected), "pose-analysis");
        worker.start();
        return START_NOT_STICKY;
    }

    private void run(Uri uri, double fps, int maxSide, boolean stopWhenFound, String[] selected) {
        String status = "done";
        JSONArray samples = new JSONArray();
        PoseLandmarker landmarker = null;
        try (VideoFrameReader reader = new VideoFrameReader(this, uri)) {
            STATE.duration = reader.duration;
            STATE.total = Math.max(1, (int) Math.floor(reader.duration * fps) + 1);
            landmarker = createLandmarker();
            final PoseLandmarker lm = landmarker;
            PoseLogic.Config cfg = new PoseLogic.Config();
            List<PoseLogic.Frame> frames = new ArrayList<>();
            FrameMetrics.CutDetector cuts = new FrameMetrics.CutDetector(0.5);
            int checkEvery = Math.max(1, (int) Math.round(fps));
            long[] lastTs = {-1};
            long[] lastNote = {0};

            reader.read(fps, maxSide, (argb, w, h, t) -> {
                if (STATE.cancel) throw new InterruptedException("cancelled");
                boolean cut = cuts.update(argb, w, h);
                Bitmap bmp = Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888);
                long ts = Math.max(Math.round(t * 1000), lastTs[0] + 1);
                lastTs[0] = ts;
                MPImage image = new BitmapImageBuilder(bmp).build();
                PoseLandmarkerResult result = lm.detectForVideo(image, ts);
                double[][] kp = null;
                Double waist = null;
                if (!result.landmarks().isEmpty()) {
                    List<NormalizedLandmark> marks = result.landmarks().get(0);
                    kp = new double[33][3];
                    for (int i = 0; i < 33 && i < marks.size(); i++) {
                        NormalizedLandmark m = marks.get(i);
                        kp[i] = new double[] {m.x() * w, m.y() * h, m.visibility().orElse(0f)};
                    }
                    waist = waist(result, kp, w, h);
                }
                double sharp = FrameMetrics.sharpness(argb, w, h, kp, 0.5);
                int index = samples.length();
                JSONObject s = new JSONObject();
                s.put("index", index).put("time", t).put("width", w).put("height", h)
                    .put("sharpness", sharp).put("cut", cut).put("waist", waist == null ? JSONObject.NULL : waist)
                    .put("kp", kp == null ? JSONObject.NULL : toJson(kp));
                samples.put(s);
                try (FileOutputStream jpg = new FileOutputStream(frameFile(this, index))) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 88, jpg);
                } catch (Exception e) {
                    Log.w(TAG, "thumbnail not stored", e);   // the app falls back to decoding the frame
                }
                PoseLogic.Frame frame = new PoseLogic.Frame(t, cut, kp, cfg);
                frames.add(frame);
                STATE.done = index + 1;
                STATE.time = t;

                long now = System.currentTimeMillis();
                if (now - lastNote[0] > 400) {
                    lastNote[0] = now;
                    updateProgress(t, frame.label);
                    Listener l = listener;
                    if (l != null) l.onProgress(progressJson(bmp, kp, frame.label));
                }
                bmp.recycle();
                image.close();

                if (stopWhenFound && selected.length > 0 && index % checkEvery == 0 && index + 1 < STATE.total) {
                    Set<String> found = PoseLogic.finishedPoses(frames, cfg, 0.5);
                    if (found.containsAll(Arrays.asList(selected))) {
                        STATE.stoppedAt = t;
                        return false;
                    }
                }
                return true;
            });
        } catch (InterruptedException e) {
            status = "cancelled";
        } catch (Throwable e) {
            Log.e(TAG, "analysis failed", e);
            status = "failed";
            STATE.error = e.getMessage() == null ? e.toString() : e.getMessage();
        } finally {
            if (landmarker != null) landmarker.close();
        }
        finish(status, samples);
    }

    private PoseLandmarker createLandmarker() {
        try {
            return PoseLandmarker.createFromOptions(this, options(Delegate.GPU));
        } catch (RuntimeException e) {
            Log.w(TAG, "GPU not available, using the CPU", e);
            return PoseLandmarker.createFromOptions(this, options(Delegate.CPU));
        }
    }

    private static PoseLandmarker.PoseLandmarkerOptions options(Delegate delegate) {
        return PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(MODEL).setDelegate(delegate).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setOutputSegmentationMasks(true)
            .build();
    }

    /** Narrowest waist on the person mask, measured in the mask's own pixels. */
    private static Double waist(PoseLandmarkerResult result, double[][] kp, int w, int h) {
        if (!result.segmentationMasks().isPresent() || result.segmentationMasks().get().isEmpty()) return null;
        MPImage mask = result.segmentationMasks().get().get(0);
        int mw = mask.getWidth(), mh = mask.getHeight();
        FloatBuffer fb = ByteBufferExtractor.extract(mask).order(ByteOrder.nativeOrder()).asFloatBuffer();
        float[] data = new float[mw * mh];
        fb.get(data, 0, Math.min(data.length, fb.remaining()));
        double sx = (double) mw / w, sy = (double) mh / h;
        double[][] mkp = new double[kp.length][];
        for (int i = 0; i < kp.length; i++) mkp[i] = new double[] {kp[i][0] * sx, kp[i][1] * sy, kp[i][2]};
        return FrameMetrics.waistRatio(data, mw, mh, mkp);
    }

    private static JSONArray toJson(double[][] kp) throws Exception {
        JSONArray a = new JSONArray();
        for (double[] p : kp) a.put(new JSONArray().put(p[0]).put(p[1]).put(p[2]));
        return a;
    }

    private JSONObject progressJson(Bitmap bmp, double[][] kp, String label) {
        JSONObject p = new JSONObject();
        try {
            p.put("done", STATE.done).put("total", STATE.total).put("time", STATE.time).put("duration", STATE.duration)
                .put("label", label == null ? JSONObject.NULL : label);
            // a small preview of the frame with its keypoints, for the progress screen
            float s = Math.min(1f, 360f / Math.max(bmp.getWidth(), bmp.getHeight()));
            Bitmap small = Bitmap.createScaledBitmap(bmp, Math.max(1, Math.round(bmp.getWidth() * s)), Math.max(1, Math.round(bmp.getHeight() * s)), true);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            small.compress(Bitmap.CompressFormat.JPEG, 70, out);
            small.recycle();
            p.put("preview", Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP));
            p.put("previewScale", s);
            if (kp != null) p.put("kp", toJson(kp));
        } catch (Exception ignored) { }
        return p;
    }

    private void finish(String status, JSONArray samples) {
        try {
            JSONObject result = new JSONObject();
            result.put("status", status).put("uri", STATE.uri).put("duration", STATE.duration)
                .put("stoppedAt", STATE.stoppedAt >= 0 ? STATE.stoppedAt : JSONObject.NULL)
                .put("error", STATE.error == null ? JSONObject.NULL : STATE.error)
                .put("samples", samples).put("seen", false);
            File tmp = new File(getFilesDir(), "analysis.json.part");
            try (FileOutputStream out = new FileOutputStream(tmp)) {
                out.write(result.toString().getBytes(StandardCharsets.UTF_8));
            }
            if (!tmp.renameTo(resultFile(this))) throw new java.io.IOException("cannot store the result");
        } catch (Exception e) {
            Log.e(TAG, "cannot store the result", e);
            status = "failed";
            STATE.error = "Could not store the result: " + e.getMessage();
        }
        STATE.status = status;
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        stopForeground(STOP_FOREGROUND_REMOVE);
        if (!"cancelled".equals(status)) doneNote(status);
        Listener l = listener;
        if (l != null) l.onFinished(status);
        stopSelf();
    }

    private void createChannels() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(CHANNEL_PROGRESS, "Analysis progress", NotificationManager.IMPORTANCE_LOW));
            nm.createNotificationChannel(new NotificationChannel(CHANNEL_DONE, "Analysis finished", NotificationManager.IMPORTANCE_DEFAULT));
        }
    }

    private PendingIntent openApp() {
        Intent i = new Intent(this, MainActivity.class).setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(this, 0, i, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
    }

    private Notification progressNote(int done, int total, String text) {
        PendingIntent cancel = PendingIntent.getService(this, 1,
            new Intent(this, AnalysisService.class).setAction(ACTION_CANCEL), PendingIntent.FLAG_IMMUTABLE);
        return new NotificationCompat.Builder(this, CHANNEL_PROGRESS)
            .setSmallIcon(R.drawable.ic_stat_pose)
            .setContentTitle("Finding your poses")
            .setContentText(text)
            .setProgress(total, done, total == 0)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openApp())
            .addAction(0, "Cancel", cancel)
            .build();
    }

    private void updateProgress(double t, String label) {
        String pose = label == null || PoseLogic.OTHER.equals(label) ? "" : " · " + label.replace('_', ' ');
        String text = String.format(java.util.Locale.US, "%.0f s of %.0f s%s", t, STATE.duration, pose);
        getSystemService(NotificationManager.class).notify(NOTE_PROGRESS, progressNote(STATE.done, STATE.total, text));
    }

    private void doneNote(String status) {
        boolean ok = "done".equals(status);
        Notification n = new NotificationCompat.Builder(this, CHANNEL_DONE)
            .setSmallIcon(R.drawable.ic_stat_pose)
            .setContentTitle(ok ? "Your poses are ready" : "The analysis failed")
            .setContentText(ok ? "Tap to choose your photos" : STATE.error)
            .setAutoCancel(true)
            .setContentIntent(openApp())
            .build();
        getSystemService(NotificationManager.class).notify(NOTE_DONE, n);
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }

    @Override
    public void onDestroy() {
        STATE.cancel = true;
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        super.onDestroy();
    }
}
