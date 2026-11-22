package io.github.amnyle.poseextractor;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.drawable.GradientDrawable;
import android.media.MediaActionSound;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import io.github.amnyle.poseextractor.logic.LiveHoldDetector;
import io.github.amnyle.poseextractor.logic.PoseLogic;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * The live "pose taker": the camera watches the athlete, and when a wanted pose is held
 * (LiveHoldDetector) it takes a burst of full-resolution photos, straight into Pictures/Poses.
 * Returns the list of photos ("photos" extra, JSON) when the athlete taps Done.
 */
public class PoseCameraActivity extends AppCompatActivity {

    static final String EXTRA_POSES = "poses", EXTRA_BURST = "burst", EXTRA_FRONT = "front",
        EXTRA_COUNTDOWN = "countdown", EXTRA_VOICE = "voice", EXTRA_FAST = "fast", EXTRA_RESULT = "photos";
    private static final String MODEL = "public/vendor/models/pose_landmarker_full.task";
    private static final String TAG = "PoseCamera";

    private PreviewView preview;
    private CameraOverlay overlay;
    private ProcessCameraProvider provider;
    private ImageCapture capture;
    private final ExecutorService analysis = Executors.newSingleThreadExecutor();
    private PoseLandmarker landmarker;
    private LiveHoldDetector detector;
    private TextToSpeech tts;
    private final MediaActionSound shutter = new MediaActionSound();

    private String[] poses;
    private int burst;
    private boolean front, voice, fast;
    private long lastTs = -1;
    private volatile boolean armed;         // countdown over: photos may be taken
    private volatile boolean shooting;      // a burst is being taken
    private final JSONArray photos = new JSONArray();
    private final String session = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Intent in = getIntent();
        poses = in.getStringArrayExtra(EXTRA_POSES);
        if (poses == null) poses = new String[0];
        burst = Math.max(1, in.getIntExtra(EXTRA_BURST, 3));
        front = in.getBooleanExtra(EXTRA_FRONT, false);
        voice = in.getBooleanExtra(EXTRA_VOICE, true);
        fast = in.getBooleanExtra(EXTRA_FAST, false);
        detector = new LiveHoldDetector(new LiveHoldDetector.Config(), new HashSet<>(Arrays.asList(poses)));

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        preview = new PreviewView(this);
        preview.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        preview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        overlay = new CameraOverlay(this);
        overlay.hint = "Step into the frame";
        root.addView(overlay, new FrameLayout.LayoutParams(-1, -1));
        root.addView(buttons(), new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM));
        setContentView(root);
        // the back button ends the session like Done: the photos taken so far are kept
        getOnBackPressedDispatcher().addCallback(this, new androidx.activity.OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() { done(); }
        });

        if (voice) tts = new TextToSpeech(this, status -> { if (status == TextToSpeech.SUCCESS) tts.setLanguage(Locale.US); });
        shutter.load(MediaActionSound.SHUTTER_CLICK);
        analysis.execute(this::createLandmarker);
        startCamera();
        countdown(in.getIntExtra(EXTRA_COUNTDOWN, 5));
    }

    private View buttons() {
        LinearLayout bar = new LinearLayout(this);
        bar.setGravity(Gravity.CENTER);
        int pad = dp(16);
        bar.setPadding(pad, pad, pad, pad + dp(24));
        bar.addView(button("Flip camera", false, v -> flip()), new LinearLayout.LayoutParams(0, dp(52), 1));
        View gap = new View(this);
        bar.addView(gap, new LinearLayout.LayoutParams(dp(12), 1));
        bar.addView(button("Done", true, v -> done()), new LinearLayout.LayoutParams(0, dp(52), 1));
        return bar;
    }

    private TextView button(String label, boolean primary, View.OnClickListener click) {
        TextView b = new TextView(this);
        b.setText(label);
        b.setGravity(Gravity.CENTER);
        b.setTextSize(16);
        b.setTypeface(b.getTypeface(), android.graphics.Typeface.BOLD);
        b.setTextColor(primary ? 0xFF1A0D05 : Color.WHITE);
        GradientDrawable bg = new GradientDrawable();
        bg.setCornerRadius(dp(14));
        bg.setColor(primary ? 0xFFFF7A33 : 0xB0000000);
        b.setBackground(bg);
        b.setOnClickListener(click);
        return b;
    }

    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    private void countdown(int seconds) {
        armed = false;
        if (seconds <= 0) { arm(); return; }
        overlay.countdown = String.valueOf(seconds);
        overlay.postInvalidate();
        overlay.postDelayed(() -> countdown(seconds - 1), 1000);
    }

    private void arm() {
        overlay.countdown = "";
        overlay.postInvalidate();
        detector.reset();
        armed = true;
        say("Ready");
    }

    private void createLandmarker() {
        try {
            landmarker = PoseLandmarker.createFromOptions(this, options(Delegate.GPU));
        } catch (RuntimeException e) {
            Log.w(TAG, "GPU not available, using the CPU", e);
            landmarker = PoseLandmarker.createFromOptions(this, options(Delegate.CPU));
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
            .build();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            try {
                provider = future.get();
                bind();
            } catch (Exception e) {
                Log.e(TAG, "camera unavailable", e);
                overlay.hint = "The camera could not be opened";
                overlay.postInvalidate();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bind() {
        provider.unbindAll();
        ResolutionSelector fourByThree = new ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build();
        Preview p = new Preview.Builder().setResolutionSelector(fourByThree).build();
        p.setSurfaceProvider(preview.getSurfaceProvider());

        ImageAnalysis a = new ImageAnalysis.Builder()
            .setResolutionSelector(new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(new ResolutionStrategy(new Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build();
        a.setAnalyzer(analysis, this::analyse);

        // full sensor resolution, the phone's best still-photo processing unless "fast" was chosen
        capture = new ImageCapture.Builder()
            .setCaptureMode(fast ? ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY : ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                .build())
            .setJpegQuality(95)
            .build();

        CameraSelector selector = front ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA;
        provider.bindToLifecycle(this, selector, p, a, capture);
    }

    private void flip() {
        if (shooting || provider == null) return;
        front = !front;
        detector.reset();
        bind();
    }

    /** One low-resolution frame: find the pose, update the screen, maybe take the photos. */
    private void analyse(@NonNull ImageProxy image) {
        try {
            if (landmarker == null) return;
            Bitmap raw = image.toBitmap();
            int rot = image.getImageInfo().getRotationDegrees();
            Bitmap up = raw;
            if (rot != 0) {
                Matrix m = new Matrix();
                m.postRotate(rot);
                up = Bitmap.createBitmap(raw, 0, 0, raw.getWidth(), raw.getHeight(), m, true);
                raw.recycle();
            }
            int w = up.getWidth(), h = up.getHeight();
            long ts = Math.max(image.getImageInfo().getTimestamp() / 1_000_000L, lastTs + 1);
            lastTs = ts;
            MPImage mp = new BitmapImageBuilder(up).build();
            PoseLandmarkerResult result = landmarker.detectForVideo(mp, ts);
            mp.close();
            up.recycle();

            double[][] kp = null;
            float[][] points = null;
            if (!result.landmarks().isEmpty()) {
                List<NormalizedLandmark> marks = result.landmarks().get(0);
                kp = new double[33][3];
                points = new float[33][3];
                for (int i = 0; i < 33 && i < marks.size(); i++) {
                    NormalizedLandmark m = marks.get(i);
                    float vis = m.visibility().orElse(0f);
                    kp[i] = new double[] {m.x() * w, m.y() * h, vis};
                    points[i] = new float[] {m.x(), m.y(), vis};
                }
            }
            LiveHoldDetector.Update u = detector.push(ts / 1000.0, kp);
            String shown = u.label == null || PoseLogic.OTHER.equals(u.label) ? null : nice(u.label);
            boolean wantedSeen = u.progress > 0 || u.alreadyTaken;
            overlay.hint = kp == null ? "Step into the frame" : "";
            overlay.update(points, (float) w / h, front, shown, armed && wantedSeen ? (float) u.progress : 0, u.alreadyTaken);
            if (armed && u.fire && !shooting) runOnUiThread(() -> shoot(u.label));
        } catch (Exception e) {
            Log.w(TAG, "frame skipped", e);
        } finally {
            image.close();
        }
    }

    /** A burst of `burst` full-resolution photos, one after the other. */
    private void shoot(String label) {
        shooting = true;
        say(nice(label));
        shootOne(label, 1);
    }

    private void shootOne(String label, int k) {
        String name = "Poses_" + session + "_" + label + "_" + String.format(Locale.US, "%02d", count(label) + 1) + ".jpg";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Poses");
        }
        ImageCapture.OutputFileOptions out = new ImageCapture.OutputFileOptions.Builder(
            getContentResolver(), MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values).build();
        shutter.play(MediaActionSound.SHUTTER_CLICK);
        overlay.flash();
        capture.takePicture(out, ContextCompat.getMainExecutor(this), new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults results) {
                Uri uri = results.getSavedUri();
                try {
                    photos.put(new JSONObject().put("uri", uri == null ? JSONObject.NULL : uri.toString())
                        .put("pose", label).put("name", name).put("time", System.currentTimeMillis()));
                } catch (Exception ignored) { }
                overlay.counts.merge(nice(label), 1, Integer::sum);
                next();
            }

            @Override
            public void onError(@NonNull ImageCaptureException e) {
                Log.e(TAG, "photo failed", e);
                next();
            }

            private void next() {
                if (k < burst && !isFinishing()) overlay.postDelayed(() -> shootOne(label, k + 1), fast ? 150 : 50);
                else shooting = false;
                overlay.postInvalidate();
            }
        });
    }

    private int count(String label) {
        int n = 0;
        for (int i = 0; i < photos.length(); i++) if (label.equals(photos.optJSONObject(i).optString("pose"))) n++;
        return n;
    }

    private void say(String words) {
        if (tts != null) tts.speak(words, TextToSpeech.QUEUE_FLUSH, null, "pose");
    }

    static String nice(String label) {
        switch (label) {
            case "front_double_biceps": return "Front double biceps";
            case "back_double_biceps": return "Back double biceps";
            case "front_lat_spread": return "Front lat spread";
            case "back_lat_spread": return "Back lat spread";
            case "abs_and_thighs": return "Abs and thighs";
            case "side_chest": return "Side chest";
            default: return label.replace('_', ' ');
        }
    }

    private void done() {
        setResult(RESULT_OK, new Intent().putExtra(EXTRA_RESULT, photos.toString()));
        finish();
    }

    @Override
    protected void onDestroy() {
        analysis.execute(() -> { if (landmarker != null) landmarker.close(); });
        analysis.shutdown();
        if (tts != null) tts.shutdown();
        shutter.release();
        super.onDestroy();
    }
}
