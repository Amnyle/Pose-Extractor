package io.github.amnyle.poseextractor;

import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.util.Base64;
import androidx.activity.result.ActivityResult;
import androidx.annotation.RequiresApi;
import androidx.core.content.ContextCompat;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/**
 * Native side of the app (see app/js/native.js):
 *   pickVideo, frame             choose a video, read one of its frames as a JPEG
 *   startAnalysis, cancelAnalysis, getAnalysis, clearAnalysis   background analysis (AnalysisService)
 *   saveFrames                   save full-resolution frames into the gallery (Pictures/Poses)
 *   appInfo, openUrl             update check (the download itself happens in the browser)
 * Events: "progress" and "finished" while an analysis runs.
 */
@CapacitorPlugin(
    name = "Poses",
    permissions = {
        @Permission(strings = { Manifest.permission.WRITE_EXTERNAL_STORAGE }, alias = "storage"),
        @Permission(strings = { Manifest.permission.POST_NOTIFICATIONS }, alias = "notifications"),
    }
)
public class PosesPlugin extends Plugin {

    private static final String ALBUM = "Poses";

    @Override
    public void load() {
        AnalysisService.listener = new AnalysisService.Listener() {
            @Override
            public void onProgress(JSONObject progress) {
                try { notifyListeners("progress", new JSObject(progress.toString())); } catch (Exception ignored) { }
            }

            @Override
            public void onFinished(String status) {
                JSObject o = new JSObject();
                o.put("status", status);
                notifyListeners("finished", o);
            }
        };
    }

    // ---- video ---------------------------------------------------------------------------------

    @PluginMethod
    public void pickVideo(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("video/*");
        startActivityForResult(call, intent, "videoPicked");
    }

    @ActivityCallback
    private void videoPicked(PluginCall call, ActivityResult result) {
        if (call == null) return;
        if (result.getResultCode() != Activity.RESULT_OK || result.getData() == null || result.getData().getData() == null) {
            call.reject("No video chosen", "CANCELLED");
            return;
        }
        Uri uri = result.getData().getData();
        try {
            getContext().getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // not every provider offers lasting access; the video is still readable now
        }
        try {
            call.resolve(describe(uri));
        } catch (Exception e) {
            call.reject("This video cannot be read: " + e.getMessage());
        }
    }

    private JSObject describe(Uri uri) throws IOException {
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(getContext(), uri);
            int w = parse(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
            int h = parse(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
            int rot = parse(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION));
            double duration = parse(r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)) / 1000.0;
            if (w == 0 || h == 0) throw new IOException("no picture");
            JSObject o = new JSObject();
            o.put("uri", uri.toString());
            o.put("name", displayName(uri));
            o.put("duration", duration);
            o.put("width", rot == 90 || rot == 270 ? h : w);
            o.put("height", rot == 90 || rot == 270 ? w : h);
            return o;
        } finally {
            r.release();
        }
    }

    private static int parse(String v) {
        try { return v == null ? 0 : Integer.parseInt(v.trim()); } catch (NumberFormatException e) { return 0; }
    }

    private String displayName(Uri uri) {
        try (Cursor c = getContext().getContentResolver().query(uri, new String[] { OpenableColumns.DISPLAY_NAME }, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) { }
        return "video";
    }

    /** The frame at `time` seconds (upright), as JPEG, scaled to `maxSide` (0 = full size). */
    private byte[] frameJpeg(MediaMetadataRetriever r, double time, int maxSide, int quality) throws IOException {
        Bitmap frame = r.getFrameAtTime(Math.round(time * 1e6), MediaMetadataRetriever.OPTION_CLOSEST);
        if (frame == null) throw new IOException("frame not found at " + time + " s");
        Bitmap out = frame;
        int longest = Math.max(frame.getWidth(), frame.getHeight());
        if (maxSide > 0 && longest > maxSide) {
            float s = (float) maxSide / longest;
            out = Bitmap.createScaledBitmap(frame, Math.round(frame.getWidth() * s), Math.round(frame.getHeight() * s), true);
            frame.recycle();
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        out.compress(Bitmap.CompressFormat.JPEG, quality, bytes);
        out.recycle();
        return bytes.toByteArray();
    }

    @PluginMethod
    public void frame(PluginCall call) {
        String uri = call.getString("uri");
        double time = call.getDouble("time", 0.0);
        int maxSide = call.getInt("maxSide", 720);
        int quality = call.getInt("quality", 85);
        getBridge().execute(() -> {
            MediaMetadataRetriever r = new MediaMetadataRetriever();
            try {
                r.setDataSource(getContext(), Uri.parse(uri));
                JSObject o = new JSObject();
                o.put("data", Base64.encodeToString(frameJpeg(r, time, maxSide, quality), Base64.NO_WRAP));
                call.resolve(o);
            } catch (Exception e) {
                call.reject("Could not read the frame: " + e.getMessage());
            } finally {
                try { r.release(); } catch (Exception ignored) { }
            }
        });
    }

    // ---- analysis ------------------------------------------------------------------------------

    @PluginMethod
    public void startAnalysis(PluginCall call) {
        if (Build.VERSION.SDK_INT >= 33 && getPermissionState("notifications") != PermissionState.GRANTED) {
            // asked once: the analysis also runs without it, only the notification is hidden
            requestPermissionForAlias("notifications", call, "notificationsAnswered");
            return;
        }
        launch(call);
    }

    @PermissionCallback
    private void notificationsAnswered(PluginCall call) { launch(call); }

    private void launch(PluginCall call) {
        JSArray selected = call.getArray("selected", new JSArray());
        String[] poses = new String[selected.length()];
        for (int i = 0; i < poses.length; i++) poses[i] = selected.optString(i);
        Intent i = new Intent(getContext(), AnalysisService.class)
            .setAction(AnalysisService.ACTION_START)
            .putExtra("uri", call.getString("uri"))
            .putExtra("sampleFps", call.getDouble("sampleFps", 6.0))
            .putExtra("maxSide", call.getInt("maxSide", 960))
            .putExtra("stopWhenFound", call.getBoolean("stopWhenFound", false))
            .putExtra("selected", poses);
        ContextCompat.startForegroundService(getContext(), i);
        call.resolve();
    }

    @PluginMethod
    public void cancelAnalysis(PluginCall call) {
        AnalysisService.STATE.cancel = true;
        call.resolve();
    }

    /** {status, done, total, time, duration} while running; the stored result (with samples) after. */
    @PluginMethod
    public void getAnalysis(PluginCall call) {
        AnalysisService.State s = AnalysisService.STATE;
        if ("running".equals(s.status)) {
            JSObject o = new JSObject();
            o.put("status", "running");
            o.put("uri", s.uri);
            o.put("done", s.done);
            o.put("total", s.total);
            o.put("time", s.time);
            o.put("duration", s.duration);
            call.resolve(o);
            return;
        }
        getBridge().execute(() -> {
            File f = AnalysisService.resultFile(getContext());
            if (!f.isFile()) {
                JSObject o = new JSObject();
                o.put("status", "idle");
                call.resolve(o);
                return;
            }
            try (InputStream in = new FileInputStream(f)) {
                byte[] bytes = new byte[(int) f.length()];
                int read = 0;
                while (read < bytes.length) {
                    int n = in.read(bytes, read, bytes.length - read);
                    if (n < 0) break;
                    read += n;
                }
                call.resolve(new JSObject(new String(bytes, 0, read, StandardCharsets.UTF_8)));
            } catch (Exception e) {
                call.reject("Could not read the analysis: " + e.getMessage());
            }
        });
    }

    /** Mark the stored result as shown, so it is not offered again when the app reopens. */
    @PluginMethod
    public void markAnalysisSeen(PluginCall call) {
        getBridge().execute(() -> {
            File f = AnalysisService.resultFile(getContext());
            try {
                if (f.isFile()) {
                    JSONObject o = new JSONObject(new String(java.nio.file.Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8));
                    o.put("seen", true);
                    try (OutputStream out = new FileOutputStream(f)) { out.write(o.toString().getBytes(StandardCharsets.UTF_8)); }
                }
                call.resolve();
            } catch (Exception e) {
                call.reject(e.getMessage());
            }
        });
    }

    // ---- saving --------------------------------------------------------------------------------

    /** saveFrames({uri, items: [{time, fileName}]}): full-resolution frames into Pictures/Poses. */
    @PluginMethod
    public void saveFrames(PluginCall call) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q && getPermissionState("storage") != PermissionState.GRANTED) {
            requestPermissionForAlias("storage", call, "storageAnswered");
            return;
        }
        doSaveFrames(call);
    }

    @PermissionCallback
    private void storageAnswered(PluginCall call) {
        if (getPermissionState("storage") == PermissionState.GRANTED) doSaveFrames(call);
        else call.reject("Permission to save photos was refused");
    }

    private void doSaveFrames(PluginCall call) {
        String uri = call.getString("uri");
        JSArray items = call.getArray("items", new JSArray());
        getBridge().execute(() -> {
            MediaMetadataRetriever r = new MediaMetadataRetriever();
            try {
                r.setDataSource(getContext(), Uri.parse(uri));
                for (int k = 0; k < items.length(); k++) {
                    JSONObject item = items.getJSONObject(k);
                    byte[] jpeg = frameJpeg(r, item.getDouble("time"), 0, 95);
                    String name = item.getString("fileName");
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) saveMediaStore(jpeg, name);
                    else saveLegacy(jpeg, name);
                }
                JSObject o = new JSObject();
                o.put("saved", items.length());
                call.resolve(o);
            } catch (Exception e) {
                call.reject("Could not save the photos: " + e.getMessage());
            } finally {
                try { r.release(); } catch (Exception ignored) { }
            }
        });
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private void saveMediaStore(byte[] bytes, String fileName) throws IOException {
        ContentResolver resolver = getContext().getContentResolver();
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + ALBUM);
        values.put(MediaStore.Images.Media.IS_PENDING, 1);
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("the gallery refused the photo");
        try (OutputStream out = resolver.openOutputStream(uri)) {
            if (out == null) throw new IOException("cannot write the photo");
            out.write(bytes);
        } catch (IOException e) {
            resolver.delete(uri, null, null);
            throw e;
        }
        values.clear();
        values.put(MediaStore.Images.Media.IS_PENDING, 0);
        resolver.update(uri, values, null, null);
    }

    @SuppressWarnings("deprecation")
    private void saveLegacy(byte[] bytes, String fileName) throws IOException {
        File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), ALBUM);
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
        File file = new File(dir, fileName);
        try (FileOutputStream out = new FileOutputStream(file)) { out.write(bytes); }
        MediaScannerConnection.scanFile(getContext(), new String[] { file.getAbsolutePath() }, new String[] { "image/jpeg" }, null);
    }

    // ---- updates -------------------------------------------------------------------------------

    @PluginMethod
    public void appInfo(PluginCall call) {
        try {
            PackageInfo p = getContext().getPackageManager().getPackageInfo(getContext().getPackageName(), 0);
            JSObject o = new JSObject();
            o.put("versionCode", Build.VERSION.SDK_INT >= 28 ? p.getLongVersionCode() : p.versionCode);
            o.put("versionName", p.versionName);
            call.resolve(o);
        } catch (PackageManager.NameNotFoundException e) {
            call.reject(e.getMessage());
        }
    }

    @PluginMethod
    public void openUrl(PluginCall call) {
        Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(call.getString("url"))).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        getContext().startActivity(i);
        call.resolve();
    }

}
