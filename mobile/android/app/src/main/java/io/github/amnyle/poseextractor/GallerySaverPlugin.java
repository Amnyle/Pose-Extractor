package io.github.amnyle.poseextractor;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.WindowManager;
import androidx.annotation.RequiresApi;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Native helpers for the web app:
 *   save({data, fileName})  JPEG (base64) into the phone's gallery, album Pictures/Poses
 *   keepAwake({on})          keep the screen on while a video is analysed
 */
@CapacitorPlugin(
    name = "GallerySaver",
    permissions = { @Permission(strings = { Manifest.permission.WRITE_EXTERNAL_STORAGE }, alias = "storage") }
)
public class GallerySaverPlugin extends Plugin {

    private static final String ALBUM = "Poses";

    @PluginMethod
    public void save(PluginCall call) {
        // Android 9 and older need the storage permission to write to the shared Pictures folder
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q && getPermissionState("storage") != PermissionState.GRANTED) {
            requestPermissionForAlias("storage", call, "storagePermission");
            return;
        }
        doSave(call);
    }

    @PermissionCallback
    private void storagePermission(PluginCall call) {
        if (getPermissionState("storage") == PermissionState.GRANTED) {
            doSave(call);
        } else {
            call.reject("Permission to save photos was refused");
        }
    }

    private void doSave(PluginCall call) {
        String data = call.getString("data");
        String fileName = call.getString("fileName", "pose.jpg");
        if (data == null) {
            call.reject("No image data");
            return;
        }
        byte[] bytes;
        try {
            bytes = Base64.decode(data, Base64.DEFAULT);
        } catch (IllegalArgumentException e) {
            call.reject("Image data is not valid base64");
            return;
        }
        try {
            String location = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ? saveMediaStore(bytes, fileName) : saveLegacy(bytes, fileName);
            JSObject result = new JSObject();
            result.put("uri", location);
            call.resolve(result);
        } catch (IOException e) {
            call.reject("Could not save the photo: " + e.getMessage());
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private String saveMediaStore(byte[] bytes, String fileName) throws IOException {
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
        return uri.toString();
    }

    @SuppressWarnings("deprecation")
    private String saveLegacy(byte[] bytes, String fileName) throws IOException {
        File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), ALBUM);
        if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
        File file = new File(dir, fileName);
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(bytes);
        }
        MediaScannerConnection.scanFile(getContext(), new String[] { file.getAbsolutePath() }, new String[] { "image/jpeg" }, null);
        return file.getAbsolutePath();
    }

    @PluginMethod
    public void keepAwake(PluginCall call) {
        boolean on = Boolean.TRUE.equals(call.getBoolean("on", false));
        getActivity().runOnUiThread(() -> {
            if (on) getActivity().getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            else getActivity().getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            call.resolve();
        });
    }
}
