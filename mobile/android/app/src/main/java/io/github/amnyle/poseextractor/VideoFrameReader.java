package io.github.amnyle.poseextractor;

import android.content.Context;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Decodes a video once, front to back, and hands over about `sampleFps` frames per second as ARGB
 * pixels, downscaled so the longest side is at most `maxSide` and turned upright (the rotation the
 * phone recorded). Much faster than seeking frame by frame: every frame is decoded anyway, but only
 * the sampled ones are converted.
 */
final class VideoFrameReader implements Closeable {

    interface FrameSink {
        /** Return false to stop reading. */
        boolean onFrame(int[] argb, int width, int height, double timeSeconds) throws Exception;
    }

    private final MediaExtractor extractor = new MediaExtractor();
    private MediaCodec codec;
    private final int rotation;
    final double duration;

    VideoFrameReader(Context context, Uri uri) throws IOException {
        extractor.setDataSource(context, uri, null);
        int track = -1;
        MediaFormat format = null;
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat f = extractor.getTrackFormat(i);
            String mime = f.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("video/")) { track = i; format = f; break; }
        }
        if (track < 0) throw new IOException("This file has no video track");
        extractor.selectTrack(track);
        rotation = readRotation(context, uri, format);
        duration = format.containsKey(MediaFormat.KEY_DURATION) ? format.getLong(MediaFormat.KEY_DURATION) / 1e6 : 0;
        String mime = format.getString(MediaFormat.KEY_MIME);
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        codec = MediaCodec.createDecoderByType(mime);
        codec.configure(format, null, null, 0);
        codec.start();
    }

    private static int readRotation(Context context, Uri uri, MediaFormat format) {
        if (format.containsKey(MediaFormat.KEY_ROTATION)) return format.getInteger(MediaFormat.KEY_ROTATION);
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(context, uri);
            String v = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
            return v == null ? 0 : Integer.parseInt(v);
        } catch (RuntimeException e) {
            return 0;
        } finally {
            try { r.release(); } catch (Exception ignored) { }
        }
    }

    /** Read the whole video, calling `sink` for the frames at 0, 1/fps, 2/fps, ... seconds. */
    void read(double sampleFps, int maxSide, FrameSink sink) throws Exception {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false;
        int nextSample = 0;
        double halfFrame = 0.5 / 30.0;
        long lastPts = -1;
        while (true) {
            if (!inputDone) {
                int in = codec.dequeueInputBuffer(10_000);
                if (in >= 0) {
                    ByteBuffer buf = codec.getInputBuffer(in);
                    int size = buf == null ? -1 : extractor.readSampleData(buf, 0);
                    if (size < 0) {
                        codec.queueInputBuffer(in, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        inputDone = true;
                    } else {
                        codec.queueInputBuffer(in, 0, size, extractor.getSampleTime(), 0);
                        extractor.advance();
                    }
                }
            }
            int out = codec.dequeueOutputBuffer(info, 10_000);
            if (out < 0) continue;
            boolean end = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            if (info.size > 0) {
                double t = info.presentationTimeUs / 1e6;
                if (lastPts >= 0 && info.presentationTimeUs > lastPts) {
                    halfFrame = Math.min(0.05, (info.presentationTimeUs - lastPts) / 2e6);
                }
                lastPts = info.presentationTimeUs;
                if (t >= nextSample / sampleFps - halfFrame) {
                    Image image = codec.getOutputImage(out);
                    boolean keepGoing = true;
                    if (image != null) {
                        try {
                            Converted c = convert(image, maxSide);
                            keepGoing = sink.onFrame(c.argb, c.width, c.height, t);
                        } finally {
                            image.close();
                        }
                    }
                    while (nextSample / sampleFps - halfFrame <= t) nextSample++;
                    if (!keepGoing) { codec.releaseOutputBuffer(out, false); return; }
                }
            }
            codec.releaseOutputBuffer(out, false);
            if (end) return;
        }
    }

    private static final class Converted {
        int[] argb;
        int width, height;
    }

    /** YUV 4:2:0 image to upright, downscaled ARGB (nearest sample: the image is only analysed). */
    private Converted convert(Image image, int maxSide) {
        int srcW = image.getCropRect().width(), srcH = image.getCropRect().height();
        int cropX = image.getCropRect().left, cropY = image.getCropRect().top;
        boolean swap = rotation == 90 || rotation == 270;
        int uprightW = swap ? srcH : srcW, uprightH = swap ? srcW : srcH;
        double scale = Math.min(1.0, (double) maxSide / Math.max(uprightW, uprightH));
        int w = Math.max(1, (int) Math.round(uprightW * scale)), h = Math.max(1, (int) Math.round(uprightH * scale));

        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuf = planes[0].getBuffer(), uBuf = planes[1].getBuffer(), vBuf = planes[2].getBuffer();
        int yRow = planes[0].getRowStride(), yPix = planes[0].getPixelStride();
        int uRow = planes[1].getRowStride(), uPix = planes[1].getPixelStride();
        int vRow = planes[2].getRowStride(), vPix = planes[2].getPixelStride();

        int[] argb = new int[w * h];
        for (int oy = 0; oy < h; oy++) {
            for (int ox = 0; ox < w; ox++) {
                // upright output pixel -> pixel of the stored (unrotated) frame
                int ux = Math.min(uprightW - 1, (int) ((ox + 0.5) / scale));
                int uy = Math.min(uprightH - 1, (int) ((oy + 0.5) / scale));
                int sx, sy;
                switch (rotation) {
                    case 90: sx = uy; sy = srcH - 1 - ux; break;
                    case 180: sx = srcW - 1 - ux; sy = srcH - 1 - uy; break;
                    case 270: sx = srcW - 1 - uy; sy = ux; break;
                    default: sx = ux; sy = uy;
                }
                sx += cropX; sy += cropY;
                int Y = yBuf.get(sy * yRow + sx * yPix) & 255;
                int U = (uBuf.get((sy / 2) * uRow + (sx / 2) * uPix) & 255) - 128;
                int V = (vBuf.get((sy / 2) * vRow + (sx / 2) * vPix) & 255) - 128;
                double yy = 1.164 * Math.max(0, Y - 16);
                int r = clamp(yy + 1.596 * V), g = clamp(yy - 0.392 * U - 0.813 * V), b = clamp(yy + 2.017 * U);
                argb[oy * w + ox] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        Converted c = new Converted();
        c.argb = argb;
        c.width = w;
        c.height = h;
        return c;
    }

    private static int clamp(double v) { return v < 0 ? 0 : v > 255 ? 255 : (int) v; }

    @Override
    public void close() {
        if (codec != null) {
            try { codec.stop(); } catch (Exception ignored) { }
            codec.release();
            codec = null;
        }
        extractor.release();
    }
}
