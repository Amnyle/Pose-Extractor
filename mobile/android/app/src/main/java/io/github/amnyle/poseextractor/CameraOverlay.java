package io.github.amnyle.poseextractor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.view.View;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Drawn over the camera preview: the athlete's skeleton, the pose being held with a ring that
 * fills up until the photos are taken, a white flash per photo, the countdown, and how many photos
 * each pose has so far.
 */
final class CameraOverlay extends View {

    private static final int ACCENT = 0xFFFF7A33;
    private static final int[][] BONES = {
        {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {11, 23}, {12, 24}, {23, 24},
        {23, 25}, {25, 27}, {24, 26}, {26, 28},
    };

    // set from the camera thread through update(); read when drawing
    private volatile float[][] points;          // 33 x (x, y, visibility), x and y 0..1 in the upright frame
    private volatile float frameAspect = 3f / 4f; // width / height of the upright frame
    private volatile boolean mirror;
    private volatile String label = "";
    private volatile float progress;
    private volatile boolean taken;
    volatile String countdown = "";
    volatile String hint = "";
    final Map<String, Integer> counts = new LinkedHashMap<>();
    private long flashAt;

    private final Paint bone = new Paint(Paint.ANTI_ALIAS_FLAG), joint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pill = new Paint(Paint.ANTI_ALIAS_FLAG), text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG), ringBack = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint big = new Paint(Paint.ANTI_ALIAS_FLAG), flash = new Paint();
    private final float dp;

    CameraOverlay(Context c) {
        super(c);
        dp = c.getResources().getDisplayMetrics().density;
        bone.setColor(ACCENT); bone.setStrokeWidth(4 * dp); bone.setStrokeCap(Paint.Cap.ROUND);
        joint.setColor(Color.WHITE);
        pill.setColor(0xB0000000);
        text.setColor(Color.WHITE); text.setTextSize(16 * dp); text.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        ring.setColor(ACCENT); ring.setStyle(Paint.Style.STROKE); ring.setStrokeWidth(5 * dp); ring.setStrokeCap(Paint.Cap.ROUND);
        ringBack.setColor(0x40FFFFFF); ringBack.setStyle(Paint.Style.STROKE); ringBack.setStrokeWidth(5 * dp);
        big.setColor(Color.WHITE); big.setTextSize(120 * dp); big.setTextAlign(Paint.Align.CENTER);
        big.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD)); big.setShadowLayer(12 * dp, 0, 0, 0x80000000);
        flash.setColor(Color.WHITE);
    }

    void update(float[][] points, float aspect, boolean mirror, String label, float progress, boolean taken) {
        this.points = points;
        this.frameAspect = aspect;
        this.mirror = mirror;
        this.label = label == null ? "" : label;
        this.progress = progress;
        this.taken = taken;
        postInvalidateOnAnimation();
    }

    void flash() {
        flashAt = System.currentTimeMillis();
        postInvalidateOnAnimation();
    }

    /** Where the camera image is on screen: centred, whole (the preview uses FIT_CENTER). */
    private RectF frameRect() {
        float w = getWidth(), h = getHeight();
        if (w / h > frameAspect) {
            float fw = h * frameAspect;
            return new RectF((w - fw) / 2, 0, (w + fw) / 2, h);
        }
        float fh = w / frameAspect;
        return new RectF(0, (h - fh) / 2, w, (h + fh) / 2);
    }

    @Override
    protected void onDraw(Canvas c) {
        RectF r = frameRect();
        float[][] p = points;
        if (p != null) {
            for (int[] b : BONES) {
                float[] a = p[b[0]], z = p[b[1]];
                if (a[2] < 0.3f || z[2] < 0.3f) continue;
                c.drawLine(x(r, a[0]), r.top + a[1] * r.height(), x(r, z[0]), r.top + z[1] * r.height(), bone);
            }
            for (int j : new int[] {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}) {
                if (p[j][2] < 0.3f) continue;
                c.drawCircle(x(r, p[j][0]), r.top + p[j][1] * r.height(), 3.5f * dp, joint);
            }
        }

        // top: the pose being seen, and the ring that fills while it is held
        float top = Math.max(r.top, 0) + 16 * dp + getRootWindowInsetsTop();
        String status = label.isEmpty() ? hint : (taken ? label + "  ✓" : label);
        if (!status.isEmpty()) {
            float tw = text.measureText(status);
            float pillH = 40 * dp, ringR = 12 * dp;
            float left = (getWidth() - (tw + 64 * dp)) / 2;
            RectF box = new RectF(left, top, left + tw + 64 * dp, top + pillH);
            c.drawRoundRect(box, pillH / 2, pillH / 2, pill);
            float cx = box.left + 22 * dp, cy = box.centerY();
            c.drawCircle(cx, cy, ringR, ringBack);
            if (taken) {
                ring.setStyle(Paint.Style.FILL);
                c.drawCircle(cx, cy, ringR, ring);
                ring.setStyle(Paint.Style.STROKE);
            } else if (progress > 0) {
                c.drawArc(new RectF(cx - ringR, cy - ringR, cx + ringR, cy + ringR), -90, 360 * progress, false, ring);
            }
            c.drawText(status, box.left + 44 * dp, cy + text.getTextSize() / 3, text);
        }

        // bottom: photos per pose
        if (!counts.isEmpty()) {
            float y = Math.min(r.bottom, getHeight()) - 104 * dp;
            float x = 16 * dp;
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                String s = e.getKey() + "  " + e.getValue();
                float tw = text.measureText(s) + 24 * dp;
                if (x + tw > getWidth() - 16 * dp) { x = 16 * dp; y -= 44 * dp; }
                c.drawRoundRect(new RectF(x, y, x + tw, y + 36 * dp), 18 * dp, 18 * dp, pill);
                c.drawText(s, x + 12 * dp, y + 24 * dp, text);
                x += tw + 8 * dp;
            }
        }

        if (!countdown.isEmpty()) c.drawText(countdown, getWidth() / 2f, getHeight() / 2f + big.getTextSize() / 3, big);

        long since = System.currentTimeMillis() - flashAt;
        if (since < 250) {
            flash.setAlpha((int) (200 * (1 - since / 250f)));
            c.drawRect(r, flash);
            postInvalidateOnAnimation();
        }
    }

    private float x(RectF r, float nx) { return r.left + (mirror ? 1 - nx : nx) * r.width(); }

    private float getRootWindowInsetsTop() {
        if (android.os.Build.VERSION.SDK_INT >= 23 && getRootWindowInsets() != null) return getRootWindowInsets().getSystemWindowInsetTop();
        return 0;
    }
}
