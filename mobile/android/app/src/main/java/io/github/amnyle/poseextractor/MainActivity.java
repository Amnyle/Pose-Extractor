package io.github.amnyle.poseextractor;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(PosesPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
