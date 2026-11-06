# Android app

The phone app in `../app`, packaged as a real Android app with [Capacitor](https://capacitorjs.com):
same screens and pose logic, plus native saving of the photos to the gallery (`Pictures/Poses`)
and keeping the screen on during an analysis (`GallerySaverPlugin.java`).

## Install on your phone

GitHub builds the APK on every push to `main` (`.github/workflows/android.yml`) and publishes it as
the **android-latest** release. On your Android phone, open

    https://github.com/Amnyle/Pose-Extractor/releases/tag/android-latest

and tap **Poses.apk**. Android asks to allow installing apps from your browser the first time: allow
it, then tap **Install**. The app appears as **Poses** with the orange icon.

## Updates and signing

Android installs a new version over the old one only if both are signed with the same key.

- **Without setup**, each build is signed with a throw-away debug key: uninstall the old app before
  installing a new build.
- **With your own key**, every build installs as an update. Do this once on a computer with Java:

  ```
  keytool -genkeypair -keystore poses.keystore -alias poses -keyalg RSA -keysize 2048 -validity 10000
  base64 -w0 poses.keystore > poses.keystore.b64          # macOS: base64 -i poses.keystore
  ```

  Then in the repository on GitHub, Settings > Secrets and variables > Actions, add the secrets
  `POSES_KEYSTORE_BASE64` (the content of `poses.keystore.b64`), `POSES_KEYSTORE_PASSWORD`,
  `POSES_KEY_ALIAS` (`poses`) and `POSES_KEY_PASSWORD` (the passwords you chose). Keep
  `poses.keystore` somewhere safe and never commit it. Uninstall the debug-signed app once; from then
  on, new builds install as updates.

## Build it yourself

Needs Node 22, Java 21 and the Android SDK.

```
../app/scripts/fetch-vendor.sh      # MediaPipe runtime + pose model into app/vendor
npm ci
npx cap sync android                # copies ../app into the Android project
cd android && ./gradlew assembleDebug
```

The APK is in `android/app/build/outputs/apk/debug/`. Or open `android/` in Android Studio.
