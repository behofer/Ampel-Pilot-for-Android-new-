/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.ampelpilot;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Vibrator;
import android.support.annotation.Nullable;
import android.support.v7.view.ActionMode;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.ampelpilot.OverlayView.DrawCallback;
import org.tensorflow.ampelpilot.env.BorderedText;
import org.tensorflow.ampelpilot.env.ImageUtils;
import org.tensorflow.ampelpilot.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {


  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";


  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final boolean MAINTAIN_ASPECT = false;

  public static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;


  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;
  LinkedList<String> safe4results = new LinkedList<String>();
  long safe1timestamp = 0;
  public static String stable_light_phase;

  //vibration patterns
  long[] red_pattern = {0, 200, 300, 200, 300, 200};
  int green_pattern = 1000;

  //tts strings
  private String talk_red  = "Es ist rot";
  private String talk_green = "Es ist grün.";
  private String security_instructions = "Benutzen Sie diese App nur als zusätzliche Hilfe! Verlassen Sie sich stets auf ihre eigene Wahrnehmung!";

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  @Nullable
  @Override
  public ActionMode onWindowStartingSupportActionMode(ActionMode.Callback callback) {
      return null;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      Toast.makeText(getApplicationContext(), "Der Classifier konnte nicht initialisiert werden!", Toast.LENGTH_SHORT).show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();

    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
          }
        });

    Toast.makeText(getApplicationContext(), security_instructions, Toast.LENGTH_LONG).show();
    //read out safety instructions
    if (read_instructions) {
        tts.speakUp(security_instructions,  false);
    }
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Paint red = new Paint();
    red.setColor(Color.RED);
    red.setStyle(Style.STROKE);


    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
              if (inference_on) {

                  final long startTime = SystemClock.uptimeMillis();
                  final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                  cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                  final Canvas canvas = new Canvas(cropCopyBitmap);
                  final Paint paint = new Paint();
                  paint.setColor(Color.RED);
                  paint.setStyle(Style.STROKE);
                  paint.setStrokeWidth(2.0f);

                  float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

                  final List<Classifier.Recognition> mappedRecognitions =
                          new LinkedList<Classifier.Recognition>();


                  for (final Classifier.Recognition result : results) {
                      final RectF location = result.getLocation();
                      if (location != null && result.getConfidence() >= minimumConfidence) {

                          canvas.drawRect(location, paint);

                          cropToFrameTransform.mapRect(location);
                          result.setLocation(location);
                          mappedRecognitions.add(result);
                      }
                  }

                  stable_light_phase = "none";

                  Classifier.Recognition biggestRecognition = biggestRecognition(mappedRecognitions);
                  if (biggestRecognition != null) {
                      String currentLight = biggestRecognition.getTitle();

                      safe4results.add(currentLight);
                      if (safe4results.size() > 4) {
                          safe4results.removeFirst();
                      }

                      if (checkStability(safe4results)) {
                          stable_light_phase = safe4results.getFirst();
                          if (System.currentTimeMillis() - safe1timestamp >= 1500) {
                              safe1timestamp = System.currentTimeMillis();
                              if (vibration) {
                                  choose_vibration(currentLight);
                              }
                              if (audio) {
                                  choose_audio(currentLight);
                              }
                          }
                      }
                  } else {
                      safe4results.add("none");
                      safe4results.removeFirst();
                  }

                  System.out.print(safe4results);

                  tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);

              } else {
                  if (System.currentTimeMillis() - safe1timestamp >= 7000) {
                      tts.speakUp("Halten Sie die Kamera bitte hoch!", false);
                      safe1timestamp = System.currentTimeMillis();
                  }
              }
            trackingOverlay.postInvalidate();

            computingDetection = false;
          }
        });
  }

  public void choose_vibration(String currentLightPhase) {
      if (currentLightPhase.equals("red")) {
          vibratePattern(red_pattern);
      } else if (currentLightPhase.equals("green")) {
          vibrate(green_pattern);
      }
  }

  public void choose_audio(String currentLightPhase) {
      if (currentLightPhase.equals("red")) {
          CameraActivity.tts.speakUp(talk_red, true);
      } else if (currentLightPhase.equals("green")) {
          CameraActivity.tts.speakUp(talk_green, true);
      }
  }

  private static boolean checkStability(LinkedList<String> safe4results) {
      if (safe4results.size() == 4) {
          for (int i = 0; i < safe4results.size()-1; i++) {
              if (safe4results.get(i).equals(safe4results.get(i+1))) {}
              else {return false;}
          }
          return true;
      } else {
          return false;
      }
  }

  public void vibrate(int duration) {
      Vibrator vibs = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
      vibs.vibrate(duration);
  }

  public void vibratePattern(long[] pattern) {
        Vibrator vibs = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        vibs.vibrate(pattern, -1);
  }

  public Classifier.Recognition biggestRecognition(List<Classifier.Recognition> mappedRecognitions) {
      Classifier.Recognition biggestResult = null;
      double biggestSize = 0.0;
      for (final Classifier.Recognition result : mappedRecognitions) {
          if (multiplySides(result.getLocation()) > biggestSize) {
              biggestSize = multiplySides(result.getLocation());
              biggestResult = result;
          }
      }
      return biggestResult;
  }

  public double multiplySides(RectF location) {
      return (location.width()*location.height());
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }
}
