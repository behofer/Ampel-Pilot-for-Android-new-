/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.ampelpilot.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;

import java.util.LinkedList;
import java.util.List;

import org.tensorflow.ampelpilot.CameraActivity;
import org.tensorflow.ampelpilot.Classifier.Recognition;
import org.tensorflow.ampelpilot.DetectorActivity;
import org.tensorflow.ampelpilot.env.BorderedText;
import org.tensorflow.ampelpilot.env.ImageUtils;


/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */
public class MultiBoxTracker {

  private static final float TEXT_SIZE_DIP = 18;

  // Maximum percentage of a box that can be overlapped by another box at detection time. Otherwise
  // the lower scored box (new or old) will be removed.
  private static final float MAX_OVERLAP = 0.2f;

  private static final float MIN_SIZE = 4.0f;

  // Allow replacement of the tracked box with new results if
  // correlation has dropped below this level.
  private static final float MARGINAL_CORRELATION = 0.75f;

  // Consider object to be lost if correlation falls below this threshold.
  private static final float MIN_CORRELATION = 0.3f;

  //value 0 dummy to fill frame
  private final float START = 0.0f;

  public ObjectTracker objectTracker;

  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();

  private static class TrackedRecognition {
    ObjectTracker.TrackedObject trackedObject;
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }

  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();

  private final Paint boxPaint = new Paint();
  private final Paint fillPaint = new Paint();

  private final float textSizePx;
  private final BorderedText borderedText;

  private Matrix frameToCanvasMatrix;

  private int frameWidth;
  private int frameHeight;
  private float rotatedWidth;
  private float rotatedHeight;

  private int sensorOrientation;
  private Context context;

  public MultiBoxTracker(final Context context) {
    this.context = context;

    //paint used to draw bounding boxes
    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(12.0f);
    boxPaint.setStrokeCap(Cap.SQUARE);
    boxPaint.setStrokeJoin(Join.BEVEL);
    boxPaint.setStrokeMiter(100);

    //paint used to fill preview frame
    fillPaint.setStyle(Style.FILL);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }


  public synchronized void trackResults(
      final List<Recognition> results, final byte[] frame, final long timestamp) {
    processResults(timestamp, results, frame);
  }

  public synchronized void draw(final Canvas canvas) {

    final boolean rotated = sensorOrientation % 180 == 90;
    rotatedWidth = rotated ? frameWidth : frameHeight;
    rotatedHeight = rotated ? frameHeight : frameWidth;
    final float multiplier =
        Math.min(canvas.getHeight() / rotatedWidth,
                 canvas.getWidth() / rotatedHeight);

    //camera preview activated: camera frame + detected bounding boxes
    if (CameraActivity.preview) {
      frameToCanvasMatrix =
              ImageUtils.getTransformationMatrix(
                      frameWidth,
                      frameHeight,
                      (int) (multiplier * rotatedHeight),
                      (int) (multiplier * rotatedWidth),
                      sensorOrientation,
                      false);
      for (final TrackedRecognition recognition : trackedObjects) {
        final RectF trackedPos =
                (objectTracker != null)
                        ? recognition.trackedObject.getTrackedPositionInPreviewFrame()
                        : new RectF(recognition.location);

        getFrameToCanvasMatrix().mapRect(trackedPos);
        boxPaint.setColor(recognition.color);

        final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
        canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

        final String labelString =
                !TextUtils.isEmpty(recognition.title)
                        ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                        : String.format("%.2f", recognition.detectionConfidence);
        borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
        }

      //camera preview deactivated: paint canvas in gray, red or green according to stable light phase
    } else {
      if (DetectorActivity.stable_light_phase == null) {
        fillPaint.setColor(Color.GRAY);
      } else if (DetectorActivity.stable_light_phase.equals("red")) {
        fillPaint.setColor(Color.RED);
      } else if (DetectorActivity.stable_light_phase.equals("green")) {
        fillPaint.setColor(Color.GREEN);
      } else {
        fillPaint.setColor(Color.GRAY);
      }
      canvas.drawRect(START, START, canvas.getWidth(), canvas.getHeight(), fillPaint);
    }
  }

  private boolean initialized = false;

  public synchronized void onFrame(
      final int w,
      final int h,
      final int rowStride,
      final int sensorOrientation,
      final byte[] frame,
      final long timestamp) {
    if (objectTracker == null && !initialized) {
      ObjectTracker.clearInstance();

      objectTracker = ObjectTracker.getInstance(w, h, rowStride, true);
      frameWidth = w;
      frameHeight = h;
      this.sensorOrientation = sensorOrientation;
      initialized = true;
    }

    if (objectTracker == null) {
      return;
    }

    objectTracker.nextFrame(frame, null, timestamp, null);

    // Clean up any objects not worth tracking any more.
    final LinkedList<TrackedRecognition> copyList =
        new LinkedList<TrackedRecognition>(trackedObjects);
    for (final TrackedRecognition recognition : copyList) {
      final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;
      final float correlation = trackedObject.getCurrentCorrelation();
      if (correlation < MIN_CORRELATION) {
        trackedObject.stopTracking();
        trackedObjects.remove(recognition);
      }
    }
  }

  private void processResults(
      final long timestamp, final List<Recognition> results, final byte[] originalFrame) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    if (rectsToTrack.isEmpty()) {
      trackedObjects.clear();
      final TrackedRecognition nothing = new TrackedRecognition();
      trackedObjects.add(nothing);
      return;
    }

    if (objectTracker == null) {
      trackedObjects.clear();
      for (final Pair<Float, Recognition> potential : rectsToTrack) {
        final TrackedRecognition trackedRecognition = new TrackedRecognition();
        trackedRecognition.detectionConfidence = potential.first;
        trackedRecognition.location = new RectF(potential.second.getLocation());
        trackedRecognition.trackedObject = null;
        trackedRecognition.title = potential.second.getTitle();
        if (trackedRecognition.title.equals("red")) {
          trackedRecognition.color = Color.RED;
        } else if (trackedRecognition.title.equals("green")) {
          trackedRecognition.color = Color.GREEN;
        }
        trackedObjects.add(trackedRecognition);

      }
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      handleDetection(originalFrame, timestamp, potential);
    }
  }

  private void handleDetection(
      final byte[] frameCopy, final long timestamp, final Pair<Float, Recognition> potential) {
    final ObjectTracker.TrackedObject potentialObject =
        objectTracker.trackObject(potential.second.getLocation(), timestamp, frameCopy);

    final float potentialCorrelation = potentialObject.getCurrentCorrelation();

    if (potentialCorrelation < MARGINAL_CORRELATION) {
      potentialObject.stopTracking();
      return;
    }

    final List<TrackedRecognition> removeList = new LinkedList<TrackedRecognition>();

    float maxIntersect = 0.0f;

    // This is the current tracked object whose color we will take. If left null we'll take the
    // first one from the color queue.
    TrackedRecognition recogToReplace = null;

    // Look for intersections that will be overridden by this object or an intersection that would
    // prevent this one from being placed.
    for (final TrackedRecognition trackedRecognition : trackedObjects) {
      final RectF a = trackedRecognition.trackedObject.getTrackedPositionInPreviewFrame();
      final RectF b = potentialObject.getTrackedPositionInPreviewFrame();
      final RectF intersection = new RectF();
      final boolean intersects = intersection.setIntersect(a, b);

      final float intersectArea = intersection.width() * intersection.height();
      final float totalArea = a.width() * a.height() + b.width() * b.height() - intersectArea;
      final float intersectOverUnion = intersectArea / totalArea;

      // If there is an intersection with this currently tracked box above the maximum overlap
      // percentage allowed, either the new recognition needs to be dismissed or the old
      // recognition needs to be removed and possibly replaced with the new one.
      if (intersects && intersectOverUnion > MAX_OVERLAP) {
        if (potential.first < trackedRecognition.detectionConfidence
            && trackedRecognition.trackedObject.getCurrentCorrelation() > MARGINAL_CORRELATION) {
          // If track for the existing object is still going strong and the detection score was
          // good, reject this new object.
          potentialObject.stopTracking();
          return;
        } else {
          removeList.add(trackedRecognition);

          // Let the previously tracked object with max intersection amount donate its color to
          // the new object.
          if (intersectOverUnion > maxIntersect) {
            maxIntersect = intersectOverUnion;
            recogToReplace = trackedRecognition;
          }
        }
      }
    }

    // If we're already tracking the max object and no intersections were found to bump off,
    // pick the worst current tracked object to remove, if it's also worse than this candidate
    // object.
    if (removeList.isEmpty()) {
      for (final TrackedRecognition candidate : trackedObjects) {
        if (candidate.detectionConfidence < potential.first) {
          if (recogToReplace == null
              || candidate.detectionConfidence < recogToReplace.detectionConfidence) {
            // Save it so that we use this color for the new object.
            recogToReplace = candidate;
          }
        }
      }
      if (recogToReplace != null) {
        removeList.add(recogToReplace);
      }
    }

    // Remove everything that got intersected.
    for (final TrackedRecognition trackedRecognition : removeList) {
      trackedRecognition.trackedObject.stopTracking();
      trackedObjects.remove(trackedRecognition);
    }

    if (recogToReplace == null) {
      potentialObject.stopTracking();
      return;
    }

    // Finally safe to say we can track this object.
    final TrackedRecognition trackedRecognition = new TrackedRecognition();
    trackedRecognition.detectionConfidence = potential.first;
    trackedRecognition.trackedObject = potentialObject;
    trackedRecognition.title = potential.second.getTitle();
  }
}
