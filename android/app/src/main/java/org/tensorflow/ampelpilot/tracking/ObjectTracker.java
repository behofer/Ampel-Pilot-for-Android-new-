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

import android.graphics.Matrix;
import android.graphics.RectF;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import javax.microedition.khronos.opengles.GL10;
import org.tensorflow.ampelpilot.env.Size;

/**
 * True object detector/tracker class that tracks objects across consecutive preview frames.
 * It provides a simplified Java interface to the analogous native object defined by
 * jni/client_vision/tracking/object_tracker.*.
 *
 * Currently, the ObjectTracker is a singleton due to native code restrictions, and so must
 * be allocated by ObjectTracker.getInstance(). In addition, release() should be called
 * as soon as the ObjectTracker is no longer needed, and before a new one is created.
 *
 * nextFrame() should be called as new frames become available, preferably as often as possible.
 *
 * After allocation, new TrackedObjects may be instantiated via trackObject(). TrackedObjects
 * are associated with the ObjectTracker that created them, and are only valid while that
 * ObjectTracker still exists.
 */
public class ObjectTracker {

  private static boolean libraryFound = false;

  static {
    try {
      System.loadLibrary("tensorflow_demo");
      libraryFound = true;
    } catch (UnsatisfiedLinkError e) {
    }
  }

  private static final int MAX_FRAME_HISTORY_SIZE = 200;

  private static final int DOWNSAMPLE_FACTOR = 2;

  private final byte[] downsampledFrame;

  protected static ObjectTracker instance;

  private final Map<String, TrackedObject> trackedObjects;

  private long lastTimestamp;

  private final LinkedList<TimestampedDeltas> timestampedDeltas;

  protected final int frameWidth;
  protected final int frameHeight;
  private final int rowStride;
  protected final boolean alwaysTrack;

  private static class TimestampedDeltas {
    final long timestamp;
    final byte[] deltas;

    public TimestampedDeltas(final long timestamp, final byte[] deltas) {
      this.timestamp = timestamp;
      this.deltas = deltas;
    }
  }

  public static synchronized ObjectTracker getInstance(
      final int frameWidth, final int frameHeight, final int rowStride, final boolean alwaysTrack) {
    if (!libraryFound) {
      return null;
    }

    if (instance == null) {
      instance = new ObjectTracker(frameWidth, frameHeight, rowStride, alwaysTrack);
      instance.init();
    } else {
      throw new RuntimeException(
          "Tried to create a new objectracker before releasing the old one!");
    }
    return instance;
  }

  public static synchronized void clearInstance() {
    if (instance != null) {
      instance.release();
    }
  }

  protected ObjectTracker(
      final int frameWidth, final int frameHeight, final int rowStride, final boolean alwaysTrack) {
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    this.rowStride = rowStride;
    this.alwaysTrack = alwaysTrack;
    this.timestampedDeltas = new LinkedList<TimestampedDeltas>();

    trackedObjects = new HashMap<String, TrackedObject>();

    downsampledFrame =
        new byte
            [(frameWidth + DOWNSAMPLE_FACTOR - 1)
                / DOWNSAMPLE_FACTOR
                * (frameWidth + DOWNSAMPLE_FACTOR - 1)
                / DOWNSAMPLE_FACTOR];
  }

  protected void init() {
    // The native tracker never sees the full frame, so pre-scale dimensions
    // by the downsample factor.
    initNative(frameWidth / DOWNSAMPLE_FACTOR, frameHeight / DOWNSAMPLE_FACTOR, alwaysTrack);
  }

  private final float[] matrixValues = new float[9];

  private long downsampledTimestamp;

  @SuppressWarnings("unused")
  public synchronized void drawOverlay(final GL10 gl,
      final Size cameraViewSize, final Matrix matrix) {
    final Matrix tempMatrix = new Matrix(matrix);
    tempMatrix.preScale(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR);
    tempMatrix.getValues(matrixValues);
    drawNative(cameraViewSize.width, cameraViewSize.height, matrixValues);
  }

  public synchronized void nextFrame(
      final byte[] frameData, final byte[] uvData,
      final long timestamp, final float[] transformationMatrix) {
    if (downsampledTimestamp != timestamp) {
      ObjectTracker.downsampleImageNative(
          frameWidth, frameHeight, rowStride, frameData, DOWNSAMPLE_FACTOR, downsampledFrame);
      downsampledTimestamp = timestamp;
    }

    nextFrameNative(downsampledFrame, uvData, timestamp, transformationMatrix);

    timestampedDeltas.add(new TimestampedDeltas(timestamp, getKeypointsPacked(DOWNSAMPLE_FACTOR)));
    while (timestampedDeltas.size() > MAX_FRAME_HISTORY_SIZE) {
      timestampedDeltas.removeFirst();
    }

    for (final TrackedObject trackedObject : trackedObjects.values()) {
      trackedObject.updateTrackedPosition();
    }

    lastTimestamp = timestamp;
  }

  public synchronized void release() {
    releaseMemoryNative();
    synchronized (ObjectTracker.class) {
      instance = null;
    }
  }

  public RectF downscaleRect(final RectF fullFrameRect) {
    return new RectF(
        fullFrameRect.left / DOWNSAMPLE_FACTOR,
        fullFrameRect.top / DOWNSAMPLE_FACTOR,
        fullFrameRect.right / DOWNSAMPLE_FACTOR,
        fullFrameRect.bottom / DOWNSAMPLE_FACTOR);
  }

  private RectF upscaleRect(final RectF downsampledFrameRect) {
    return new RectF(
        downsampledFrameRect.left * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.top * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.right * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.bottom * DOWNSAMPLE_FACTOR);
  }

  /**
   * A TrackedObject represents a native TrackedObject, and provides access to the
   * relevant native tracking information available after every frame update. They may
   * be safely passed around and accessed externally, but will become invalid after
   * stopTracking() is called or the related creating ObjectTracker is deactivated.
   *
   * @author andrewharp@google.com (Andrew Harp)
   */
  public class TrackedObject {
    private final String id;

    private long lastExternalPositionTime;

    private RectF lastTrackedPosition;

    private boolean isDead;

    TrackedObject(final RectF position, final long timestamp, final byte[] data) {
      isDead = false;

      id = Integer.toString(this.hashCode());

      lastExternalPositionTime = timestamp;

      synchronized (ObjectTracker.this) {
        registerInitialAppearance(position, data);
        setPreviousPosition(position, timestamp);
        trackedObjects.put(id, this);
      }
    }

    public void stopTracking() {
      checkValidObject();

      synchronized (ObjectTracker.this) {
        isDead = true;
        forgetNative(id);
        trackedObjects.remove(id);
      }
    }

    public float getCurrentCorrelation() {
      checkValidObject();
      return ObjectTracker.this.getCurrentCorrelation(id);
    }

    void registerInitialAppearance(final RectF position, final byte[] data) {
      final RectF externalPosition = downscaleRect(position);
      registerNewObjectWithAppearanceNative(id,
            externalPosition.left, externalPosition.top,
            externalPosition.right, externalPosition.bottom,
            data);
    }

    synchronized void setPreviousPosition(final RectF position, final long timestamp) {
      checkValidObject();
      synchronized (ObjectTracker.this) {
        if (lastExternalPositionTime > timestamp) {
          return;
        }
        final RectF externalPosition = downscaleRect(position);
        lastExternalPositionTime = timestamp;

        setPreviousPositionNative(id,
            externalPosition.left, externalPosition.top,
            externalPosition.right, externalPosition.bottom,
            lastExternalPositionTime);

        updateTrackedPosition();
      }
    }

    private synchronized void updateTrackedPosition() {
      checkValidObject();

      final float[] delta = new float[4];
      getTrackedPositionNative(id, delta);
      lastTrackedPosition = new RectF(delta[0], delta[1], delta[2], delta[3]);

    }

    public synchronized RectF getTrackedPositionInPreviewFrame() {
      checkValidObject();

      if (lastTrackedPosition == null) {
        return null;
      }
      return upscaleRect(lastTrackedPosition);
    }

    private void checkValidObject() {
      if (isDead) {
        throw new RuntimeException("TrackedObject already removed from tracking!");
      } else if (ObjectTracker.this != instance) {
        throw new RuntimeException("TrackedObject created with another ObjectTracker!");
      }
    }
  }

  public synchronized TrackedObject trackObject(
      final RectF position, final long timestamp, final byte[] frameData) {
    if (downsampledTimestamp != timestamp) {
      ObjectTracker.downsampleImageNative(
          frameWidth, frameHeight, rowStride, frameData, DOWNSAMPLE_FACTOR, downsampledFrame);
      downsampledTimestamp = timestamp;
    }
    return new TrackedObject(position, timestamp, downsampledFrame);
  }

  public synchronized TrackedObject trackObject(final RectF position, final byte[] frameData) {
    return new TrackedObject(position, lastTimestamp, frameData);
  }

  /** ********************* NATIVE CODE ************************************ */

  /** This will contain an opaque pointer to the native ObjectTracker */

  private native void initNative(int imageWidth, int imageHeight, boolean alwaysTrack);

  protected native void registerNewObjectWithAppearanceNative(
      String objectId, float x1, float y1, float x2, float y2, byte[] data);

  protected native void setPreviousPositionNative(
      String objectId, float x1, float y1, float x2, float y2, long timestamp);

  protected native void forgetNative(String key);

  protected native float getCurrentCorrelation(String key);

  protected native void getTrackedPositionNative(String key, float[] points);

  protected native void nextFrameNative(
      byte[] frameData, byte[] uvData, long timestamp, float[] frameAlignMatrix);

  protected native void releaseMemoryNative();

  protected native byte[] getKeypointsPacked(float scaleFactor);

  protected native void drawNative(int viewWidth, int viewHeight, float[] frameToCanvas);

  protected static native void downsampleImageNative(
      int width, int height, int rowStride, byte[] input, int factor, byte[] output);
}
