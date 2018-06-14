package com.aeloyq.tflitedemo.test;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Build;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Size;
import android.widget.Toast;

import com.aeloyq.tflitedemo.CameraActivity;
import com.aeloyq.tflitedemo.Classifier;
import com.aeloyq.tflitedemo.DetectorADASFaceUintActivity;
import com.aeloyq.tflitedemo.env.BorderedText;

import org.tensorflow.lite.Interpreter;

import java.util.Vector;

public class MinorImplementActivity extends AppCompatActivity {

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE = "adas_face_ssdlite_mobilenet_v2_baseline_quantized.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/adas_face_list.txt";

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  // Only return this many results.
  private static final int NUM_RESULTS = 1917;
  private static final int NUM_CLASSES = 91;

  private static final float Y_SCALE = 10.0f;
  private static final float X_SCALE = 10.0f;
  private static final float H_SCALE = 5.0f;
  private static final float W_SCALE = 5.0f;

  // Config values.
  private int inputSize = 300;

  private final float[][] boxPriors = new float[4][NUM_RESULTS];

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues = new int[inputSize * inputSize];
  private float[][][][] image_float = new float[1][inputSize][inputSize][3];
  private Float[][][] outputLocations_float = new Float[1][NUM_RESULTS][4];
  private Float[][][] outputClasses_float = new Float[1][NUM_RESULTS][NUM_CLASSES];
  private byte[][][][] image_uint = new byte[1][inputSize][inputSize][3];
  private byte[][][] outputLocations_uint = new byte[1][NUM_RESULTS][4];
  private byte[][][] outputClasses_uint = new byte[1][NUM_RESULTS][NUM_CLASSES];

  private Interpreter tfLite;// Which detection model to use: by default uses Tensorflow Object Detection API frozen

  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(com.aeloyq.tflitedemo.R.layout.activity_minor_implement);
    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }
  }

  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    if (requestCode == PERMISSIONS_REQUEST) {
      if (grantResults.length > 0
              && grantResults[0] == PackageManager.PERMISSION_GRANTED
              && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
              checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
              shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
        Toast.makeText(CameraActivity.this,
                "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
    }
  }
}
