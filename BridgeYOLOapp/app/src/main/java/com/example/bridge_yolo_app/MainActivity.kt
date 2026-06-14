package com.example.bridge_yolo_app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import androidx.core.graphics.scale


class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: CameraOverlayView
    private lateinit var interpreter: Interpreter

    private val INPUT_SIZE = 640
    private val CONF_THRESHOLD = 0.25f
    private val IOU_THRESHOLD = 0.45f

    private var SCREEN_HEIGHT = 0
    private var SCREEN_WIDTH = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val displayMetrics = DisplayMetrics()
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics)
        SCREEN_HEIGHT = displayMetrics.heightPixels
        SCREEN_WIDTH = displayMetrics.widthPixels

        viewFinder = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            setupModel()
            startCamera()
        } else {
            requestPermissions()
        }
    }

    private fun setupModel() {
        try {
            interpreter = Interpreter(loadModelFile(this, "model.tflite"))
            Log.d("YOLO", "Output shape: " + interpreter.getOutputTensor(0).shape().joinToString())

            Log.d("YOLO", "Model loaded")
        } catch (e: Exception) {
            Log.e("YOLO", "Model load failed", e)
        }
    }

    private fun loadModelFile(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val input = preprocess(bitmap)
        val output = Array(1) { Array(56) { FloatArray(8400) } }

        interpreter.run(input, output)

        val detections = nonMaximumSupression(processOutput(output))

        runOnUiThread {
            overlayView.setResults(detections, bitmap.width, bitmap.height)
        }

        imageProxy.close()
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = bitmap.scale(INPUT_SIZE, INPUT_SIZE)

        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)

                buffer.putFloat(((pixel shr 16 and 0xFF) / 255f))
                buffer.putFloat(((pixel shr 8 and 0xFF) / 255f))
                buffer.putFloat((pixel and 0xFF) / 255f)
            }
        }

        return buffer
    }

    data class Detection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val classId: Int
    )

    private fun processOutput(output: Array<Array<FloatArray>>): List<Detection> {
        val detections = mutableListOf<Detection>()

        val numAnchors = 8400
        val numClasses = 52

        var detectionCount = 0 // For debugging

        for (i in 0 until numAnchors) {
            var maxClassScore = 0f
            var classId = -1

            // Find best class
            for (c in 0 until numClasses) {
                val score = output[0][c + 4][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    classId = c
                }
            }

            if (maxClassScore > CONF_THRESHOLD) {
                detectionCount++

                val cx = output[0][0][i]
                val cy = output[0][1][i]
                val w = output[0][2][i]
                val h = output[0][3][i]
                Log.d("Overlay", "cx: $cx, cy: $cy, w: $w, h: $h, screen_width: $SCREEN_WIDTH, screen_height: $SCREEN_HEIGHT")

                val x1 = cx - (w / 2f)
                val y1 = cy - (h / 2f)
                val x2 = cx + (w / 2f)
                val y2 = cy + (h / 2f)

                detections.add(Detection(x1, y1, x2, y2, maxClassScore, classId))
            }
        }

        if (detectionCount > 0) {
            Log.d("YOLO", "Detected $detectionCount objects before NMS")
        }

        return detections
    }

    private fun nonMaximumSupression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val result = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)

            sorted.removeAll { intersectionOverUnion(best, it) > IOU_THRESHOLD }
        }

        return result
    }

    private fun intersectionOverUnion(a: Detection, b: Detection): Float {
        // this calculated now much of the area of the box overlap with the "best"
        // guess so far, if it is above iou threshold we assume it is the same
        // guess and discard it

        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)

        val interArea = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)

        return interArea / (areaA + areaB - interArea)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = viewFinder.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        processImage(image)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (allPermissionsGranted()) {
                setupModel()
                startCamera()
            }
        }

    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}