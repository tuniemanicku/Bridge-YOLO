package com.example.bridge_yolo_app

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class LabelDictionary() {
    val labels = mapOf(
        0 to "2C",
        1 to "2D",
        2 to "2H",
        3 to "2S",
        4 to "3C",
        5 to "3D",
        6 to "3H",
        7 to "3S",
        8 to "4C",
        9 to "4D",
        10 to "4H",
        11 to "4S",
        12 to "5C",
        13 to "5D",
        14 to "5H",
        15 to "5S",
        16 to "6C",
        17 to "6D",
        18 to "6H",
        19 to "6S",
        20 to "7C",
        21 to "7D",
        22 to "7H",
        23 to "7S",
        24 to "8C",
        25 to "8D",
        26 to "8H",
        27 to "8S",
        28 to "9C",
        29 to "9D",
        30 to "9H",
        31 to "9S",
        32 to "AC",
        33 to "AD",
        34 to "AH",
        35 to "AS",
        36 to "JC",
        37 to "JD",
        38 to "JH",
        39 to "JS",
        40 to "KC",
        41 to "KD",
        42 to "KH",
        43 to "KS",
        44 to "QC",
        45 to "QD",
        46 to "QH",
        47 to "QS",
        48 to "10C",
        49 to "10D",
        50 to "10H",
        51 to "10S"
        )
}

class CameraOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var results: List<MainActivity.Detection> = listOf()
    private var imgWidth = 1
    private var imgHeight = 1
    private val labelDictionary = LabelDictionary()


    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
    }

    fun setResults(results: List<MainActivity.Detection>, width: Int, height: Int) {
        this.results = results
        this.imgWidth = width
        this.imgHeight = height
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (results.isEmpty()) return

        // Camera is nominally rotated 90deg so we need to unflip it
        val actualImgWidth = imgHeight.toFloat() // 480
        val actualImgHeight = imgWidth.toFloat() // 640

        val scaleX = width.toFloat() / actualImgWidth
        val scaleY = height.toFloat() / actualImgHeight

        // scaleY is bigger so we use it (i think)
        val scale = Math.max(scaleX, scaleY)

        // now one of the axis will be greater then screen dimension so we crop
        val offsetX = (actualImgWidth * scale - width) / 2f
        val offsetY = (actualImgHeight * scale - height) / 2f

        for (det in results) {
            // mapping of the new scale to the normalized values
            val frameX1 = (1 - det.y1) * actualImgWidth
            val frameY1 = det.x1 * actualImgHeight
            val frameX2 = (1 - det.y2) * actualImgWidth
            val frameY2 = det.x2 * actualImgHeight

            val left = frameX1 * scale - offsetX
            val top = frameY1 * scale - offsetY
            val right = frameX2 * scale - offsetX
            val bottom = frameY2 * scale - offsetY

            val rect = RectF(left, top, right, bottom)
            canvas.drawRect(rect, boxPaint)

            val label = "ID:${labelDictionary.labels[det.classId]} ${"%.2f".format(det.score)}"
            canvas.drawText(label, left, top - 10f, textPaint)
        }
    }

}