package com.example.apptry

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.rememberAsyncImagePainter
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : ComponentActivity() {

    private var imageUri by mutableStateOf<Uri?>(null)
    private var imageFile by mutableStateOf<File?>(null)
    private var resultText by mutableStateOf("")
    private var capturedBitmap by mutableStateOf<Bitmap?>(null)

    private val chooserLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data

            val uri = data?.data
            if (uri != null) {
                imageUri = uri
                imageFile = File(uri.path ?: "")
                uploadImageToServer(imageFile!!)
            } else {
                val bitmap = data?.extras?.get("data") as? Bitmap
                if (bitmap != null) {
                    capturedBitmap = bitmap
                    val file = File.createTempFile("captured_", ".jpg", cacheDir)
                    val out = FileOutputStream(file)
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
                    out.flush()
                    out.close()

                    imageFile = file
                    imageUri = Uri.fromFile(file)
                    uploadImageToServer(file)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            var textInput by remember { mutableStateOf("") }

            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically)
                    ) {
                        // Image preview
                        Box(
                            modifier = Modifier
                                .size(200.dp)
                                .padding(8.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            when {
                                capturedBitmap != null -> {
                                    Image(
                                        bitmap = capturedBitmap!!.asImageBitmap(),
                                        contentDescription = null,
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                                imageUri != null -> {
                                    Image(
                                        painter = rememberAsyncImagePainter(imageUri),
                                        contentDescription = null,
                                        modifier = Modifier.fillMaxSize(),
                                        contentScale = ContentScale.Crop
                                    )
                                }
                                else -> {
                                    Text("No image selected", style = MaterialTheme.typography.bodyMedium)
                                }
                            }
                        }

                        // Launch chooser
                        Button(onClick = { openCameraOrGalleryChooser() }) {
                            Text("Pick Image")
                        }

                        // Text input
                        TextField(
                            value = textInput,
                            onValueChange = { textInput = it },
                            label = { Text("Enter item description") },
                            modifier = Modifier.fillMaxWidth()
                        )

                        // Text classify button
                        Button(onClick = {
                            if (textInput.isNotBlank()) classifyTextDescription(textInput)
                            else resultText = "Please enter a description."
                        }) {
                            Text("Classify Text")
                        }

                        // Result display
                        Text(text = resultText, style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }
        }
    }

    private fun openCameraOrGalleryChooser() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val galleryIntent = Intent(Intent.ACTION_PICK).apply {
            type = "image/*"
        }

        val chooser = Intent.createChooser(galleryIntent, "Select Image Source")
        chooser.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(cameraIntent))
        chooserLauncher.launch(chooser)
    }

    private fun uploadImageToServer(file: File) {
        resultText = "Classifying image..."

        val client = OkHttpClient()

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                file.name,
                file.asRequestBody("image/*".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url("http://192.168.1.4:8000/classify/")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    resultText = "Error: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body?.string()
                val json = JSONObject(responseBody ?: "{}")
                val category = json.optString("category", "Unknown")
                runOnUiThread {
                    resultText = "Image classified as: $category"
                }
            }
        })
    }

    private fun classifyTextDescription(description: String) {
        resultText = "Classifying text..."

        val client = OkHttpClient()
        val json = JSONObject().put("description", description).toString()
        val body = json.toRequestBody("application/json".toMediaTypeOrNull())

        val request = Request.Builder()
            .url("http://192.168.1.4:8000/classify-text/")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    resultText = "Error: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body?.string()
                val json = JSONObject(responseBody ?: "{}")
                val category = json.optString("category", "Unknown")
                runOnUiThread {
                    resultText = "Text classified as: $category"
                }
            }
        })
    }
}