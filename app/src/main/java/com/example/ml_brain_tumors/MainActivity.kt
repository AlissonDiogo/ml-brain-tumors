package com.example.ml_brain_tumors

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import coil3.compose.rememberAsyncImagePainter
import com.example.ml_brain_tumors.ui.theme.MlbraintumorsTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MlbraintumorsTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ImagePredictionScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}


@Composable
fun ImagePredictionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var predictionResult by remember { mutableStateOf<String?>(null) }
    var photoUri by remember { mutableStateOf<Uri?>(null) }
    val cameraPermission = android.Manifest.permission.CAMERA

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success && photoUri != null) {
            imageUri = photoUri
            handleImage(context, photoUri!!) { result -> predictionResult = result }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            // se for concedida, tira a foto
            val photoFile = File.createTempFile("photo_", ".jpg", context.cacheDir).apply {
                deleteOnExit()
            }
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", photoFile)
            photoUri = uri
            takePictureLauncher.launch(uri)
        } else {
            Toast.makeText(context, "Permissão da câmera negada", Toast.LENGTH_SHORT).show()
        }
    }

    // Launcher: selecionar da galeria
    val pickImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            imageUri = it
            handleImage(context, it) { result -> predictionResult = result }
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Botões
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Button(onClick = { pickImageLauncher.launch("image/*") }) {
                Icon(Icons.Default.Add, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Galeria")
            }

            Button(onClick = {
                permissionLauncher.launch(cameraPermission)

            }) {
                Icon(Icons.Outlined.Home, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Câmera")
            }
        }

        // Exibir imagem
        imageUri?.let { uri ->
            val inputStream = context.contentResolver.openInputStream(uri)
            inputStream?.let {
                val bitmap = BitmapFactory.decodeStream(it)
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp),
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(8.dp)
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Imagem",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }

        // Resultado
        predictionResult?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

fun handleImage(context: Context, uri: Uri, onResult: (String) -> Unit) {
    val inputStream = context.contentResolver.openInputStream(uri)
    val originalBitmap = BitmapFactory.decodeStream(inputStream)
    val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 150, 150, true)

    val input = convertBitmapToByteBuffer(resizedBitmap)
    val output = Array(1) { FloatArray(4) }
    val interpreter = loadTFLiteModel(context, "brain_tumor_cnn.tflite")
    interpreter?.run(input, output)

    val labels = listOf("Glioma", "Meningioma", "Sem tumor", "Pituitary")
    val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
    val confidence = output[0][maxIndex]
    val resultText = "🧠 ${labels[maxIndex]} - ${(confidence * 100).toInt()}%"
    onResult(resultText)
}

fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    val inputSize = 150
    val inputChannels = 3
    val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * inputChannels)
    byteBuffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(inputSize * inputSize)
    bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

    for (pixel in pixels) {
        val r = (pixel shr 16 and 0xFF) / 255.0f
        val g = (pixel shr 8 and 0xFF) / 255.0f
        val b = (pixel and 0xFF) / 255.0f

        byteBuffer.putFloat(r)
        byteBuffer.putFloat(g)
        byteBuffer.putFloat(b)
    }

    return byteBuffer
}

// Carrega modelo .tflite
fun loadTFLiteModel(context: android.content.Context, modelName: String): Interpreter? {
    return try {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        Interpreter(mappedByteBuffer)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}
