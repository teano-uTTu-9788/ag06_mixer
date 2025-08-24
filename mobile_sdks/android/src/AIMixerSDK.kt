/**
 * AI Mixer SDK for Android
 * 
 * Kotlin/Java wrapper around the cross-platform AI Mixer Core,
 * optimized for Android audio processing and integration.
 * 
 * Features:
 * - Native Android API with coroutines support
 * - AudioTrack/AudioRecord integration
 * - Real-time audio processing
 * - Background processing support
 * - Android-specific optimizations
 */

package com.aimixer.sdk

import android.content.Context
import android.media.*
import android.media.audiofx.*
import android.os.Build
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

// MARK: - Public Types

enum class AIMixerError(val message: String) : Exception(message) {
    INVALID_PARAMETER("Invalid parameter"),
    NOT_INITIALIZED("Mixer not initialized"),
    PROCESSING_FAILED("Audio processing failed"),
    MEMORY_ALLOCATION("Memory allocation failed"),
    MODEL_LOAD_FAILED("AI model load failed"),
    AUDIO_SESSION_ERROR("Audio session configuration failed"),
    UNSUPPORTED_FORMAT("Unsupported audio format"),
    PERMISSION_DENIED("Audio permission denied")
}

enum class Genre(val id: Int, val displayName: String) {
    SPEECH(0, "Speech"),
    ROCK(1, "Rock"),
    JAZZ(2, "Jazz"),
    ELECTRONIC(3, "Electronic"),
    CLASSICAL(4, "Classical"),
    UNKNOWN(5, "Unknown");
    
    companion object {
        fun fromId(id: Int): Genre = values().find { it.id == id } ?: UNKNOWN
    }
}

data class DSPConfiguration(
    // Noise Gate
    var gateThresholdDB: Float = -50.0f,
    var gateRatio: Float = 4.0f,
    var gateAttackMS: Float = 1.0f,
    var gateReleaseMS: Float = 100.0f,
    
    // Compressor
    var compThresholdDB: Float = -18.0f,
    var compRatio: Float = 3.0f,
    var compAttackMS: Float = 5.0f,
    var compReleaseMS: Float = 50.0f,
    var compKneeDB: Float = 2.0f,
    
    // Parametric EQ
    var eqLowGainDB: Float = 0.0f,
    var eqLowFreq: Float = 100.0f,
    var eqMidGainDB: Float = 0.0f,
    var eqMidFreq: Float = 1000.0f,
    var eqHighGainDB: Float = 0.0f,
    var eqHighFreq: Float = 8000.0f,
    
    // Limiter
    var limiterThresholdDB: Float = -3.0f,
    var limiterReleaseMS: Float = 10.0f,
    var limiterLookaheadMS: Float = 5.0f
)

data class ProcessingMetadata(
    val detectedGenre: Genre,
    val confidence: Float,
    val processingTimeMS: Float,
    val cpuUsagePercent: Float,
    val frameCount: Int,
    
    // Audio analysis
    val rmsLevelDB: Float,
    val peakLevelDB: Float,
    val spectralCentroid: Float,
    val zeroCrossingRate: Float,
    
    // DSP status
    val gateActive: Boolean,
    val compGainReductionDB: Float,
    val limiterActive: Boolean
)

data class PerformanceMetrics(
    val avgProcessingTimeMS: Float,
    val peakProcessingTimeMS: Float,
    val cpuUsagePercent: Float
)

// MARK: - Callback Interfaces

interface AIMixerCallback {
    fun onGenreDetected(genre: Genre, confidence: Float)
    fun onError(error: AIMixerError)
    fun onMetricsUpdated(metrics: PerformanceMetrics)
}

// MARK: - Main SDK Class

class AIMixerSDK private constructor(
    private val context: Context,
    private val config: DSPConfiguration?
) {
    companion object {
        // Audio format constants
        const val SAMPLE_RATE = 48000
        const val FRAME_SIZE = 960 // 20ms at 48kHz
        const val CHANNEL_COUNT = 2
        const val BYTES_PER_SAMPLE = 4 // Float32
        const val FEATURE_SIZE = 13 // MFCC features
        
        private var instance: AIMixerSDK? = null
        
        /**
         * Create singleton instance of AI Mixer SDK
         */
        @JvmStatic
        fun getInstance(context: Context, config: DSPConfiguration? = null): AIMixerSDK {
            return instance ?: synchronized(this) {
                instance ?: AIMixerSDK(context.applicationContext, config).also { instance = it }
            }
        }
        
        // Load native library
        init {
            System.loadLibrary("aimixer_core")
        }
    }
    
    // MARK: - Properties
    
    var callback: AIMixerCallback? = null
    
    private var mixerContext: Long = 0 // Native pointer
    private var audioManager: AudioManager? = null
    private var audioTrack: AudioTrack? = null
    private var audioRecord: AudioRecord? = null
    
    private var isInitialized = false
    private var isProcessing = false
    
    private val processingScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val callbackScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // Audio buffers
    private val inputBuffer = FloatBuffer.allocate(FRAME_SIZE * CHANNEL_COUNT)
    private val outputBuffer = FloatBuffer.allocate(FRAME_SIZE * CHANNEL_COUNT)
    
    // MARK: - Public API
    
    /**
     * Initialize the AI mixer with optional DSP configuration
     */
    suspend fun initialize(): Unit = suspendCoroutine { continuation ->
        processingScope.launch {
            try {
                initializeInternal()
                continuation.resume(Unit)
            } catch (e: Exception) {
                continuation.resumeWithException(e)
            }
        }
    }
    
    /**
     * Start real-time audio processing
     */
    suspend fun startProcessing(): Unit = suspendCoroutine { continuation ->
        if (!isInitialized) {
            continuation.resumeWithException(AIMixerError.NOT_INITIALIZED)
            return@suspendCoroutine
        }
        
        processingScope.launch {
            try {
                startAudioProcessing()
                isProcessing = true
                continuation.resume(Unit)
            } catch (e: Exception) {
                continuation.resumeWithException(e)
            }
        }
    }
    
    /**
     * Stop real-time audio processing
     */
    suspend fun stopProcessing(): Unit = suspendCoroutine { continuation ->
        processingScope.launch {
            stopAudioProcessing()
            isProcessing = false
            continuation.resume(Unit)
        }
    }
    
    /**
     * Process a single audio buffer
     */
    suspend fun processBuffer(inputData: FloatArray): Pair<FloatArray, ProcessingMetadata> = 
        suspendCoroutine { cont ->
            if (!isInitialized) {
                cont.resumeWithException(AIMixerError.NOT_INITIALIZED)
                return@suspendCoroutine
            }
            
            if (inputData.size != FRAME_SIZE) {
                cont.resumeWithException(AIMixerError.INVALID_PARAMETER)
                return@suspendCoroutine
            }
            
            processingScope.launch {
                try {
                    val result = processAudioFrame(inputData)
                    cont.resume(result)
                } catch (e: Exception) {
                    cont.resumeWithException(e)
                }
            }
        }
    
    /**
     * Update DSP configuration at runtime
     */
    suspend fun updateConfiguration(config: DSPConfiguration): Unit = suspendCoroutine { cont ->
        if (!isInitialized) {
            cont.resumeWithException(AIMixerError.NOT_INITIALIZED)
            return@suspendCoroutine
        }
        
        processingScope.launch {
            try {
                val result = nativeUpdateConfig(mixerContext, config)
                if (result == 0) {
                    cont.resume(Unit)
                } else {
                    cont.resumeWithException(convertNativeError(result))
                }
            } catch (e: Exception) {
                cont.resumeWithException(e)
            }
        }
    }
    
    /**
     * Load custom AI model for genre detection
     */
    suspend fun loadCustomModel(modelData: ByteArray): Unit = suspendCoroutine { cont ->
        if (!isInitialized) {
            cont.resumeWithException(AIMixerError.NOT_INITIALIZED)
            return@suspendCoroutine
        }
        
        processingScope.launch {
            try {
                val result = nativeLoadCustomModel(mixerContext, modelData)
                if (result == 0) {
                    cont.resume(Unit)
                } else {
                    cont.resumeWithException(convertNativeError(result))
                }
            } catch (e: Exception) {
                cont.resumeWithException(e)
            }
        }
    }
    
    /**
     * Set manual genre override
     */
    suspend fun setManualGenre(genre: Genre, bypass: Boolean = true): Unit = suspendCoroutine { cont ->
        if (!isInitialized) {
            cont.resumeWithException(AIMixerError.NOT_INITIALIZED)
            return@suspendCoroutine
        }
        
        processingScope.launch {
            try {
                val result = nativeSetManualGenre(mixerContext, genre.id, bypass)
                if (result == 0) {
                    cont.resume(Unit)
                } else {
                    cont.resumeWithException(convertNativeError(result))
                }
            } catch (e: Exception) {
                cont.resumeWithException(e)
            }
        }
    }
    
    /**
     * Get current performance metrics
     */
    suspend fun getPerformanceMetrics(): PerformanceMetrics = suspendCoroutine { cont ->
        if (!isInitialized) {
            cont.resumeWithException(AIMixerError.NOT_INITIALIZED)
            return@suspendCoroutine
        }
        
        processingScope.launch {
            try {
                val metrics = nativeGetPerformanceMetrics(mixerContext)
                cont.resume(metrics)
            } catch (e: Exception) {
                cont.resumeWithException(e)
            }
        }
    }
    
    /**
     * Shutdown and cleanup resources
     */
    fun shutdown() {
        processingScope.launch {
            stopAudioProcessing()
            
            if (isInitialized && mixerContext != 0L) {
                nativeDestroy(mixerContext)
                mixerContext = 0
                isInitialized = false
            }
        }
        
        processingScope.cancel()
        callbackScope.cancel()
    }
    
    // MARK: - Private Implementation
    
    private fun initializeInternal() {
        audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        
        // Check audio permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // Would check permissions here in production
        }
        
        // Initialize native mixer
        mixerContext = nativeCreate(config)
        if (mixerContext == 0L) {
            throw AIMixerError.MEMORY_ALLOCATION
        }
        
        // Set up genre detection callback
        nativeSetGenreCallback(mixerContext) { genreId, confidence ->
            val genre = Genre.fromId(genreId)
            callbackScope.launch {
                callback?.onGenreDetected(genre, confidence)
            }
        }
        
        isInitialized = true
    }
    
    private fun startAudioProcessing() {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        
        if (minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            throw AIMixerError.UNSUPPORTED_FORMAT
        }
        
        // Create AudioRecord for input
        audioRecord = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBufferSize * 2)
            .build()
        
        // Create AudioTrack for output
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBufferSize * 2)
            .build()
        
        audioRecord?.startRecording()
        audioTrack?.play()
        
        // Start processing loop
        processingScope.launch {
            audioProcessingLoop()
        }
    }
    
    private fun stopAudioProcessing() {
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
    }
    
    private suspend fun audioProcessingLoop() {
        val inputArray = FloatArray(FRAME_SIZE)
        val outputArray = FloatArray(FRAME_SIZE)
        
        while (isProcessing && audioRecord != null) {
            try {
                // Read audio input
                val readResult = audioRecord?.read(inputArray, 0, FRAME_SIZE, AudioRecord.READ_BLOCKING)
                
                if (readResult != FRAME_SIZE) {
                    delay(1) // Brief pause before retrying
                    continue
                }
                
                // Process audio frame
                val (processedAudio, metadata) = processAudioFrame(inputArray)
                
                // Write processed audio output
                audioTrack?.write(processedAudio, 0, processedAudio.size, AudioTrack.WRITE_BLOCKING)
                
                // Update metrics periodically
                if (metadata.frameCount % 100 == 0) {
                    val metrics = getPerformanceMetrics()
                    callbackScope.launch {
                        callback?.onMetricsUpdated(metrics)
                    }
                }
                
            } catch (e: Exception) {
                callbackScope.launch {
                    callback?.onError(AIMixerError.PROCESSING_FAILED)
                }
                delay(10) // Brief pause before retrying
            }
        }
    }
    
    private fun processAudioFrame(inputData: FloatArray): Pair<FloatArray, ProcessingMetadata> {
        val outputData = FloatArray(FRAME_SIZE)
        val metadata = nativeProcessFrame(mixerContext, inputData, outputData)
        return Pair(outputData, metadata)
    }
    
    private fun convertNativeError(errorCode: Int): AIMixerError {
        return when (errorCode) {
            -1 -> AIMixerError.INVALID_PARAMETER
            -2 -> AIMixerError.NOT_INITIALIZED
            -3 -> AIMixerError.PROCESSING_FAILED
            -4 -> AIMixerError.MEMORY_ALLOCATION
            -5 -> AIMixerError.MODEL_LOAD_FAILED
            else -> AIMixerError.PROCESSING_FAILED
        }
    }
    
    // MARK: - Native Interface
    
    private external fun nativeCreate(config: DSPConfiguration?): Long
    private external fun nativeDestroy(context: Long)
    private external fun nativeUpdateConfig(context: Long, config: DSPConfiguration): Int
    private external fun nativeProcessFrame(context: Long, input: FloatArray, output: FloatArray): ProcessingMetadata
    private external fun nativeGetPerformanceMetrics(context: Long): PerformanceMetrics
    private external fun nativeLoadCustomModel(context: Long, modelData: ByteArray): Int
    private external fun nativeSetManualGenre(context: Long, genreId: Int, bypass: Boolean): Int
    private external fun nativeSetGenreCallback(context: Long, callback: (Int, Float) -> Unit)
}

// MARK: - Builder Pattern for Easy Configuration

class AIMixerSDKBuilder(private val context: Context) {
    private var config: DSPConfiguration? = null
    private var callback: AIMixerCallback? = null
    
    fun withConfiguration(config: DSPConfiguration): AIMixerSDKBuilder {
        this.config = config
        return this
    }
    
    fun withCallback(callback: AIMixerCallback): AIMixerSDKBuilder {
        this.callback = callback
        return this
    }
    
    suspend fun build(): AIMixerSDK {
        val sdk = AIMixerSDK.getInstance(context, config)
        sdk.callback = callback
        sdk.initialize()
        return sdk
    }
}

// MARK: - Convenience Extensions

fun Context.createAIMixer(): AIMixerSDKBuilder {
    return AIMixerSDKBuilder(this)
}

// MARK: - Java Interoperability

class AIMixerSDKJava private constructor(
    private val kotlinSDK: AIMixerSDK
) {
    companion object {
        @JvmStatic
        fun getInstance(context: Context): AIMixerSDKJava {
            val kotlinSDK = AIMixerSDK.getInstance(context)
            return AIMixerSDKJava(kotlinSDK)
        }
    }
    
    interface JavaCallback {
        fun onGenreDetected(genre: Genre, confidence: Float)
        fun onError(error: AIMixerError)
        fun onMetricsUpdated(metrics: PerformanceMetrics)
    }
    
    fun setCallback(callback: JavaCallback) {
        kotlinSDK.callback = object : AIMixerCallback {
            override fun onGenreDetected(genre: Genre, confidence: Float) {
                callback.onGenreDetected(genre, confidence)
            }
            
            override fun onError(error: AIMixerError) {
                callback.onError(error)
            }
            
            override fun onMetricsUpdated(metrics: PerformanceMetrics) {
                callback.onMetricsUpdated(metrics)
            }
        }
    }
    
    // Blocking Java-friendly methods using runBlocking
    fun initialize() {
        runBlocking { kotlinSDK.initialize() }
    }
    
    fun startProcessing() {
        runBlocking { kotlinSDK.startProcessing() }
    }
    
    fun stopProcessing() {
        runBlocking { kotlinSDK.stopProcessing() }
    }
    
    fun updateConfiguration(config: DSPConfiguration) {
        runBlocking { kotlinSDK.updateConfiguration(config) }
    }
    
    fun loadCustomModel(modelData: ByteArray) {
        runBlocking { kotlinSDK.loadCustomModel(modelData) }
    }
    
    fun setManualGenre(genre: Genre, bypass: Boolean) {
        runBlocking { kotlinSDK.setManualGenre(genre, bypass) }
    }
    
    fun getPerformanceMetrics(): PerformanceMetrics {
        return runBlocking { kotlinSDK.getPerformanceMetrics() }
    }
    
    fun shutdown() {
        kotlinSDK.shutdown()
    }
}