/**
 * AI Mixer SDK for iOS
 * 
 * Swift wrapper around the cross-platform AI Mixer Core,
 * optimized for iOS audio processing and integration.
 * 
 * Features:
 * - Native Swift API with async/await support
 * - AVAudioEngine integration
 * - Real-time audio processing
 * - Background processing support
 * - iOS-specific optimizations
 */

import Foundation
import AVFoundation
import Accelerate

// MARK: - Public Types

public enum AIMixerError: Error, CustomStringConvertible {
    case invalidParameter
    case notInitialized
    case processingFailed
    case memoryAllocation
    case modelLoadFailed
    case audioSessionError
    case unsupportedFormat
    
    public var description: String {
        switch self {
        case .invalidParameter: return "Invalid parameter"
        case .notInitialized: return "Mixer not initialized"
        case .processingFailed: return "Audio processing failed"
        case .memoryAllocation: return "Memory allocation failed"
        case .modelLoadFailed: return "AI model load failed"
        case .audioSessionError: return "Audio session configuration failed"
        case .unsupportedFormat: return "Unsupported audio format"
        }
    }
}

public enum Genre: Int, CaseIterable, CustomStringConvertible {
    case speech = 0
    case rock = 1
    case jazz = 2
    case electronic = 3
    case classical = 4
    case unknown = 5
    
    public var description: String {
        switch self {
        case .speech: return "Speech"
        case .rock: return "Rock"
        case .jazz: return "Jazz"
        case .electronic: return "Electronic"
        case .classical: return "Classical"
        case .unknown: return "Unknown"
        }
    }
}

public struct DSPConfiguration {
    // Noise Gate
    public var gateThresholdDB: Float = -50.0
    public var gateRatio: Float = 4.0
    public var gateAttackMS: Float = 1.0
    public var gateReleaseMS: Float = 100.0
    
    // Compressor
    public var compThresholdDB: Float = -18.0
    public var compRatio: Float = 3.0
    public var compAttackMS: Float = 5.0
    public var compReleaseMS: Float = 50.0
    public var compKneeDB: Float = 2.0
    
    // Parametric EQ
    public var eqLowGainDB: Float = 0.0
    public var eqLowFreq: Float = 100.0
    public var eqMidGainDB: Float = 0.0
    public var eqMidFreq: Float = 1000.0
    public var eqHighGainDB: Float = 0.0
    public var eqHighFreq: Float = 8000.0
    
    // Limiter
    public var limiterThresholdDB: Float = -3.0
    public var limiterReleaseMS: Float = 10.0
    public var limiterLookaheadMS: Float = 5.0
    
    public init() {}
}

public struct ProcessingMetadata {
    public let detectedGenre: Genre
    public let confidence: Float
    public let processingTimeMS: Float
    public let cpuUsagePercent: Float
    public let frameCount: UInt32
    
    // Audio analysis
    public let rmsLevelDB: Float
    public let peakLevelDB: Float
    public let spectralCentroid: Float
    public let zeroCrossingRate: Float
    
    // DSP status
    public let gateActive: Bool
    public let compGainReductionDB: Float
    public let limiterActive: Bool
}

public struct PerformanceMetrics {
    public let avgProcessingTimeMS: Float
    public let peakProcessingTimeMS: Float
    public let cpuUsagePercent: Float
}

// MARK: - Delegate Protocol

@objc public protocol AIMixerDelegate: AnyObject {
    func mixerDidDetectGenre(_ genre: Genre, confidence: Float)
    func mixerDidEncounterError(_ error: AIMixerError)
    @objc optional func mixerDidUpdateMetrics(_ metrics: PerformanceMetrics)
}

// MARK: - Main SDK Class

@objc public class AIMixerSDK: NSObject {
    
    // MARK: - Properties
    
    public weak var delegate: AIMixerDelegate?
    
    private var mixerContext: OpaquePointer?
    private var audioEngine: AVAudioEngine?
    private var audioUnit: AVAudioUnit?
    private var isProcessing = false
    
    private let processingQueue = DispatchQueue(label: "com.aimixer.processing", qos: .userInteractive)
    private let callbackQueue = DispatchQueue(label: "com.aimixer.callbacks", qos: .userInitiated)
    
    // Audio format constants
    private let sampleRate: Double = 48000.0
    private let frameSize: Int = 960 // 20ms at 48kHz
    private let channelCount: Int = 2
    private let featureSize: Int = 13 // MFCC features
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupAudioSession()
    }
    
    deinit {
        shutdown()
    }
    
    // MARK: - Public API
    
    /**
     * Initialize the AI mixer with optional DSP configuration
     * Uses async/await for modern Swift concurrency
     */
    public func initialize(config: DSPConfiguration? = nil) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                do {
                    try self.initializeInternal(config: config)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /**
     * Start real-time audio processing using async/await
     */
    public func startProcessing() async throws {
        guard mixerContext != nil else {
            throw AIMixerError.notInitialized
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                do {
                    try self.startAudioEngine()
                    self.isProcessing = true
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /**
     * Stop real-time audio processing
     */
    public func stopProcessing() async {
        return await withCheckedContinuation { continuation in
            processingQueue.async {
                self.stopAudioEngine()
                self.isProcessing = false
                continuation.resume()
            }
        }
    }
    
    /**
     * Process a single audio buffer
     */
    public func processBuffer(_ buffer: AVAudioPCMBuffer) async throws -> (AVAudioPCMBuffer, ProcessingMetadata) {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        guard let floatChannelData = buffer.floatChannelData,
              buffer.frameLength == frameSize else {
            throw AIMixerError.invalidParameter
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                do {
                    let result = try self.processAudioData(floatChannelData[0], frameCount: Int(buffer.frameLength))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /**
     * Update DSP configuration at runtime
     */
    public func updateConfiguration(_ config: DSPConfiguration) async throws {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                let result = ai_mixer_update_config(mixerContext, self.convertConfig(config))
                if result == AI_MIXER_SUCCESS {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: self.convertError(result))
                }
            }
        }
    }
    
    /**
     * Load custom AI model for genre detection
     */
    public func loadCustomModel(from url: URL) async throws {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                do {
                    let modelData = try Data(contentsOf: url)
                    let result = modelData.withUnsafeBytes { bytes in
                        return ai_mixer_load_custom_model(
                            mixerContext,
                            bytes.bindMemory(to: UInt8.self).baseAddress,
                            UInt32(modelData.count)
                        )
                    }
                    
                    if result == AI_MIXER_SUCCESS {
                        continuation.resume()
                    } else {
                        continuation.resume(throwing: self.convertError(result))
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /**
     * Set manual genre override
     */
    public func setManualGenre(_ genre: Genre, bypass: Bool = true) async throws {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                let result = ai_mixer_set_manual_genre(mixerContext, ai_mixer_genre_t(genre.rawValue), bypass)
                if result == AI_MIXER_SUCCESS {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: self.convertError(result))
                }
            }
        }
    }
    
    /**
     * Get current performance metrics
     */
    public func getPerformanceMetrics() async throws -> PerformanceMetrics {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            processingQueue.async {
                var avgTime: Float = 0
                var peakTime: Float = 0
                var cpuUsage: Float = 0
                
                let result = ai_mixer_get_performance_metrics(mixerContext, &avgTime, &peakTime, &cpuUsage)
                
                if result == AI_MIXER_SUCCESS {
                    let metrics = PerformanceMetrics(
                        avgProcessingTimeMS: avgTime,
                        peakProcessingTimeMS: peakTime,
                        cpuUsagePercent: cpuUsage
                    )
                    continuation.resume(returning: metrics)
                } else {
                    continuation.resume(throwing: self.convertError(result))
                }
            }
        }
    }
    
    /**
     * Shutdown and cleanup resources
     */
    public func shutdown() {
        processingQueue.sync {
            stopAudioEngine()
            
            if let context = mixerContext {
                ai_mixer_destroy(context)
                mixerContext = nil
            }
        }
    }
    
    // MARK: - Private Implementation
    
    private func initializeInternal(config: DSPConfiguration?) throws {
        var cConfig: UnsafeMutablePointer<ai_mixer_dsp_config_t>?
        
        if let config = config {
            cConfig = UnsafeMutablePointer<ai_mixer_dsp_config_t>.allocate(capacity: 1)
            cConfig?.pointee = convertConfig(config).pointee
        }
        
        mixerContext = ai_mixer_create(cConfig)
        
        if let cConfig = cConfig {
            cConfig.deallocate()
        }
        
        guard mixerContext != nil else {
            throw AIMixerError.memoryAllocation
        }
        
        // Set up genre detection callback
        let callbackContext = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        let result = ai_mixer_set_genre_callback(mixerContext!, { genre, confidence, userData in
            guard let userData = userData else { return }
            let mixer = Unmanaged<AIMixerSDK>.fromOpaque(userData).takeUnretainedValue()
            mixer.handleGenreCallback(Genre(rawValue: Int(genre.rawValue)) ?? .unknown, confidence: confidence)
        }, callbackContext)
        
        if result != AI_MIXER_SUCCESS {
            throw convertError(result)
        }
    }
    
    private func setupAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setPreferredSampleRate(sampleRate)
            try session.setPreferredIOBufferDuration(Double(frameSize) / sampleRate)
            try session.setActive(true)
        } catch {
            DispatchQueue.main.async {
                self.delegate?.mixerDidEncounterError(.audioSessionError)
            }
        }
    }
    
    private func startAudioEngine() throws {
        audioEngine = AVAudioEngine()
        
        let inputNode = audioEngine!.inputNode
        let outputNode = audioEngine!.outputNode
        
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: AVAudioChannelCount(channelCount))!
        
        // Install tap for processing
        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(frameSize), format: format) { [weak self] buffer, time in
            self?.processAudioBuffer(buffer, at: time)
        }
        
        // Connect nodes
        audioEngine!.connect(inputNode, to: outputNode, format: format)
        
        try audioEngine!.start()
    }
    
    private func stopAudioEngine() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, at time: AVAudioTime) {
        guard let mixerContext = mixerContext,
              let floatChannelData = buffer.floatChannelData else { return }
        
        processingQueue.async { [weak self] in
            do {
                let _ = try self?.processAudioData(floatChannelData[0], frameCount: Int(buffer.frameLength))
            } catch {
                DispatchQueue.main.async {
                    self?.delegate?.mixerDidEncounterError(error as? AIMixerError ?? .processingFailed)
                }
            }
        }
    }
    
    private func processAudioData(_ inputData: UnsafeMutablePointer<Float>, frameCount: Int) throws -> (AVAudioPCMBuffer, ProcessingMetadata) {
        guard let mixerContext = mixerContext else {
            throw AIMixerError.notInitialized
        }
        
        // Allocate output buffer
        let outputData = UnsafeMutablePointer<Float>.allocate(capacity: frameCount)
        defer { outputData.deallocate() }
        
        var metadata = ai_mixer_metadata_t()
        
        let result = ai_mixer_process_frame(
            mixerContext,
            inputData,
            outputData,
            UInt32(frameCount),
            &metadata
        )
        
        guard result == AI_MIXER_SUCCESS else {
            throw convertError(result)
        }
        
        // Create output buffer
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount))!
        outputBuffer.frameLength = AVAudioFrameCount(frameCount)
        
        // Copy processed data
        let outputChannelData = outputBuffer.floatChannelData![0]
        memcpy(outputChannelData, outputData, frameCount * MemoryLayout<Float>.size)
        
        // Convert metadata
        let swiftMetadata = ProcessingMetadata(
            detectedGenre: Genre(rawValue: Int(metadata.detected_genre.rawValue)) ?? .unknown,
            confidence: metadata.confidence,
            processingTimeMS: metadata.processing_time_ms,
            cpuUsagePercent: metadata.cpu_usage_percent,
            frameCount: metadata.frame_count,
            rmsLevelDB: metadata.rms_level_db,
            peakLevelDB: metadata.peak_level_db,
            spectralCentroid: metadata.spectral_centroid,
            zeroCrossingRate: metadata.zero_crossing_rate,
            gateActive: metadata.gate_active,
            compGainReductionDB: metadata.comp_gain_reduction_db,
            limiterActive: metadata.limiter_active
        )
        
        return (outputBuffer, swiftMetadata)
    }
    
    private func handleGenreCallback(_ genre: Genre, confidence: Float) {
        callbackQueue.async { [weak self] in
            DispatchQueue.main.async {
                self?.delegate?.mixerDidDetectGenre(genre, confidence: confidence)
            }
        }
    }
    
    private func convertConfig(_ config: DSPConfiguration) -> UnsafeMutablePointer<ai_mixer_dsp_config_t> {
        let cConfig = UnsafeMutablePointer<ai_mixer_dsp_config_t>.allocate(capacity: 1)
        
        cConfig.pointee.gate_threshold_db = config.gateThresholdDB
        cConfig.pointee.gate_ratio = config.gateRatio
        cConfig.pointee.gate_attack_ms = config.gateAttackMS
        cConfig.pointee.gate_release_ms = config.gateReleaseMS
        
        cConfig.pointee.comp_threshold_db = config.compThresholdDB
        cConfig.pointee.comp_ratio = config.compRatio
        cConfig.pointee.comp_attack_ms = config.compAttackMS
        cConfig.pointee.comp_release_ms = config.compReleaseMS
        cConfig.pointee.comp_knee_db = config.compKneeDB
        
        cConfig.pointee.eq_low_gain_db = config.eqLowGainDB
        cConfig.pointee.eq_low_freq = config.eqLowFreq
        cConfig.pointee.eq_mid_gain_db = config.eqMidGainDB
        cConfig.pointee.eq_mid_freq = config.eqMidFreq
        cConfig.pointee.eq_high_gain_db = config.eqHighGainDB
        cConfig.pointee.eq_high_freq = config.eqHighFreq
        
        cConfig.pointee.limiter_threshold_db = config.limiterThresholdDB
        cConfig.pointee.limiter_release_ms = config.limiterReleaseMS
        cConfig.pointee.limiter_lookahead_ms = config.limiterLookaheadMS
        
        return cConfig
    }
    
    private func convertError(_ result: ai_mixer_result_t) -> AIMixerError {
        switch result {
        case AI_MIXER_ERROR_INVALID_PARAM:
            return .invalidParameter
        case AI_MIXER_ERROR_NOT_INITIALIZED:
            return .notInitialized
        case AI_MIXER_ERROR_PROCESSING_FAILED:
            return .processingFailed
        case AI_MIXER_ERROR_MEMORY_ALLOCATION:
            return .memoryAllocation
        case AI_MIXER_ERROR_MODEL_LOAD_FAILED:
            return .modelLoadFailed
        default:
            return .processingFailed
        }
    }
}

// MARK: - C Bridge

// Import C functions
@_silgen_name("ai_mixer_create")
private func ai_mixer_create(_: UnsafePointer<ai_mixer_dsp_config_t>?) -> OpaquePointer?

@_silgen_name("ai_mixer_destroy")
private func ai_mixer_destroy(_: OpaquePointer?)

@_silgen_name("ai_mixer_update_config")
private func ai_mixer_update_config(_: OpaquePointer?, _: UnsafePointer<ai_mixer_dsp_config_t>?) -> ai_mixer_result_t

@_silgen_name("ai_mixer_process_frame")
private func ai_mixer_process_frame(_: OpaquePointer?, _: UnsafePointer<Float>?, _: UnsafeMutablePointer<Float>?, _: UInt32, _: UnsafeMutablePointer<ai_mixer_metadata_t>?) -> ai_mixer_result_t

@_silgen_name("ai_mixer_get_performance_metrics")
private func ai_mixer_get_performance_metrics(_: OpaquePointer?, _: UnsafeMutablePointer<Float>?, _: UnsafeMutablePointer<Float>?, _: UnsafeMutablePointer<Float>?) -> ai_mixer_result_t

@_silgen_name("ai_mixer_load_custom_model")
private func ai_mixer_load_custom_model(_: OpaquePointer?, _: UnsafePointer<UInt8>?, _: UInt32) -> ai_mixer_result_t

@_silgen_name("ai_mixer_set_genre_callback")
private func ai_mixer_set_genre_callback(_: OpaquePointer?, _: @convention(c) (ai_mixer_genre_t, Float, UnsafeMutableRawPointer?) -> Void, _: UnsafeMutableRawPointer?) -> ai_mixer_result_t

@_silgen_name("ai_mixer_set_manual_genre")
private func ai_mixer_set_manual_genre(_: OpaquePointer?, _: ai_mixer_genre_t, _: Bool) -> ai_mixer_result_t

// C types bridge
private typealias ai_mixer_result_t = Int32
private typealias ai_mixer_genre_t = UInt32

private let AI_MIXER_SUCCESS: ai_mixer_result_t = 0
private let AI_MIXER_ERROR_INVALID_PARAM: ai_mixer_result_t = -1
private let AI_MIXER_ERROR_NOT_INITIALIZED: ai_mixer_result_t = -2
private let AI_MIXER_ERROR_PROCESSING_FAILED: ai_mixer_result_t = -3
private let AI_MIXER_ERROR_MEMORY_ALLOCATION: ai_mixer_result_t = -4
private let AI_MIXER_ERROR_MODEL_LOAD_FAILED: ai_mixer_result_t = -5

private struct ai_mixer_dsp_config_t {
    var gate_threshold_db: Float
    var gate_ratio: Float
    var gate_attack_ms: Float
    var gate_release_ms: Float
    var comp_threshold_db: Float
    var comp_ratio: Float
    var comp_attack_ms: Float
    var comp_release_ms: Float
    var comp_knee_db: Float
    var eq_low_gain_db: Float
    var eq_low_freq: Float
    var eq_mid_gain_db: Float
    var eq_mid_freq: Float
    var eq_high_gain_db: Float
    var eq_high_freq: Float
    var limiter_threshold_db: Float
    var limiter_release_ms: Float
    var limiter_lookahead_ms: Float
}

private struct ai_mixer_metadata_t {
    var detected_genre: ai_mixer_genre_t
    var confidence: Float
    var processing_time_ms: Float
    var cpu_usage_percent: Float
    var frame_count: UInt32
    var rms_level_db: Float
    var peak_level_db: Float
    var spectral_centroid: Float
    var zero_crossing_rate: Float
    var gate_active: Bool
    var comp_gain_reduction_db: Float
    var limiter_active: Bool
}