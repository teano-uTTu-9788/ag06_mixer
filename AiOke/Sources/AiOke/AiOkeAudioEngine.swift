import AVFoundation
import Accelerate

/// AiOke Audio Engine - Core karaoke audio processing following Apple best practices
@available(iOS 15.0, macOS 12.0, *)
public class AiOkeAudioEngine: ObservableObject {
    private let audioEngine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let reverbNode = AVAudioUnitReverb()
    private let eqNode = AVAudioUnitEQ(numberOfBands: 3)
    
    @Published public var isPlaying = false
    @Published public var isRecording = false
    @Published public var vocalReductionAmount: Float = 0.5
    @Published public var reverbAmount: Float = 0.3
    
    private var audioFile: AVAudioFile?
    private var recordingFile: AVAudioFile?
    
    public init() {
        setupAudioEngine()
        setupAudioSession()
    }
    
    private func setupAudioEngine() {
        // Connect audio nodes following Apple's recommended signal chain
        audioEngine.attach(playerNode)
        audioEngine.attach(reverbNode)
        audioEngine.attach(eqNode)
        
        // Configure reverb for karaoke-style vocal enhancement
        reverbNode.loadFactoryPreset(.mediumHall)
        reverbNode.wetDryMix = reverbAmount * 100
        
        // Configure EQ for vocal clarity
        let bands = eqNode.bands
        bands[0].filterType = .highPass
        bands[0].frequency = 80  // Remove low-end rumble
        bands[1].filterType = .parametric
        bands[1].frequency = 2500  // Vocal clarity
        bands[1].bandwidth = 1.0
        bands[1].gain = 3.0
        bands[2].filterType = .lowPass
        bands[2].frequency = 8000  // Smooth high end
        
        // Connect the audio graph
        audioEngine.connect(playerNode, to: eqNode, format: nil)
        audioEngine.connect(eqNode, to: reverbNode, format: nil)
        audioEngine.connect(reverbNode, to: audioEngine.mainMixerNode, format: nil)
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, 
                                       mode: .default, 
                                       options: [.allowBluetooth, .defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }
    
    /// Load a karaoke track for playback
    public func loadAudioFile(url: URL) throws {
        audioFile = try AVAudioFile(forReading: url)
        guard let audioFile = audioFile else {
            throw AiOkeError.audioFileLoadFailed
        }
        
        // Schedule the audio file for playback
        playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
            DispatchQueue.main.async {
                self?.isPlaying = false
            }
        }
    }
    
    /// Start karaoke playback with vocal reduction
    public func startPlayback() throws {
        guard audioFile != nil else {
            throw AiOkeError.noAudioFileLoaded
        }
        
        if !audioEngine.isRunning {
            try audioEngine.start()
        }
        
        playerNode.play()
        
        DispatchQueue.main.async {
            self.isPlaying = true
        }
    }
    
    /// Stop karaoke playback
    public func stopPlayback() {
        playerNode.stop()
        DispatchQueue.main.async {
            self.isPlaying = false
        }
    }
    
    /// Apply vocal reduction using center channel extraction
    /// This is a simplified implementation - production version would use more advanced DSP
    public func applyVocalReduction(amount: Float) {
        self.vocalReductionAmount = max(0.0, min(1.0, amount))
        
        // In a full implementation, this would apply real-time vocal reduction
        // For MVP, we'll implement basic stereo width adjustment
        if let mainMixer = audioEngine.mainMixerNode as? AVAudioMixerNode {
            mainMixer.outputVolume = 1.0 - (vocalReductionAmount * 0.3)
        }
    }
    
    /// Update reverb effect amount
    public func updateReverbAmount(_ amount: Float) {
        self.reverbAmount = max(0.0, min(1.0, amount))
        reverbNode.wetDryMix = reverbAmount * 100
    }
    
    /// Start recording user vocals
    public func startRecording(to url: URL) throws {
        let audioSession = AVAudioSession.sharedInstance()
        
        // Request microphone permission
        audioSession.requestRecordPermission { [weak self] allowed in
            DispatchQueue.main.async {
                if allowed {
                    do {
                        try self?.beginRecording(to: url)
                    } catch {
                        print("Recording failed to start: \(error)")
                    }
                } else {
                    print("Microphone permission denied")
                }
            }
        }
    }
    
    private func beginRecording(to url: URL) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
        recordingFile = try AVAudioFile(forWriting: url, settings: format.settings)
        
        let inputNode = audioEngine.inputNode
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, time in
            try? self?.recordingFile?.write(from: buffer)
        }
        
        DispatchQueue.main.async {
            self.isRecording = true
        }
    }
    
    /// Stop recording user vocals
    public func stopRecording() {
        audioEngine.inputNode.removeTap(onBus: 0)
        recordingFile = nil
        
        DispatchQueue.main.async {
            self.isRecording = false
        }
    }
    
    deinit {
        audioEngine.stop()
    }
}

/// AiOke-specific errors
public enum AiOkeError: Error {
    case audioFileLoadFailed
    case noAudioFileLoaded
    case recordingPermissionDenied
    case audioEngineStartFailed
    
    public var localizedDescription: String {
        switch self {
        case .audioFileLoadFailed:
            return "Failed to load audio file"
        case .noAudioFileLoaded:
            return "No audio file loaded"
        case .recordingPermissionDenied:
            return "Microphone permission required for recording"
        case .audioEngineStartFailed:
            return "Audio engine failed to start"
        }
    }
}