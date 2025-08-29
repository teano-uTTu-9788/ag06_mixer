import Foundation
import AVFoundation
import Accelerate
import Combine

// MARK: - Audio Engine for Karaoke Processing
class AiOkeAudioEngine: ObservableObject {
    // MARK: - Published Properties
    @Published var isPlaying: Bool = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var vocalReductionLevel: Float = 0.5
    @Published var reverbLevel: Float = 0.3
    @Published var microphoneVolume: Float = 0.8
    @Published var musicVolume: Float = 0.7
    @Published var isRecording: Bool = false
    @Published var audioEngineError: String?
    
    // MARK: - Core Audio Components
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let microphoneNode: AVAudioInputNode
    private let reverbNode = AVAudioUnitReverb()
    private let vocalReductionNode = AVAudioUnitEQ()
    private let musicMixerNode = AVAudioMixerNode()
    private let micMixerNode = AVAudioMixerNode()
    private let mainMixerNode = AVAudioMixerNode()
    
    // MARK: - Recording Components
    private var audioFile: AVAudioFile?
    private var recordingFile: AVAudioFile?
    
    // MARK: - Playback State
    private var currentSong: Song?
    private var playbackTimer: Timer?
    
    // MARK: - Audio Format
    private let audioFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
    
    init() {
        microphoneNode = engine.inputNode
        setupAudioEngine()
    }
    
    // MARK: - Audio Engine Setup
    private func setupAudioEngine() {
        // Connect nodes to engine
        engine.attach(playerNode)
        engine.attach(reverbNode)
        engine.attach(vocalReductionNode)
        engine.attach(musicMixerNode)
        engine.attach(micMixerNode)
        engine.attach(mainMixerNode)
        
        // Configure reverb
        reverbNode.loadFactoryPreset(.mediumHall)
        reverbNode.wetDryMix = 30
        
        // Configure vocal reduction EQ
        setupVocalReductionEQ()
        
        // Connect audio graph
        connectAudioNodes()
        
        // Start the engine
        startEngine()
    }
    
    private func setupVocalReductionEQ() {
        // Configure EQ bands for vocal reduction
        let bands = vocalReductionNode.bands
        
        // Reduce vocal frequency range (200-3000 Hz)
        if bands.count > 0 {
            bands[0].frequency = 1000 // Center vocal frequency
            bands[0].gain = -6 // Reduce by 6dB
            bands[0].bandwidth = 2 // Wide bandwidth
            bands[0].filterType = .parametric
            bands[0].bypass = false
        }
    }
    
    private func connectAudioNodes() {
        // Music path: Player -> Vocal Reduction -> Music Mixer
        engine.connect(playerNode, to: vocalReductionNode, format: audioFormat)
        engine.connect(vocalReductionNode, to: musicMixerNode, format: audioFormat)
        
        // Microphone path: Input -> Reverb -> Mic Mixer
        engine.connect(microphoneNode, to: reverbNode, format: audioFormat)
        engine.connect(reverbNode, to: micMixerNode, format: audioFormat)
        
        // Final mix: Both mixers -> Main mixer -> Output
        engine.connect(musicMixerNode, to: mainMixerNode, format: audioFormat)
        engine.connect(micMixerNode, to: mainMixerNode, format: audioFormat)
        engine.connect(mainMixerNode, to: engine.mainMixerNode, format: audioFormat)
    }
    
    private func startEngine() {
        do {
            try engine.start()
        } catch {
            audioEngineError = "Failed to start audio engine: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Playback Control
    func loadSong(_ song: Song) {
        guard let audioURL = song.audioURL else {
            audioEngineError = "Could not find audio file for \(song.title)"
            return
        }
        
        do {
            audioFile = try AVAudioFile(forReading: audioURL)
            currentSong = song
            duration = Double(audioFile?.length ?? 0) / audioFile!.processingFormat.sampleRate
            currentTime = 0
            audioEngineError = nil
        } catch {
            audioEngineError = "Failed to load song: \(error.localizedDescription)"
        }
    }
    
    func play() {
        guard let audioFile = audioFile else { return }
        
        // Schedule the audio file
        playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
            DispatchQueue.main.async {
                self?.isPlaying = false
                self?.stopPlaybackTimer()
            }
        }
        
        playerNode.play()
        isPlaying = true
        startPlaybackTimer()
    }
    
    func pause() {
        playerNode.pause()
        isPlaying = false
        stopPlaybackTimer()
    }
    
    func stop() {
        playerNode.stop()
        isPlaying = false
        currentTime = 0
        stopPlaybackTimer()
    }
    
    func seek(to time: TimeInterval) {
        let wasPlaying = isPlaying
        
        stop()
        
        guard let audioFile = audioFile else { return }
        
        let sampleRate = audioFile.processingFormat.sampleRate
        let startFrame = AVAudioFramePosition(time * sampleRate)
        
        if startFrame < audioFile.length {
            let frameCount = AVAudioFrameCount(audioFile.length - startFrame)
            
            playerNode.scheduleSegment(audioFile, 
                                     startingFrame: startFrame, 
                                     frameCount: frameCount, 
                                     at: nil) { [weak self] in
                DispatchQueue.main.async {
                    self?.isPlaying = false
                    self?.stopPlaybackTimer()
                }
            }
            
            currentTime = time
            
            if wasPlaying {
                play()
            }
        }
    }
    
    // MARK: - Audio Effects Control
    func setVocalReductionLevel(_ level: Float) {
        vocalReductionLevel = level
        
        // Apply vocal reduction by adjusting EQ gain
        let bands = vocalReductionNode.bands
        if bands.count > 0 {
            bands[0].gain = -12 * level // Scale from 0 to -12dB
        }
    }
    
    func setReverbLevel(_ level: Float) {
        reverbLevel = level
        reverbNode.wetDryMix = level * 100 // Convert to percentage
    }
    
    func setMicrophoneVolume(_ level: Float) {
        microphoneVolume = level
        micMixerNode.outputVolume = level
    }
    
    func setMusicVolume(_ level: Float) {
        musicVolume = level
        musicMixerNode.outputVolume = level
    }
    
    // MARK: - Recording Functionality
    func startRecording() {
        guard !isRecording else { return }
        
        do {
            // Create recording file
            let documentsPath = FileManager.default.urls(for: .documentDirectory, 
                                                       in: .userDomainMask)[0]
            let recordingURL = documentsPath.appendingPathComponent("recording_\(Date().timeIntervalSince1970).m4a")
            
            recordingFile = try AVAudioFile(forWriting: recordingURL, 
                                          settings: audioFormat.settings)
            
            // Install tap on main mixer to capture mixed audio
            mainMixerNode.installTap(onBus: 0, 
                                   bufferSize: 1024, 
                                   format: audioFormat) { [weak self] (buffer, _) in
                do {
                    try self?.recordingFile?.write(from: buffer)
                } catch {
                    DispatchQueue.main.async {
                        self?.audioEngineError = "Recording failed: \(error.localizedDescription)"
                    }
                }
            }
            
            isRecording = true
        } catch {
            audioEngineError = "Failed to start recording: \(error.localizedDescription)"
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        mainMixerNode.removeTap(onBus: 0)
        recordingFile = nil
        isRecording = false
    }
    
    // MARK: - Playback Timer
    private func startPlaybackTimer() {
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateCurrentTime()
        }
    }
    
    private func stopPlaybackTimer() {
        playbackTimer?.invalidate()
        playbackTimer = nil
    }
    
    private func updateCurrentTime() {
        guard let audioFile = audioFile,
              let nodeTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: nodeTime) else {
            return
        }
        
        let sampleRate = audioFile.processingFormat.sampleRate
        currentTime = Double(playerTime.sampleTime) / sampleRate
    }
    
    // MARK: - Cleanup
    deinit {
        engine.stop()
        stopPlaybackTimer()
    }
}

// MARK: - Audio Engine Extensions
extension AiOkeAudioEngine {
    var progress: Double {
        guard duration > 0 else { return 0 }
        return currentTime / duration
    }
    
    var formattedCurrentTime: String {
        return formatTime(currentTime)
    }
    
    var formattedDuration: String {
        return formatTime(duration)
    }
    
    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}