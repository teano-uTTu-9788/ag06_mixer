import Foundation
import AVFoundation

// MARK: - Recording Model
struct Recording: Identifiable, Codable {
    let id = UUID()
    let fileName: String
    let title: String
    let duration: TimeInterval
    let createdAt: Date
    let songTitle: String?
    
    var fileURL: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, 
                                                   in: .userDomainMask)[0]
        return documentsPath.appendingPathComponent(fileName)
    }
    
    var formattedDuration: String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }
}

// MARK: - Recording Manager
class RecordingManager: ObservableObject {
    @Published var recordings: [Recording] = []
    @Published var isRecording: Bool = false
    @Published var currentRecording: Recording?
    @Published var recordingError: String?
    
    private var audioRecorder: AVAudioRecorder?
    private var recordingSession: AVAudioSession?
    private let recordingsKey = "AiOkeRecordings"
    
    init() {
        loadRecordings()
        setupAudioSession()
    }
    
    // MARK: - Audio Session Setup
    private func setupAudioSession() {
        recordingSession = AVAudioSession.sharedInstance()
        
        do {
            try recordingSession?.setCategory(.playAndRecord, 
                                            mode: .default, 
                                            options: [.defaultToSpeaker])
            try recordingSession?.setActive(true)
        } catch {
            recordingError = "Failed to setup recording session: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Recording Control
    func startRecording(songTitle: String? = nil) {
        guard !isRecording else { return }
        
        requestMicrophonePermission { [weak self] granted in
            DispatchQueue.main.async {
                if granted {
                    self?.beginRecording(songTitle: songTitle)
                } else {
                    self?.recordingError = "Microphone permission is required for recording"
                }
            }
        }
    }
    
    private func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        switch AVAudioSession.sharedInstance().recordPermission {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                completion(granted)
            }
        @unknown default:
            completion(false)
        }
    }
    
    private func beginRecording(songTitle: String?) {
        let fileName = "recording_\(Date().timeIntervalSince1970).m4a"
        let documentsPath = FileManager.default.urls(for: .documentDirectory, 
                                                   in: .userDomainMask)[0]
        let audioURL = documentsPath.appendingPathComponent(fileName)
        
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44100,
            AVNumberOfChannelsKey: 2,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        do {
            audioRecorder = try AVAudioRecorder(url: audioURL, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.isMeteringEnabled = true
            
            if audioRecorder?.record() == true {
                isRecording = true
                recordingError = nil
                
                // Create temporary recording object
                currentRecording = Recording(
                    fileName: fileName,
                    title: "Recording in Progress...",
                    duration: 0,
                    createdAt: Date(),
                    songTitle: songTitle
                )
            } else {
                recordingError = "Failed to start recording"
            }
        } catch {
            recordingError = "Recording setup failed: \(error.localizedDescription)"
        }
    }
    
    func stopRecording() {
        guard isRecording, let recorder = audioRecorder else { return }
        
        recorder.stop()
        isRecording = false
        
        // Create final recording object
        if let current = currentRecording {
            let finalRecording = Recording(
                fileName: current.fileName,
                title: generateRecordingTitle(),
                duration: recorder.currentTime,
                createdAt: current.createdAt,
                songTitle: current.songTitle
            )
            
            recordings.insert(finalRecording, at: 0)
            saveRecordings()
        }
        
        currentRecording = nil
        audioRecorder = nil
    }
    
    private func generateRecordingTitle() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, h:mm a"
        return "Recording • \(formatter.string(from: Date()))"
    }
    
    // MARK: - Playback Control
    func playRecording(_ recording: Recording) {
        // This would integrate with the audio engine for playback
        // For now, we'll use a simple AVAudioPlayer approach
        do {
            let player = try AVAudioPlayer(contentsOf: recording.fileURL)
            player.play()
        } catch {
            recordingError = "Failed to play recording: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Recording Management
    func deleteRecording(_ recording: Recording) {
        // Remove from array
        recordings.removeAll { $0.id == recording.id }
        
        // Delete file
        do {
            try FileManager.default.removeItem(at: recording.fileURL)
        } catch {
            recordingError = "Failed to delete recording file: \(error.localizedDescription)"
        }
        
        saveRecordings()
    }
    
    func renameRecording(_ recording: Recording, to newTitle: String) {
        if let index = recordings.firstIndex(where: { $0.id == recording.id }) {
            let updatedRecording = Recording(
                fileName: recording.fileName,
                title: newTitle,
                duration: recording.duration,
                createdAt: recording.createdAt,
                songTitle: recording.songTitle
            )
            recordings[index] = updatedRecording
            saveRecordings()
        }
    }
    
    // MARK: - Data Persistence
    private func saveRecordings() {
        do {
            let data = try JSONEncoder().encode(recordings)
            UserDefaults.standard.set(data, forKey: recordingsKey)
        } catch {
            recordingError = "Failed to save recordings: \(error.localizedDescription)"
        }
    }
    
    private func loadRecordings() {
        guard let data = UserDefaults.standard.data(forKey: recordingsKey),
              let savedRecordings = try? JSONDecoder().decode([Recording].self, from: data) else {
            return
        }
        
        // Filter out recordings that no longer exist on disk
        recordings = savedRecordings.filter { recording in
            FileManager.default.fileExists(atPath: recording.fileURL.path)
        }
        
        // Save filtered list if any recordings were removed
        if recordings.count != savedRecordings.count {
            saveRecordings()
        }
    }
    
    // MARK: - Recording Statistics
    var totalRecordings: Int {
        recordings.count
    }
    
    var totalDuration: TimeInterval {
        recordings.reduce(0) { $0 + $1.duration }
    }
    
    var formattedTotalDuration: String {
        let hours = Int(totalDuration) / 3600
        let minutes = Int(totalDuration) / 60 % 60
        
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }
}

// MARK: - AVAudioRecorderDelegate
extension RecordingManager: AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if !flag {
            recordingError = "Recording failed to complete"
            isRecording = false
            currentRecording = nil
        }
    }
    
    func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        recordingError = "Recording encoding error: \(error?.localizedDescription ?? "Unknown error")"
        isRecording = false
        currentRecording = nil
    }
}