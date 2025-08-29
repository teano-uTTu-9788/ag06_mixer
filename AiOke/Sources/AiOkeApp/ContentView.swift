import SwiftUI
import AiOke

/// Main AiOke app interface following iOS Human Interface Guidelines
struct ContentView: View {
    @StateObject private var audioEngine = AiOkeAudioEngine()
    @State private var selectedSong: DemoSong?
    @State private var showingSongPicker = false
    @State private var errorMessage: String?
    
    private let demoSongs: [DemoSong] = [
        DemoSong(title: "Demo Track 1", artist: "AiOke", filename: "demo1"),
        DemoSong(title: "Demo Track 2", artist: "AiOke", filename: "demo2"),
        DemoSong(title: "Demo Track 3", artist: "AiOke", filename: "demo3")
    ]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header with app logo and title
                VStack(spacing: 10) {
                    Image(systemName: "music.mic")
                        .font(.system(size: 60))
                        .foregroundColor(.purple)
                    
                    Text("AiOke")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("AI-Powered Karaoke")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 20)
                
                Spacer()
                
                // Song selection
                VStack(spacing: 15) {
                    Text("Select a Song")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Button(action: {
                        showingSongPicker = true
                    }) {
                        HStack {
                            Image(systemName: "music.note.list")
                            Text(selectedSong?.title ?? "Choose Demo Track")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    }
                    .foregroundColor(.primary)
                }
                .padding(.horizontal)
                
                // Audio controls
                VStack(spacing: 20) {
                    // Play/Pause button
                    Button(action: togglePlayback) {
                        HStack {
                            Image(systemName: audioEngine.isPlaying ? "pause.fill" : "play.fill")
                                .font(.title2)
                            Text(audioEngine.isPlaying ? "Pause" : "Play")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(selectedSong != nil ? Color.purple : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(25)
                    }
                    .disabled(selectedSong == nil)
                    
                    // Recording button
                    Button(action: toggleRecording) {
                        HStack {
                            Image(systemName: audioEngine.isRecording ? "stop.fill" : "mic.fill")
                                .font(.title2)
                            Text(audioEngine.isRecording ? "Stop Recording" : "Record Vocals")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(audioEngine.isRecording ? Color.red : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(25)
                    }
                }
                .padding(.horizontal)
                
                // Audio controls
                VStack(spacing: 15) {
                    // Vocal reduction slider
                    VStack {
                        HStack {
                            Text("Vocal Reduction")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(audioEngine.vocalReductionAmount * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(value: Binding(
                            get: { audioEngine.vocalReductionAmount },
                            set: { audioEngine.applyVocalReduction(amount: $0) }
                        ), in: 0...1)
                        .accentColor(.purple)
                    }
                    
                    // Reverb slider
                    VStack {
                        HStack {
                            Text("Reverb")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(audioEngine.reverbAmount * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Slider(value: Binding(
                            get: { audioEngine.reverbAmount },
                            set: { audioEngine.updateReverbAmount($0) }
                        ), in: 0...1)
                        .accentColor(.blue)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Footer with version info
                Text("AiOke MVP v1.0")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 20)
            }
            .navigationTitle("AiOke")
            .navigationBarHidden(true)
            .sheet(isPresented: $showingSongPicker) {
                SongPickerView(songs: demoSongs, selectedSong: $selectedSong)
            }
            .alert("Error", isPresented: .constant(errorMessage != nil)) {
                Button("OK") {
                    errorMessage = nil
                }
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }
    
    private func togglePlayback() {
        guard let selectedSong = selectedSong else { return }
        
        do {
            if audioEngine.isPlaying {
                audioEngine.stopPlayback()
            } else {
                // Load demo song from bundle (in production app)
                // For now, we'll simulate loading
                let demoURL = Bundle.main.url(forResource: selectedSong.filename, withExtension: "mp3") ??
                             URL(fileURLWithPath: "/System/Library/Sounds/Ping.aiff") // Fallback
                
                try audioEngine.loadAudioFile(url: demoURL)
                try audioEngine.startPlayback()
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
    
    private func toggleRecording() {
        do {
            if audioEngine.isRecording {
                audioEngine.stopRecording()
            } else {
                let documentsURL = FileManager.default.urls(for: .documentDirectory, 
                                                          in: .userDomainMask).first!
                let recordingURL = documentsURL.appendingPathComponent("recording.m4a")
                try audioEngine.startRecording(to: recordingURL)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

/// Song picker sheet view
struct SongPickerView: View {
    let songs: [DemoSong]
    @Binding var selectedSong: DemoSong?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            List(songs) { song in
                Button(action: {
                    selectedSong = song
                    dismiss()
                }) {
                    VStack(alignment: .leading) {
                        Text(song.title)
                            .font(.headline)
                        Text(song.artist)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .foregroundColor(.primary)
                }
            }
            .navigationTitle("Choose Song")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

/// Demo song model
struct DemoSong: Identifiable, Hashable {
    let id = UUID()
    let title: String
    let artist: String
    let filename: String
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}