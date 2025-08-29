import SwiftUI
import AVFoundation

struct KaraokeMainView: View {
    @EnvironmentObject var audioEngine: AiOkeAudioEngine
    @EnvironmentObject var songManager: SongManager
    @EnvironmentObject var recordingManager: RecordingManager
    
    @State private var selectedSong: Song?
    @State private var showingSongPicker = false
    @State private var showingEffectsPanel = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with song info
                songHeaderView
                
                // Main controls area
                mainControlsView
                    .padding(.horizontal, 20)
                
                // Audio effects panel (collapsible)
                if showingEffectsPanel {
                    audioEffectsPanel
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                
                Spacer()
                
                // Bottom controls
                bottomControlsView
                    .padding(.horizontal, 20)
                    .padding(.bottom, 34) // Safe area padding
            }
            .navigationTitle("AiOke")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Songs") {
                        showingSongPicker = true
                    }
                }
            }
            .sheet(isPresented: $showingSongPicker) {
                SongPickerView(selectedSong: $selectedSong)
            }
            .onChange(of: selectedSong) { newSong in
                if let song = newSong {
                    audioEngine.loadSong(song)
                }
            }
            .onAppear {
                // Load first demo song if none selected
                if selectedSong == nil && !songManager.songs.isEmpty {
                    selectedSong = songManager.songs.first
                    audioEngine.loadSong(songManager.songs.first!)
                }
            }
        }
    }
    
    // MARK: - Song Header View
    private var songHeaderView: some View {
        VStack(spacing: 12) {
            // Song artwork placeholder
            RoundedRectangle(cornerRadius: 12)
                .fill(LinearGradient(
                    colors: [.purple.opacity(0.8), .blue.opacity(0.6)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))
                .frame(height: 200)
                .overlay {
                    VStack {
                        Image(systemName: "music.note")
                            .font(.system(size: 48))
                            .foregroundColor(.white)
                        
                        if let song = selectedSong {
                            Text(song.title)
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                            
                            Text(song.artist)
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.8))
                        } else {
                            Text("No Song Selected")
                                .font(.title2)
                                .fontWeight(.medium)
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                }
                .padding(.horizontal, 20)
            
            // Progress bar
            progressBarView
                .padding(.horizontal, 20)
        }
        .padding(.top, 20)
        .background(Color(UIColor.systemBackground))
    }
    
    // MARK: - Progress Bar
    private var progressBarView: some View {
        VStack(spacing: 8) {
            // Time labels
            HStack {
                Text(audioEngine.formattedCurrentTime)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(audioEngine.formattedDuration)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Progress slider
            Slider(value: Binding(
                get: { audioEngine.progress },
                set: { newValue in
                    let seekTime = newValue * audioEngine.duration
                    audioEngine.seek(to: seekTime)
                }
            ))
            .accentColor(.purple)
        }
    }
    
    // MARK: - Main Controls
    private var mainControlsView: some View {
        VStack(spacing: 30) {
            // Playback controls
            HStack(spacing: 40) {
                Button(action: {
                    // Previous song (future feature)
                }) {
                    Image(systemName: "backward.fill")
                        .font(.title2)
                        .foregroundColor(.gray)
                }
                .disabled(true)
                
                // Main play/pause button
                Button(action: {
                    if audioEngine.isPlaying {
                        audioEngine.pause()
                    } else {
                        audioEngine.play()
                    }
                }) {
                    Image(systemName: audioEngine.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 64))
                        .foregroundColor(.purple)
                }
                .disabled(selectedSong == nil)
                
                // Stop button
                Button(action: {
                    audioEngine.stop()
                }) {
                    Image(systemName: "stop.fill")
                        .font(.title2)
                        .foregroundColor(audioEngine.isPlaying ? .primary : .gray)
                }
                .disabled(!audioEngine.isPlaying)
                
                Button(action: {
                    // Next song (future feature)
                }) {
                    Image(systemName: "forward.fill")
                        .font(.title2)
                        .foregroundColor(.gray)
                }
                .disabled(true)
            }
            
            // Recording control
            recordingControlView
        }
        .padding(.vertical, 20)
    }
    
    // MARK: - Recording Control
    private var recordingControlView: some View {
        HStack(spacing: 16) {
            // Recording button
            Button(action: {
                if recordingManager.isRecording {
                    recordingManager.stopRecording()
                } else {
                    recordingManager.startRecording(songTitle: selectedSong?.title)
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: recordingManager.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .foregroundColor(recordingManager.isRecording ? .red : .purple)
                    
                    Text(recordingManager.isRecording ? "Stop Recording" : "Record")
                        .fontWeight(.medium)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 25)
                        .fill(Color(UIColor.systemGray6))
                )
            }
            
            // Effects panel toggle
            Button(action: {
                withAnimation(.spring()) {
                    showingEffectsPanel.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: showingEffectsPanel ? "waveform.path.ecg.rectangle.fill" : "waveform.path.ecg.rectangle")
                        .foregroundColor(.purple)
                    
                    Text("Effects")
                        .fontWeight(.medium)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 25)
                        .fill(showingEffectsPanel ? Color.purple.opacity(0.1) : Color(UIColor.systemGray6))
                )
            }
        }
    }
    
    // MARK: - Audio Effects Panel
    private var audioEffectsPanel: some View {
        VStack(spacing: 20) {
            Text("Audio Effects")
                .font(.headline)
                .padding(.top)
            
            VStack(spacing: 16) {
                // Vocal Reduction
                effectSlider(
                    title: "Vocal Reduction",
                    value: $audioEngine.vocalReductionLevel,
                    range: 0...1,
                    icon: "person.wave.2.fill",
                    action: audioEngine.setVocalReductionLevel
                )
                
                // Reverb
                effectSlider(
                    title: "Reverb",
                    value: $audioEngine.reverbLevel,
                    range: 0...1,
                    icon: "waveform.path",
                    action: audioEngine.setReverbLevel
                )
                
                // Microphone Volume
                effectSlider(
                    title: "Mic Volume",
                    value: $audioEngine.microphoneVolume,
                    range: 0...1,
                    icon: "mic.fill",
                    action: audioEngine.setMicrophoneVolume
                )
                
                // Music Volume
                effectSlider(
                    title: "Music Volume",
                    value: $audioEngine.musicVolume,
                    range: 0...1,
                    icon: "music.note",
                    action: audioEngine.setMusicVolume
                )
            }
            .padding(.horizontal)
            .padding(.bottom)
        }
        .background(Color(UIColor.systemGray6))
        .cornerRadius(16, corners: [.topLeft, .topRight])
    }
    
    // MARK: - Effect Slider Helper
    private func effectSlider(
        title: String,
        value: Binding<Float>,
        range: ClosedRange<Float>,
        icon: String,
        action: @escaping (Float) -> Void
    ) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.purple)
                .frame(width: 20)
            
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .frame(width: 100, alignment: .leading)
            
            Slider(value: value, in: range) { _ in
                action(value.wrappedValue)
            }
            .accentColor(.purple)
            
            Text("\(Int(value.wrappedValue * 100))%")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(width: 35, alignment: .trailing)
        }
    }
    
    // MARK: - Bottom Controls
    private var bottomControlsView: some View {
        HStack {
            // Quick access to recordings
            NavigationLink(destination: RecordingsView()) {
                VStack(spacing: 4) {
                    Image(systemName: "waveform")
                        .foregroundColor(.purple)
                    Text("Recordings")
                        .font(.caption)
                        .foregroundColor(.purple)
                }
            }
            
            Spacer()
            
            // Song selection shortcut
            Button(action: {
                showingSongPicker = true
            }) {
                VStack(spacing: 4) {
                    Image(systemName: "music.note.list")
                        .foregroundColor(.purple)
                    Text("Songs")
                        .font(.caption)
                        .foregroundColor(.purple)
                }
            }
        }
    }
}

// MARK: - Custom Corner Radius
extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

#Preview {
    KaraokeMainView()
        .environmentObject(AiOkeAudioEngine())
        .environmentObject(SongManager())
        .environmentObject(RecordingManager())
}