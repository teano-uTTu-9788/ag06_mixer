import SwiftUI
import AVFoundation

@main
struct AiOkeApp: App {
    @StateObject private var audioEngine = AiOkeAudioEngine()
    @StateObject private var songManager = SongManager()
    @StateObject private var recordingManager = RecordingManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(audioEngine)
                .environmentObject(songManager)
                .environmentObject(recordingManager)
                .onAppear {
                    setupAudioSession()
                }
        }
    }
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, 
                                       mode: .default, 
                                       options: [.defaultToSpeaker, .allowBluetooth])
            try audioSession.setActive(true)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var audioEngine: AiOkeAudioEngine
    @EnvironmentObject var songManager: SongManager
    @EnvironmentObject var recordingManager: RecordingManager
    
    var body: some View {
        TabView {
            KaraokeMainView()
                .tabItem {
                    Image(systemName: "music.mic")
                    Text("Karaoke")
                }
            
            SongLibraryView()
                .tabItem {
                    Image(systemName: "music.note.list")
                    Text("Songs")
                }
            
            RecordingsView()
                .tabItem {
                    Image(systemName: "waveform")
                    Text("Recordings")
                }
            
            SettingsView()
                .tabItem {
                    Image(systemName: "gearshape")
                    Text("Settings")
                }
        }
        .accentColor(.purple)
    }
}

#Preview {
    ContentView()
        .environmentObject(AiOkeAudioEngine())
        .environmentObject(SongManager())
        .environmentObject(RecordingManager())
}