import SwiftUI
import AVFoundation

struct SettingsView: View {
    @EnvironmentObject var audioEngine: AiOkeAudioEngine
    @EnvironmentObject var recordingManager: RecordingManager
    @AppStorage("audioQuality") private var audioQuality: AudioQuality = .high
    @AppStorage("autoRecordingEnabled") private var autoRecordingEnabled = false
    @AppStorage("visualEffectsEnabled") private var visualEffectsEnabled = true
    @AppStorage("hapticFeedbackEnabled") private var hapticFeedbackEnabled = true
    
    @State private var showingAboutView = false
    @State private var showingPrivacyPolicy = false
    @State private var showingTermsOfService = false
    @State private var showingResetAlert = false
    
    var body: some View {
        NavigationView {
            List {
                // Audio Settings Section
                audioSettingsSection
                
                // Recording Settings Section
                recordingSettingsSection
                
                // Interface Settings Section
                interfaceSettingsSection
                
                // Privacy & Legal Section
                privacyLegalSection
                
                // Support Section
                supportSection
                
                // App Info Section
                appInfoSection
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showingAboutView) {
                AboutView()
            }
            .alert("Reset All Settings", isPresented: $showingResetAlert) {
                Button("Reset", role: .destructive) {
                    resetAllSettings()
                }
                Button("Cancel", role: .cancel) { }
            } message: {
                Text("This will reset all app settings to their default values.")
            }
        }
    }
    
    // MARK: - Audio Settings Section
    private var audioSettingsSection: some View {
        Section {
            // Audio Quality Picker
            Picker("Audio Quality", selection: $audioQuality) {
                ForEach(AudioQuality.allCases, id: \.self) { quality in
                    Text(quality.displayName).tag(quality)
                }
            }
            .pickerStyle(MenuPickerStyle())
            
            // Audio buffer info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Buffer Size")
                    Spacer()
                    Text(audioQuality.bufferDescription)
                        .foregroundColor(.secondary)
                }
                
                Text("Lower buffer = less delay, higher CPU usage")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
        } header: {
            Label("Audio Settings", systemImage: "speaker.wave.3")
        } footer: {
            Text("Higher quality settings may drain battery faster but provide better audio performance.")
        }
    }
    
    // MARK: - Recording Settings Section
    private var recordingSettingsSection: some View {
        Section {
            Toggle("Auto-Record Sessions", isOn: $autoRecordingEnabled)
            
            // Storage usage
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Storage Used")
                    Spacer()
                    Text(recordingStorageUsed)
                        .foregroundColor(.secondary)
                }
                
                if recordingManager.totalRecordings > 0 {
                    HStack {
                        Text("Total Recordings")
                        Spacer()
                        Text("\(recordingManager.totalRecordings)")
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            if recordingManager.totalRecordings > 0 {
                Button("Clear All Recordings", role: .destructive) {
                    clearAllRecordings()
                }
            }
            
        } header: {
            Label("Recording Settings", systemImage: "waveform")
        } footer: {
            Text("Auto-record will automatically start recording when you begin karaoke.")
        }
    }
    
    // MARK: - Interface Settings Section
    private var interfaceSettingsSection: some View {
        Section {
            Toggle("Visual Effects", isOn: $visualEffectsEnabled)
            Toggle("Haptic Feedback", isOn: $hapticFeedbackEnabled)
            
        } header: {
            Label("Interface", systemImage: "paintbrush")
        } footer: {
            Text("Disable visual effects to improve performance on older devices.")
        }
    }
    
    // MARK: - Privacy & Legal Section
    private var privacyLegalSection: some View {
        Section {
            Button("Privacy Policy") {
                showingPrivacyPolicy = true
            }
            
            Button("Terms of Service") {
                showingTermsOfService = true
            }
            
            // Microphone permission status
            microphonePermissionView
            
        } header: {
            Label("Privacy & Legal", systemImage: "hand.raised")
        }
    }
    
    private var microphonePermissionView: some View {
        HStack {
            Text("Microphone Access")
            Spacer()
            
            switch AVAudioSession.sharedInstance().recordPermission {
            case .granted:
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Allowed")
                        .foregroundColor(.green)
                }
            case .denied:
                HStack(spacing: 4) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.red)
                    Button("Enable in Settings") {
                        openAppSettings()
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
            case .undetermined:
                Text("Not Requested")
                    .foregroundColor(.secondary)
            @unknown default:
                Text("Unknown")
                    .foregroundColor(.secondary)
            }
        }
    }
    
    // MARK: - Support Section
    private var supportSection: some View {
        Section {
            Button("Send Feedback") {
                sendFeedback()
            }
            
            Button("Report Issue") {
                reportIssue()
            }
            
            Button("Rate AiOke") {
                rateApp()
            }
            
            Button("Reset All Settings", role: .destructive) {
                showingResetAlert = true
            }
            
        } header: {
            Label("Support", systemImage: "questionmark.circle")
        }
    }
    
    // MARK: - App Info Section
    private var appInfoSection: some View {
        Section {
            HStack {
                Text("Version")
                Spacer()
                Text(appVersion)
                    .foregroundColor(.secondary)
            }
            
            HStack {
                Text("Build")
                Spacer()
                Text(buildNumber)
                    .foregroundColor(.secondary)
            }
            
            Button("About AiOke") {
                showingAboutView = true
            }
            
        } header: {
            Label("App Information", systemImage: "info.circle")
        }
    }
    
    // MARK: - Computed Properties
    private var recordingStorageUsed: String {
        let totalDuration = recordingManager.totalDuration
        let estimatedSizeMB = totalDuration * 0.5 // Rough estimate: 0.5MB per minute
        
        if estimatedSizeMB < 1 {
            return "< 1 MB"
        } else if estimatedSizeMB < 1000 {
            return String(format: "%.0f MB", estimatedSizeMB)
        } else {
            return String(format: "%.1f GB", estimatedSizeMB / 1000)
        }
    }
    
    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
    }
    
    private var buildNumber: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown"
    }
    
    // MARK: - Actions
    private func resetAllSettings() {
        audioQuality = .high
        autoRecordingEnabled = false
        visualEffectsEnabled = true
        hapticFeedbackEnabled = true
    }
    
    private func clearAllRecordings() {
        for recording in recordingManager.recordings {
            recordingManager.deleteRecording(recording)
        }
    }
    
    private func sendFeedback() {
        if let url = URL(string: "mailto:feedback@aioke.app?subject=AiOke%20Feedback") {
            UIApplication.shared.open(url)
        }
    }
    
    private func reportIssue() {
        if let url = URL(string: "mailto:support@aioke.app?subject=AiOke%20Issue%20Report") {
            UIApplication.shared.open(url)
        }
    }
    
    private func rateApp() {
        if let url = URL(string: "https://apps.apple.com/app/aioke/id123456789?action=write-review") {
            UIApplication.shared.open(url)
        }
    }
    
    private func openAppSettings() {
        if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(settingsUrl)
        }
    }
}

// MARK: - Audio Quality Enum
enum AudioQuality: String, CaseIterable {
    case low = "low"
    case medium = "medium" 
    case high = "high"
    case max = "max"
    
    var displayName: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .max: return "Maximum"
        }
    }
    
    var bufferDescription: String {
        switch self {
        case .low: return "2048 samples"
        case .medium: return "1024 samples"
        case .high: return "512 samples"
        case .max: return "256 samples"
        }
    }
    
    var bufferSize: Int {
        switch self {
        case .low: return 2048
        case .medium: return 1024
        case .high: return 512
        case .max: return 256
        }
    }
}

// MARK: - About View
struct AboutView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // App icon and title
                    VStack(spacing: 16) {
                        Image(systemName: "music.mic.circle.fill")
                            .font(.system(size: 80))
                            .foregroundColor(.purple)
                        
                        VStack(spacing: 8) {
                            Text("AiOke")
                                .font(.largeTitle)
                                .fontWeight(.bold)
                            
                            Text("AI-Powered Karaoke Experience")
                                .font(.headline)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Description
                    VStack(alignment: .leading, spacing: 16) {
                        Text("About AiOke")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("AiOke transforms your iPhone into a professional karaoke machine with AI-powered vocal reduction, real-time effects, and high-quality recording capabilities.")
                            .font(.body)
                            .foregroundColor(.secondary)
                        
                        Text("Perfect for parties, practice sessions, or just having fun with friends and family!")
                            .font(.body)
                            .foregroundColor(.secondary)
                    }
                    
                    // Features
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Key Features")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            featureRow("🎤", "Real-time vocal reduction")
                            featureRow("🎵", "Professional audio effects")
                            featureRow("📱", "Intuitive iOS interface")
                            featureRow("📹", "High-quality recording")
                            featureRow("🎧", "Bluetooth headphone support")
                            featureRow("🌟", "No internet required")
                        }
                    }
                    
                    // Credits
                    VStack(spacing: 12) {
                        Text("Made with ❤️ for karaoke lovers")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Text("© 2024 AiOke. All rights reserved.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func featureRow(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 12) {
            Text(icon)
                .font(.title3)
            
            Text(text)
                .font(.subheadline)
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(AiOkeAudioEngine())
        .environmentObject(RecordingManager())
}