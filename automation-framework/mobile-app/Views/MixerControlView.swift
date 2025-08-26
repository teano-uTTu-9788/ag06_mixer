import SwiftUI
import Combine

struct MixerControlView: View {
    @EnvironmentObject var mixerService: MixerService
    @EnvironmentObject var configManager: ConfigurationManager
    @State private var showingSettings = false
    @State private var showingSubscription = false
    @State private var pendingSettings = MixerSettings()
    @State private var isApplyingSettings = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header with connection status
                    headerSection
                    
                    // Audio level meters
                    audioMetersSection
                    
                    // Control transport
                    transportControlsSection
                    
                    // Mixer controls
                    mixerControlsSection
                    
                    // Device information
                    deviceInfoSection
                }
                .padding()
            }
            .navigationTitle("🎚️ AG06 Mixer")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Settings") {
                        showingSettings = true
                    }
                }
            }
            .refreshable {
                await mixerService.refreshStatus()
            }
        }
        .onAppear {
            pendingSettings = mixerService.mixerSettings
            startRealTimeUpdates()
        }
        .onDisappear {
            stopRealTimeUpdates()
        }
        .sheet(isPresented: $showingSettings) {
            MixerSettingsView()
        }
        .sheet(isPresented: $showingSubscription) {
            SubscriptionView()
        }
    }
    
    // MARK: - Header Section
    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("AG06 Mixer")
                    .font(.title2)
                    .fontWeight(.bold)
                
                HStack(spacing: 12) {
                    // Connection indicator
                    HStack(spacing: 4) {
                        Circle()
                            .fill(mixerService.connectionStatus.isConnected ? .green : .red)
                            .frame(width: 8, height: 8)
                        
                        Text(mixerService.connectionStatus.isConnected ? "Connected" : "Disconnected")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    // Latency indicator
                    if let latency = mixerService.connectionStatus.latency {
                        Text(String(format: "%.0fms", latency * 1000))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            // Subscription indicator
            Button(action: { showingSubscription = true }) {
                HStack(spacing: 4) {
                    Text(configManager.configuration.subscriptionTier.displayName)
                        .font(.caption)
                        .fontWeight(.semibold)
                    
                    if configManager.configuration.subscriptionTier != .studio {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.caption)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(subscriptionColor.opacity(0.2))
                .foregroundColor(subscriptionColor)
                .cornerRadius(12)
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Audio Meters Section
    private var audioMetersSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Audio Levels")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(spacing: 12) {
                // RMS Level Meter
                AudioMeter(
                    title: "RMS",
                    value: mixerService.audioMetrics.rmsDB,
                    range: -60...0,
                    color: .blue,
                    unit: "dB"
                )
                
                // Peak Level Meter
                AudioMeter(
                    title: "Peak",
                    value: mixerService.audioMetrics.peakDB,
                    range: -60...0,
                    color: mixerService.audioMetrics.isClipping ? .red : .green,
                    unit: "dB"
                )
                
                // LUFS Meter (Pro+ only)
                if configManager.configuration.subscriptionTier.hasAIProcessing {
                    AudioMeter(
                        title: "LUFS",
                        value: mixerService.audioMetrics.lufsEst,
                        range: -40...0,
                        color: .orange,
                        unit: "LUFS"
                    )
                } else {
                    SubscriptionLockedMeter(feature: "LUFS Metering")
                }
            }
            
            // Status indicators
            HStack(spacing: 16) {
                StatusIndicator(
                    title: "Engine",
                    isActive: mixerService.audioMetrics.isRunning,
                    activeText: "Running",
                    inactiveText: "Stopped"
                )
                
                StatusIndicator(
                    title: "Clipping",
                    isActive: mixerService.audioMetrics.isClipping,
                    activeText: "CLIP",
                    inactiveText: "OK",
                    isWarning: true
                )
                
                if mixerService.audioMetrics.dropouts > 0 {
                    StatusIndicator(
                        title: "Dropouts",
                        isActive: true,
                        activeText: "\(mixerService.audioMetrics.dropouts)",
                        inactiveText: "0",
                        isWarning: true
                    )
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Transport Controls
    private var transportControlsSection: some View {
        HStack(spacing: 20) {
            // Start button
            Button(action: startMixer) {
                HStack(spacing: 8) {
                    Image(systemName: mixerService.audioMetrics.isRunning ? "pause.fill" : "play.fill")
                        .font(.title2)
                    
                    Text(mixerService.audioMetrics.isRunning ? "Stop" : "Start")
                        .font(.headline)
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(mixerService.audioMetrics.isRunning ? .red : .green)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(mixerService.isLoading || !mixerService.connectionStatus.isConnected)
            
            // Refresh button
            Button(action: refreshStatus) {
                Image(systemName: "arrow.clockwise")
                    .font(.title2)
                    .frame(width: 50, height: 50)
                    .background(.secondary.opacity(0.2))
                    .foregroundColor(.primary)
                    .cornerRadius(12)
            }
            .disabled(mixerService.isLoading)
        }
    }
    
    // MARK: - Mixer Controls
    private var mixerControlsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Mix Controls")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(spacing: 20) {
                // AI Mix Control
                ControlSlider(
                    title: "AI Mix",
                    value: $pendingSettings.aiMix,
                    range: 0...1,
                    format: "%.0f%%",
                    multiplier: 100,
                    onChange: scheduleSettingsUpdate
                )
                
                // Target LUFS (Pro+ only)
                if configManager.configuration.subscriptionTier.hasAIProcessing {
                    ControlSlider(
                        title: "Target LUFS",
                        value: $pendingSettings.targetLUFS,
                        range: -30...(-6),
                        format: "%.1f LUFS",
                        onChange: scheduleSettingsUpdate
                    )
                } else {
                    SubscriptionLockedControl(feature: "LUFS Targeting")
                }
                
                // Advanced controls (Studio only)
                if configManager.configuration.subscriptionTier == .studio {
                    Divider()
                    
                    Text("Advanced")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                    
                    HStack(spacing: 16) {
                        VStack {
                            Text("Block Size")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Picker("Block Size", selection: $pendingSettings.blockSize) {
                                Text("64").tag(64)
                                Text("128").tag(128)
                                Text("256").tag(256)
                                Text("512").tag(512)
                                Text("1024").tag(1024)
                            }
                            .pickerStyle(.segmented)
                            .onChange(of: pendingSettings.blockSize) { _ in
                                scheduleSettingsUpdate()
                            }
                        }
                        
                        VStack {
                            Text("Sample Rate")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Picker("Sample Rate", selection: $pendingSettings.sampleRate) {
                                Text("44.1k").tag(44100)
                                Text("48k").tag(48000)
                                Text("96k").tag(96000)
                            }
                            .pickerStyle(.segmented)
                            .onChange(of: pendingSettings.sampleRate) { _ in
                                scheduleSettingsUpdate()
                            }
                        }
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Device Info Section
    private var deviceInfoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Audio Devices")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(alignment: .leading, spacing: 8) {
                DeviceRow(
                    title: "Input",
                    deviceName: mixerService.audioMetrics.deviceIn ?? "Not connected",
                    isAG06: mixerService.audioMetrics.deviceIn?.contains("AG06") == true
                )
                
                DeviceRow(
                    title: "Output",
                    deviceName: mixerService.audioMetrics.deviceOut ?? "Not connected",
                    isAG06: mixerService.audioMetrics.deviceOut?.contains("AG06") == true
                )
            }
        }
        .padding()
        .background(.regularMaterial)
        .cornerRadius(12)
    }
    
    // MARK: - Computed Properties
    private var subscriptionColor: Color {
        switch configManager.configuration.subscriptionTier {
        case .free: return .gray
        case .pro: return .blue
        case .studio: return .purple
        }
    }
    
    // MARK: - Actions
    private func startMixer() {
        Task {
            if mixerService.audioMetrics.isRunning {
                _ = await mixerService.stopMixer()
            } else {
                let result = await mixerService.startMixer()
                
                if case .failure(let error) = result,
                   case .subscriptionRequired(_) = error {
                    showingSubscription = true
                }
            }
        }
    }
    
    private func refreshStatus() {
        Task {
            await mixerService.refreshStatus()
        }
    }
    
    // MARK: - Settings Management
    private var settingsUpdateTimer: Timer?
    
    private func scheduleSettingsUpdate() {
        settingsUpdateTimer?.invalidate()
        
        settingsUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: false) { _ in
            Task { @MainActor in
                await applySettings()
            }
        }
    }
    
    private func applySettings() async {
        guard pendingSettings != mixerService.mixerSettings else { return }
        
        isApplyingSettings = true
        defer { isApplyingSettings = false }
        
        let result = await mixerService.updateSettings(pendingSettings)
        
        if case .failure(let error) = result {
            // Revert to previous settings on error
            pendingSettings = mixerService.mixerSettings
            
            if case .subscriptionRequired(_) = error {
                showingSubscription = true
            }
        }
    }
    
    // MARK: - Real-time Updates
    private func startRealTimeUpdates() {
        // Real-time updates are handled by MixerService
        // This is where we could add UI-specific real-time effects
    }
    
    private func stopRealTimeUpdates() {
        settingsUpdateTimer?.invalidate()
        settingsUpdateTimer = nil
    }
}

// MARK: - Supporting Views
struct AudioMeter: View {
    let title: String
    let value: Float
    let range: ClosedRange<Float>
    let color: Color
    let unit: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text(String(format: "%.1f%@", value, unit))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(color)
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(.quaternary)
                        .frame(height: 6)
                    
                    // Level indicator
                    Rectangle()
                        .fill(color)
                        .frame(width: max(0, CGFloat((value - range.lowerBound) / (range.upperBound - range.lowerBound)) * geometry.size.width), height: 6)
                }
                .cornerRadius(3)
            }
            .frame(height: 6)
        }
    }
}

struct ControlSlider: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String
    var multiplier: Float = 1
    let onChange: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text(String(format: format, value * multiplier))
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)
            }
            
            Slider(
                value: $value,
                in: range,
                step: (range.upperBound - range.lowerBound) / 100
            ) {
                onChange()
            }
            .tint(.accentColor)
        }
    }
}

struct StatusIndicator: View {
    let title: String
    let isActive: Bool
    let activeText: String
    let inactiveText: String
    var isWarning: Bool = false
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            
            Text(isActive ? activeText : inactiveText)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(
                    isActive ? (isWarning ? .red : .green) : .secondary
                )
        }
    }
}

struct DeviceRow: View {
    let title: String
    let deviceName: String
    let isAG06: Bool
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
                .frame(width: 60, alignment: .leading)
            
            HStack(spacing: 8) {
                if isAG06 {
                    Image(systemName: "waveform.path.ecg")
                        .foregroundColor(.green)
                }
                
                Text(deviceName)
                    .font(.subheadline)
                    .lineLimit(1)
            }
            
            Spacer()
        }
    }
}

struct SubscriptionLockedMeter: View {
    let feature: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(feature)
                    .font(.caption)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("Pro Required")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.orange)
            }
            
            Rectangle()
                .fill(.quaternary)
                .frame(height: 6)
                .cornerRadius(3)
                .overlay(
                    Image(systemName: "lock.fill")
                        .font(.caption2)
                        .foregroundColor(.orange)
                )
        }
    }
}

struct SubscriptionLockedControl: View {
    let feature: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(feature)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("Pro Required")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.orange)
            }
            
            Rectangle()
                .fill(.quaternary)
                .frame(height: 30)
                .cornerRadius(8)
                .overlay(
                    HStack(spacing: 4) {
                        Image(systemName: "lock.fill")
                        Text("Upgrade to unlock")
                            .font(.caption)
                    }
                    .foregroundColor(.orange)
                )
        }
    }
}

#Preview {
    MixerControlView()
        .environmentObject(MixerService(configuration: MixerConfiguration()))
        .environmentObject(ConfigurationManager())
}