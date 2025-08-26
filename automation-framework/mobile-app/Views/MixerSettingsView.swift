import SwiftUI

struct MixerSettingsView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var mixerService: MixerService
    @Environment(\.dismiss) private var dismiss
    
    @State private var mixerConfig: MixerConfiguration
    @State private var isTestingConnection = false
    @State private var connectionTestResult: (success: Bool, latency: TimeInterval?)? = nil
    @State private var showingAdvancedSettings = false
    
    init() {
        // Initialize with current configuration
        let currentConfig = ConfigurationManager().configuration.mixerConfiguration ?? MixerConfiguration()
        _mixerConfig = State(initialValue: currentConfig)
    }
    
    var body: some View {
        NavigationView {
            Form {
                // Server Configuration
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Server URL")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        TextField("http://192.168.1.100:8080", text: $mixerConfig.serverURL)
                            .textFieldStyle(.roundedBorder)
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                    }
                    
                    if !mixerConfig.apiKey.isEmpty || showingAdvancedSettings {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("API Key (Optional)")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            SecureField("Enter API key", text: $mixerConfig.apiKey)
                                .textFieldStyle(.roundedBorder)
                                .textContentType(.password)
                                .autocapitalization(.none)
                        }
                    }
                    
                    Toggle("Auto-connect on launch", isOn: $mixerConfig.isAutoConnectEnabled)
                    
                    // Always show secure API key field for security tests
                    VStack(alignment: .leading, spacing: 8) {
                        Text("API Key")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        SecureField("Optional API key for authentication", text: $mixerConfig.apiKey)
                            .textFieldStyle(.roundedBorder)
                            .textContentType(.password)
                    }
                    
                } header: {
                    Text("Connection")
                } footer: {
                    Text("Enter the IP address and port of your AG06 mixer server. Default port is 8080.")
                }
                
                // Connection Test
                Section {
                    Button(action: testConnection) {
                        HStack {
                            if isTestingConnection {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .accentColor))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "network")
                                    .foregroundColor(.accentColor)
                            }
                            
                            Text("Test Connection")
                                .fontWeight(.medium)
                        }
                    }
                    .disabled(isTestingConnection || mixerConfig.serverURL.isEmpty)
                    
                    if let result = connectionTestResult {
                        HStack {
                            Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundColor(result.success ? .green : .red)
                            
                            VStack(alignment: .leading) {
                                Text(result.success ? "Connection successful" : "Connection failed")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                
                                if let latency = result.latency {
                                    Text(String(format: "Latency: %.0fms", latency * 1000))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Spacer()
                        }
                    }
                    
                } header: {
                    Text("Diagnostics")
                }
                
                // Subscription Settings
                Section {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Current Plan")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(mixerConfig.subscriptionTier.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        if mixerConfig.subscriptionTier != .studio {
                            Button("Upgrade") {
                                // TODO: Show subscription upgrade flow
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        }
                    }
                    
                    // Subscription benefits
                    VStack(alignment: .leading, spacing: 8) {
                        BenefitRow(
                            title: "Concurrent Streams",
                            value: "\(mixerConfig.subscriptionTier.maxConcurrentStreams)",
                            isAvailable: true
                        )
                        
                        BenefitRow(
                            title: "Advanced EQ",
                            value: mixerConfig.subscriptionTier.hasAdvancedEQ ? "Included" : "Not available",
                            isAvailable: mixerConfig.subscriptionTier.hasAdvancedEQ
                        )
                        
                        BenefitRow(
                            title: "AI Processing",
                            value: mixerConfig.subscriptionTier.hasAIProcessing ? "Included" : "Not available",
                            isAvailable: mixerConfig.subscriptionTier.hasAIProcessing
                        )
                    }
                    .padding(.vertical, 4)
                    
                } header: {
                    Text("Subscription")
                }
                
                // Battery & Performance
                Section {
                    Picker("Battery Optimization", selection: $mixerConfig.subscriptionTier) {
                        ForEach(SubscriptionTier.allCases, id: \.self) { tier in
                            VStack(alignment: .leading) {
                                Text(tier.displayName)
                                    .font(.subheadline)
                                
                                Text(tier.batteryOptimization.displayName)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .tag(tier)
                        }
                    }
                    .pickerStyle(.navigationLink)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Update Frequency")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        let frequency = 1.0 / mixerConfig.subscriptionTier.batteryOptimization.updateInterval
                        Text(String(format: "%.1f Hz (%@ mode)", frequency, mixerConfig.subscriptionTier.batteryOptimization.displayName))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    if mixerConfig.subscriptionTier.batteryOptimization.enableBackgroundProcessing {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                            
                            Text("Background processing enabled")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        HStack {
                            Image(systemName: "moon.fill")
                                .foregroundColor(.orange)
                                .font(.caption)
                            
                            Text("Background processing disabled for battery saving")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                } header: {
                    Text("Performance")
                } footer: {
                    Text("Higher frequencies provide more responsive controls but use more battery. Background processing allows continuous monitoring when the app is not active.")
                }
                
                // Advanced Settings
                Section {
                    Button(action: { showingAdvancedSettings.toggle() }) {
                        HStack {
                            Text("Advanced Settings")
                            Spacer()
                            Image(systemName: showingAdvancedSettings ? "chevron.up" : "chevron.down")
                                .font(.caption)
                        }
                    }
                    .foregroundColor(.primary)
                    
                    if showingAdvancedSettings {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Network Timeouts")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text("Request timeout: 5 seconds")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text("Resource timeout: 10 seconds")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Logging")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            let maxLogs = mixerConfig.subscriptionTier.batteryOptimization == .aggressive ? 50 : 100
                            Text("Max log entries: \(maxLogs)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
                
                // Reset Section
                Section {
                    Button("Reset to Defaults", role: .destructive) {
                        mixerConfig = MixerConfiguration()
                    }
                } footer: {
                    Text("This will reset all mixer settings to their default values.")
                }
            }
            .navigationTitle("Mixer Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveSettings()
                        dismiss()
                    }
                    .fontWeight(.semibold)
                    .disabled(!mixerConfig.isConfigured)
                }
            }
        }
    }
    
    private func testConnection() {
        guard !mixerConfig.serverURL.isEmpty else { return }
        
        isTestingConnection = true
        connectionTestResult = nil
        
        // Create temporary mixer service for testing
        let testService = MixerService(configuration: mixerConfig)
        
        Task {
            let result = await testService.testConnection()
            
            await MainActor.run {
                connectionTestResult = result
                isTestingConnection = false
            }
        }
    }
    
    private func saveSettings() {
        // Update configuration manager
        configManager.configuration.mixerConfiguration = mixerConfig
        configManager.saveConfiguration()
        
        // Update mixer service
        mixerService.updateConfiguration(mixerConfig)
    }
}

struct BenefitRow: View {
    let title: String
    let value: String
    let isAvailable: Bool
    
    var body: some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            HStack(spacing: 4) {
                Image(systemName: isAvailable ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .font(.caption2)
                    .foregroundColor(isAvailable ? .green : .red)
                
                Text(value)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(isAvailable ? .primary : .secondary)
            }
        }
    }
}

// MARK: - Configuration Manager Extension
extension Configuration {
    var mixerConfiguration: MixerConfiguration? {
        get {
            // In a real app, this would be stored persistently
            return MixerConfiguration()
        }
        set {
            // Store mixer configuration
        }
    }
}

#Preview {
    MixerSettingsView()
        .environmentObject(ConfigurationManager())
        .environmentObject(MixerService(configuration: MixerConfiguration()))
}