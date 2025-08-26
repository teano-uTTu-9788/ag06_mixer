import SwiftUI

@main
struct MobileAG06App: App {
    @StateObject private var configManager = ConfigurationManager()
    @StateObject private var mixerService: MixerService
    @StateObject private var automationService = AutomationService()
    
    init() {
        // Initialize mixer service with configuration
        let config = MixerConfiguration()
        let service = MixerService(configuration: config)
        _mixerService = StateObject(wrappedValue: service)
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(configManager)
                .environmentObject(mixerService)
                .environmentObject(automationService)
                .onAppear {
                    setupInitialConfiguration()
                }
        }
    }
    
    private func setupInitialConfiguration() {
        // Update mixer service with stored configuration
        if let storedConfig = configManager.configuration.mixerConfiguration {
            mixerService.updateConfiguration(storedConfig)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var mixerService: MixerService
    @EnvironmentObject var automationService: AutomationService
    
    var body: some View {
        TabView {
            // AG06 Mixer Tab
            MixerControlView()
                .tabItem {
                    Image(systemName: "slider.horizontal.3")
                    Text("Mixer")
                }
            
            // Automation Dashboard Tab
            DashboardView()
                .tabItem {
                    Image(systemName: "gearshape.2")
                    Text("Automation")
                }
            
            // Notion Integration Tab
            NotionView()
                .tabItem {
                    Image(systemName: "doc.text")
                    Text("Notion")
                }
            
            // Logs Tab
            LogsView()
                .tabItem {
                    Image(systemName: "doc.text.magnifyingglass")
                    Text("Logs")
                }
            
            // Settings Tab
            SettingsView()
                .tabItem {
                    Image(systemName: "gear")
                    Text("Settings")
                }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            mixerService.enterBackgroundMode()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
            mixerService.enterForegroundMode()
        }
    }
}

// MARK: - Enhanced Settings View with Mixer Configuration
struct SettingsView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var mixerService: MixerService
    @EnvironmentObject var automationService: AutomationService
    
    @State private var showingMixerSettings = false
    @State private var showingSubscription = false
    @State private var showingAbout = false
    
    var body: some View {
        NavigationView {
            List {
                // Mixer Configuration Section
                Section {
                    NavigationLink(destination: MixerSettingsView()) {
                        HStack {
                            Image(systemName: "slider.horizontal.3")
                                .foregroundColor(.accentColor)
                                .frame(width: 24)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Mixer Settings")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                
                                Text(mixerConfigurationStatus)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            connectionStatusIndicator
                        }
                    }
                    
                    NavigationLink(destination: SubscriptionView()) {
                        HStack {
                            Image(systemName: "crown")
                                .foregroundColor(.purple)
                                .frame(width: 24)
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Subscription")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                
                                Text(configManager.configuration.subscriptionTier?.displayName ?? "Free")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            if configManager.configuration.subscriptionTier != .studio {
                                Text("Upgrade")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundColor(.accentColor)
                            }
                        }
                    }
                    
                } header: {
                    Text("AG06 Mixer")
                }
                
                // GitHub Configuration Section
                Section {
                    HStack {
                        Image(systemName: "person.crop.circle.badge.plus")
                            .foregroundColor(.blue)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("GitHub Token")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(configManager.configuration.isGitHubConfigured ? "Configured" : "Not set")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Circle()
                            .fill(configManager.configuration.isGitHubConfigured ? .green : .red)
                            .frame(width: 8, height: 8)
                    }
                    
                    HStack {
                        Image(systemName: "folder")
                            .foregroundColor(.orange)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Repository")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(configManager.configuration.githubRepository.isEmpty ? "Not set" : configManager.configuration.githubRepository)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                } header: {
                    Text("GitHub Integration")
                }
                
                // Notion Configuration Section
                Section {
                    HStack {
                        Image(systemName: "doc.text")
                            .foregroundColor(.gray)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Notion Token")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(configManager.configuration.isNotionConfigured ? "Configured" : "Not set")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Circle()
                            .fill(configManager.configuration.isNotionConfigured ? .green : .red)
                            .frame(width: 8, height: 8)
                    }
                    
                    HStack {
                        Image(systemName: "doc.badge.gearshape")
                            .foregroundColor(.purple)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Default Page")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(configManager.configuration.defaultNotionPageId.isEmpty ? "Not set" : "Configured")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                } header: {
                    Text("Notion Integration")
                }
                
                // Performance Section
                Section {
                    HStack {
                        Image(systemName: "speedometer")
                            .foregroundColor(.green)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Performance Mode")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(performanceModeText)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    HStack {
                        Image(systemName: "battery.100")
                            .foregroundColor(.yellow)
                            .frame(width: 24)
                        
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Battery Optimization")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            Text(batteryOptimizationText)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                } header: {
                    Text("Performance")
                }
                
                // About Section
                Section {
                    Button(action: { showingAbout = true }) {
                        HStack {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                                .frame(width: 24)
                            
                            Text("About")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.primary)
                        }
                    }
                    
                    Button("Test Connections") {
                        testAllConnections()
                    }
                    .foregroundColor(.accentColor)
                    
                } header: {
                    Text("Support")
                }
            }
            .navigationTitle("Settings")
        }
        .sheet(isPresented: $showingAbout) {
            AboutView()
        }
    }
    
    // MARK: - Computed Properties
    private var mixerConfigurationStatus: String {
        if let config = configManager.configuration.mixerConfiguration {
            return config.isConfigured ? "Connected to \(config.serverURL)" : "Not configured"
        }
        return "Not configured"
    }
    
    @ViewBuilder
    private var connectionStatusIndicator: some View {
        Circle()
            .fill(mixerService.connectionStatus.isConnected ? .green : .red)
            .frame(width: 8, height: 8)
    }
    
    private var performanceModeText: String {
        let tier = configManager.configuration.subscriptionTier ?? .free
        return tier.batteryOptimization.displayName
    }
    
    private var batteryOptimizationText: String {
        let tier = configManager.configuration.subscriptionTier ?? .free
        let frequency = 1.0 / tier.batteryOptimization.updateInterval
        return String(format: "%.1f Hz updates", frequency)
    }
    
    // MARK: - Actions
    private func testAllConnections() {
        Task {
            // Test mixer connection
            _ = await mixerService.testConnection()
            
            // Test GitHub and Notion connections
            _ = await automationService.testConnections(configuration: configManager.configuration)
        }
    }
}

// MARK: - About View
struct AboutView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // App icon and info
                VStack(spacing: 16) {
                    Image(systemName: "waveform.path.ecg.rectangle")
                        .font(.system(size: 80))
                        .foregroundColor(.accentColor)
                    
                    VStack(spacing: 8) {
                        Text("AG06 Mixer Control")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text("Version 1.0.0")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Description
                Text("Professional audio mixing and automation control for your AG06 hardware. Features real-time processing, AI-powered mixing, and seamless integration with your creative workflow.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                
                Spacer()
                
                // Credits and links
                VStack(spacing: 16) {
                    Text("Built with SwiftUI and powered by the AG06 Mixer API")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    Button("Privacy Policy") {
                        // Open privacy policy
                    }
                    .font(.caption)
                    
                    Button("Terms of Service") {
                        // Open terms of service
                    }
                    .font(.caption)
                }
            }
            .padding()
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
}

// MARK: - Configuration Extensions
extension Configuration {
    var subscriptionTier: SubscriptionTier? {
        get {
            // In real app, get from subscription manager
            return .free
        }
        set {
            // Store subscription tier
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ConfigurationManager())
        .environmentObject(MixerService(configuration: MixerConfiguration()))
        .environmentObject(AutomationService())
}