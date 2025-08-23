import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @State private var showingGitHubConfig = false
    @State private var showingNotionConfig = false
    @State private var showingValidation = false
    @State private var validationErrors: [String] = []
    @State private var showingResetAlert = false
    @State private var testingConnections = false
    @State private var connectionResults: (github: Bool, notion: Bool) = (false, false)
    @State private var showingConnectionResults = false
    
    var body: some View {
        NavigationView {
            List {
                // App Info Section
                appInfoSection
                
                // Configuration Sections
                gitHubSection
                notionSection
                
                // Actions Section
                actionsSection
                
                // Advanced Section
                advancedSection
                
                // About Section
                aboutSection
            }
            .navigationTitle("⚙️ Settings")
            .sheet(isPresented: $showingGitHubConfig) {
                GitHubConfigView()
            }
            .sheet(isPresented: $showingNotionConfig) {
                NotionConfigView()
            }
            .alert("Configuration Validation", isPresented: $showingValidation) {
                Button("OK") { }
            } message: {
                if validationErrors.isEmpty {
                    Text("✅ Configuration is valid!")
                } else {
                    Text("❌ Issues found:\n\(validationErrors.joined(separator: "\n"))")
                }
            }
            .alert("Reset Configuration", isPresented: $showingResetAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Reset", role: .destructive) {
                    configManager.clearConfiguration()
                }
            } message: {
                Text("This will delete all your configuration settings. This action cannot be undone.")
            }
            .alert("Connection Test Results", isPresented: $showingConnectionResults) {
                Button("OK") { }
            } message: {
                Text("GitHub: \(connectionResults.github ? "✅ Connected" : "❌ Failed")\nNotion: \(connectionResults.notion ? "✅ Connected" : "❌ Failed")")
            }
        }
    }
    
    private var appInfoSection: some View {
        Section {
            HStack {
                Image(systemName: "iphone")
                    .font(.title2)
                    .foregroundColor(.blue)
                    .frame(width: 32)
                
                VStack(alignment: .leading) {
                    Text("AiCan Automation")
                        .font(.headline)
                    Text("Remote macOS Terminal Control")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Text("v1.0.0")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.vertical, 4)
        }
    }
    
    private var gitHubSection: some View {
        Section("GitHub Configuration") {
            ConfigurationRow(
                icon: "github",
                title: "GitHub Integration",
                subtitle: configManager.configuration.isGitHubConfigured ?
                    configManager.configuration.githubRepository : "Not configured",
                isConfigured: configManager.configuration.isGitHubConfigured,
                action: { showingGitHubConfig = true }
            )
            
            if configManager.configuration.isGitHubConfigured {
                HStack {
                    Image(systemName: "person")
                        .foregroundColor(.secondary)
                        .frame(width: 24)
                    
                    VStack(alignment: .leading) {
                        Text("Repository")
                        Text(configManager.configuration.repositoryOwner)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text(configManager.configuration.repositoryName)
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
        }
    }
    
    private var notionSection: some View {
        Section("Notion Configuration") {
            ConfigurationRow(
                icon: "doc.text",
                title: "Notion Integration",
                subtitle: configManager.configuration.isNotionConfigured ?
                    "Connected" : "Not configured",
                isConfigured: configManager.configuration.isNotionConfigured,
                action: { showingNotionConfig = true }
            )
            
            if configManager.configuration.isNotionConfigured && 
               !configManager.configuration.defaultNotionPageId.isEmpty {
                HStack {
                    Image(systemName: "doc.badge.gearshape")
                        .foregroundColor(.secondary)
                        .frame(width: 24)
                    
                    VStack(alignment: .leading) {
                        Text("Default Page")
                        Text(configManager.configuration.defaultNotionPageId)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    
                    Spacer()
                }
            }
        }
    }
    
    private var actionsSection: some View {
        Section("Actions") {
            Button(action: validateConfiguration) {
                HStack {
                    Image(systemName: "checkmark.shield")
                        .foregroundColor(.blue)
                        .frame(width: 24)
                    
                    Text("Validate Configuration")
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Button(action: testConnections) {
                HStack {
                    Image(systemName: testingConnections ? "arrow.clockwise" : "network")
                        .foregroundColor(.green)
                        .frame(width: 24)
                        .rotationEffect(.degrees(testingConnections ? 360 : 0))
                        .animation(testingConnections ? .linear(duration: 1).repeatForever(autoreverses: false) : .default, value: testingConnections)
                    
                    Text("Test Connections")
                    
                    Spacer()
                    
                    if testingConnections {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "chevron.right")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .disabled(testingConnections || (!configManager.configuration.isGitHubConfigured && !configManager.configuration.isNotionConfigured))
        }
    }
    
    private var advancedSection: some View {
        Section("Advanced") {
            NavigationLink(destination: AdvancedSettingsView()) {
                HStack {
                    Image(systemName: "gearshape.2")
                        .foregroundColor(.gray)
                        .frame(width: 24)
                    
                    Text("Advanced Settings")
                }
            }
            
            Button(action: { showingResetAlert = true }) {
                HStack {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                        .frame(width: 24)
                    
                    Text("Reset Configuration")
                        .foregroundColor(.red)
                    
                    Spacer()
                }
            }
        }
    }
    
    private var aboutSection: some View {
        Section("About") {
            HStack {
                Image(systemName: "info.circle")
                    .foregroundColor(.blue)
                    .frame(width: 24)
                
                Text("Version")
                
                Spacer()
                
                Text("1.0.0")
                    .foregroundColor(.secondary)
            }
            
            HStack {
                Image(systemName: "hammer")
                    .foregroundColor(.orange)
                    .frame(width: 24)
                
                Text("Build")
                
                Spacer()
                
                Text("1")
                    .foregroundColor(.secondary)
            }
            
            Link(destination: URL(string: "https://github.com")!) {
                HStack {
                    Image(systemName: "link")
                        .foregroundColor(.blue)
                        .frame(width: 24)
                    
                    Text("GitHub Repository")
                    
                    Spacer()
                    
                    Image(systemName: "arrow.up.right")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
        }
    }
    
    private func validateConfiguration() {
        validationErrors = configManager.validateConfiguration()
        showingValidation = true
    }
    
    private func testConnections() {
        guard !testingConnections else { return }
        
        testingConnections = true
        
        Task {
            let results = await AutomationService().testConnections(
                configuration: configManager.configuration
            )
            
            await MainActor.run {
                connectionResults = results
                showingConnectionResults = true
                testingConnections = false
            }
        }
    }
}

struct ConfigurationRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let isConfigured: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(isConfigured ? .green : .orange)
                    .frame(width: 24)
                
                VStack(alignment: .leading) {
                    Text(title)
                        .foregroundColor(.primary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .buttonStyle(.plain)
    }
}

struct GitHubConfigView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @Environment(\.presentationMode) var presentationMode
    @State private var githubToken = ""
    @State private var githubRepository = ""
    @State private var showingTokenHelp = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("GitHub Personal Access Token")) {
                    SecureField("ghp_xxxxxxxxxxxxxxxxxxxx", text: $githubToken)
                        .textContentType(.password)
                    
                    Button("How to create a token?") {
                        showingTokenHelp = true
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
                
                Section(header: Text("Repository"), footer: Text("Format: owner/repository-name")) {
                    TextField("username/repo-name", text: $githubRepository)
                        .textContentType(.none)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
                
                Section {
                    Button("Save Configuration") {
                        configManager.updateGitHubSettings(
                            token: githubToken,
                            repository: githubRepository
                        )
                        presentationMode.wrappedValue.dismiss()
                    }
                    .disabled(githubToken.isEmpty || githubRepository.isEmpty)
                }
            }
            .navigationTitle("GitHub Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
            .onAppear {
                githubToken = configManager.configuration.githubToken
                githubRepository = configManager.configuration.githubRepository
            }
            .alert("GitHub Token Setup", isPresented: $showingTokenHelp) {
                Button("OK") { }
            } message: {
                Text("1. Go to GitHub Settings → Developer settings → Personal access tokens\n2. Click 'Generate new token'\n3. Select scopes: 'repo' and 'workflow'\n4. Copy the token and paste it here")
            }
        }
    }
}

struct NotionConfigView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @Environment(\.presentationMode) var presentationMode
    @State private var notionToken = ""
    @State private var defaultPageId = ""
    @State private var showingTokenHelp = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Notion Integration Token")) {
                    SecureField("secret_xxxxxxxxxxxxxxxxxxxx", text: $notionToken)
                        .textContentType(.password)
                    
                    Button("How to create an integration?") {
                        showingTokenHelp = true
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
                
                Section(header: Text("Default Page ID (Optional)"), footer: Text("Page ID for quick status updates")) {
                    TextField("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", text: $defaultPageId)
                        .textContentType(.none)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
                
                Section {
                    Button("Save Configuration") {
                        configManager.updateNotionSettings(
                            token: notionToken,
                            defaultPageId: defaultPageId
                        )
                        presentationMode.wrappedValue.dismiss()
                    }
                    .disabled(notionToken.isEmpty)
                }
            }
            .navigationTitle("Notion Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
            .onAppear {
                notionToken = configManager.configuration.notionToken
                defaultPageId = configManager.configuration.defaultNotionPageId
            }
            .alert("Notion Integration Setup", isPresented: $showingTokenHelp) {
                Button("OK") { }
            } message: {
                Text("1. Go to notion.so/my-integrations\n2. Create a new integration\n3. Copy the 'Internal Integration Token'\n4. Share your database/page with the integration\n5. Paste the token here")
            }
        }
    }
}

struct AdvancedSettingsView: View {
    @AppStorage("enableDebugMode") private var enableDebugMode = false
    @AppStorage("maxLogEntries") private var maxLogEntries = 100
    @AppStorage("autoRefreshInterval") private var autoRefreshInterval = 30
    @AppStorage("enableNotifications") private var enableNotifications = true
    
    var body: some View {
        Form {
            Section("Debug") {
                Toggle("Debug Mode", isOn: $enableDebugMode)
            }
            
            Section(header: Text("Logging"), footer: Text("Maximum number of log entries to keep in memory")) {
                Stepper("Max Log Entries: \(maxLogEntries)", value: $maxLogEntries, in: 50...500, step: 50)
            }
            
            Section(header: Text("Auto Refresh"), footer: Text("Seconds between automatic status updates")) {
                Stepper("Interval: \(autoRefreshInterval)s", value: $autoRefreshInterval, in: 10...300, step: 10)
            }
            
            Section("Notifications") {
                Toggle("Enable Notifications", isOn: $enableNotifications)
            }
        }
        .navigationTitle("Advanced")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    SettingsView()
        .environmentObject(ConfigurationManager())
}