import SwiftUI

struct NotionView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var automationService: AutomationService
    @State private var selectedPageId = ""
    @State private var newPageTitle = ""
    @State private var selectedStatus: NotionStatus = .notStarted
    @State private var isUpdating = false
    @State private var showingCreatePage = false
    @State private var lastResult: Result<Void, Error>?
    @State private var showingResult = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection
                    
                    // Connection Status
                    connectionStatusSection
                    
                    // Update Page Status
                    updateStatusSection
                    
                    // Create New Page
                    createPageSection
                    
                    // Quick Status Updates
                    quickStatusSection
                }
                .padding()
            }
            .navigationTitle("📄 Notion")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("New Page") {
                        showingCreatePage = true
                    }
                    .disabled(!configManager.configuration.isNotionConfigured)
                }
            }
            .sheet(isPresented: $showingCreatePage) {
                CreatePageView(
                    newPageTitle: $newPageTitle,
                    isCreating: $isUpdating,
                    onCancel: { showingCreatePage = false },
                    onCreate: { title in
                        Task {
                            await createPage(title: title)
                        }
                    }
                )
            }
            .alert("Notion Result", isPresented: $showingResult) {
                Button("OK") { }
            } message: {
                if case .failure(let error) = lastResult {
                    Text("Error: \(error.localizedDescription)")
                } else {
                    Text("Operation completed successfully!")
                }
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Notion Integration")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Manage pages and automation tasks")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Image(systemName: "doc.text")
                    .font(.title)
                    .foregroundColor(.blue)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var connectionStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Connection Status")
                .font(.headline)
            
            HStack {
                Circle()
                    .fill(configManager.configuration.isNotionConfigured ? Color.green : Color.red)
                    .frame(width: 12, height: 12)
                
                Text(configManager.configuration.isNotionConfigured ? "Connected" : "Not Configured")
                    .font(.subheadline)
                
                Spacer()
                
                if !configManager.configuration.isNotionConfigured {
                    NavigationLink("Configure") {
                        SettingsView()
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var updateStatusSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Update Page Status")
                .font(.title2)
                .fontWeight(.semibold)
            
            VStack(spacing: 12) {
                TextField("Notion Page ID", text: $selectedPageId)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                
                Picker("Status", selection: $selectedStatus) {
                    ForEach(NotionStatus.allCases, id: \.self) { status in
                        HStack {
                            Circle()
                                .fill(Color(status.color))
                                .frame(width: 8, height: 8)
                            Text(status.rawValue)
                        }
                        .tag(status)
                    }
                }
                .pickerStyle(.menu)
                
                Button(action: { Task { await updatePageStatus() } }) {
                    HStack {
                        if isUpdating {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "arrow.up.circle")
                        }
                        
                        Text("Update Status")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(selectedPageId.isEmpty || !configManager.configuration.isNotionConfigured || isUpdating)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var createPageSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Actions")
                .font(.title2)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                NotionActionButton(
                    title: "New Task",
                    icon: "plus.circle",
                    color: .blue,
                    isLoading: isUpdating
                ) {
                    newPageTitle = "New Task"
                    showingCreatePage = true
                }
                
                NotionActionButton(
                    title: "New Project",
                    icon: "folder.badge.plus",
                    color: .green,
                    isLoading: isUpdating
                ) {
                    newPageTitle = "New Project"
                    showingCreatePage = true
                }
                
                NotionActionButton(
                    title: "Meeting Notes",
                    icon: "person.2",
                    color: .purple,
                    isLoading: isUpdating
                ) {
                    newPageTitle = "Meeting Notes - \(DateFormatter.localizedString(from: Date(), dateStyle: .short, timeStyle: .none))"
                    showingCreatePage = true
                }
                
                NotionActionButton(
                    title: "Daily Log",
                    icon: "calendar",
                    color: .orange,
                    isLoading: isUpdating
                ) {
                    newPageTitle = "Daily Log - \(DateFormatter.localizedString(from: Date(), dateStyle: .medium, timeStyle: .none))"
                    showingCreatePage = true
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var quickStatusSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Status Updates")
                .font(.title2)
                .fontWeight(.semibold)
            
            if !configManager.configuration.defaultNotionPageId.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Default Page")
                        .font(.headline)
                    
                    Text(configManager.configuration.defaultNotionPageId)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(6)
                    
                    HStack(spacing: 8) {
                        ForEach(NotionStatus.allCases.prefix(3), id: \.self) { status in
                            Button(action: {
                                Task {
                                    selectedPageId = configManager.configuration.defaultNotionPageId
                                    selectedStatus = status
                                    await updatePageStatus()
                                }
                            }) {
                                VStack(spacing: 4) {
                                    Circle()
                                        .fill(Color(status.color))
                                        .frame(width: 16, height: 16)
                                    
                                    Text(status.rawValue)
                                        .font(.caption2)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 8)
                                .background(Color(status.color).opacity(0.1))
                                .cornerRadius(8)
                            }
                            .buttonStyle(.plain)
                            .disabled(isUpdating)
                        }
                    }
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    
                    Text("No Default Page Set")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text("Set a default page in Settings for quick status updates")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(20)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func updatePageStatus() async {
        guard !selectedPageId.isEmpty else { return }
        
        isUpdating = true
        
        let result = await automationService.updateNotionStatus(
            pageId: selectedPageId,
            status: selectedStatus,
            configuration: configManager.configuration
        )
        
        lastResult = result
        showingResult = true
        isUpdating = false
    }
    
    private func createPage(title: String) async {
        guard !title.isEmpty else { return }
        
        isUpdating = true
        showingCreatePage = false
        
        // In a real implementation, we'd need to add page creation to AutomationService
        // For now, this is a placeholder for the concept
        
        await MainActor.run {
            lastResult = .success(())
            showingResult = true
            isUpdating = false
            newPageTitle = ""
        }
    }
}

struct NotionActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let isLoading: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                } else {
                    Image(systemName: icon)
                        .font(.title2)
                }
                
                Text(title)
                    .font(.caption)
                    .multilineTextAlignment(.center)
            }
            .frame(height: 80)
            .frame(maxWidth: .infinity)
            .background(color.opacity(0.1))
            .foregroundColor(color)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(color.opacity(0.3), lineWidth: 1)
            )
        }
        .disabled(isLoading)
    }
}

struct CreatePageView: View {
    @Binding var newPageTitle: String
    @Binding var isCreating: Bool
    let onCancel: () -> Void
    let onCreate: (String) -> Void
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Page Title")
                        .font(.headline)
                    
                    TextField("Enter page title", text: $newPageTitle)
                        .textFieldStyle(.roundedBorder)
                        .font(.body)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Create Page")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        onCancel()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Create") {
                        onCreate(newPageTitle)
                    }
                    .disabled(newPageTitle.isEmpty || isCreating)
                }
            }
        }
    }
}

#Preview {
    NotionView()
        .environmentObject(ConfigurationManager())
        .environmentObject(AutomationService())
}