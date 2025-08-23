import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var configManager: ConfigurationManager
    @EnvironmentObject var automationService: AutomationService
    @State private var selectedTask: AutomationTask = .doctor
    @State private var notionPageId = ""
    @State private var showingTaskResult = false
    @State private var lastTaskResult: Result<Void, Error>?
    @State private var connectionStatus: (github: Bool, notion: Bool) = (false, false)
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    headerSection
                    
                    // Status Cards
                    statusCardsSection
                    
                    // Quick Actions
                    quickActionsSection
                    
                    // Recent Runs
                    recentRunsSection
                }
                .padding()
            }
            .navigationTitle("🚀 Dashboard")
            .refreshable {
                await refreshStatus()
            }
        }
        .alert("Task Result", isPresented: $showingTaskResult) {
            Button("OK") { }
        } message: {
            if case .failure(let error) = lastTaskResult {
                Text("Error: \(error.localizedDescription)")
            } else {
                Text("Task executed successfully!")
            }
        }
        .task {
            await refreshStatus()
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading) {
                    Text("AiCan Automation")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Remote macOS Terminal Control")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: { Task { await refreshStatus() } }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.title2)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var statusCardsSection: some View {
        HStack(spacing: 16) {
            StatusCard(
                title: "GitHub",
                status: connectionStatus.github ? "Connected" : "Not Connected",
                isConnected: connectionStatus.github,
                icon: "github"
            )
            
            StatusCard(
                title: "Notion",
                status: connectionStatus.notion ? "Connected" : "Not Connected",
                isConnected: connectionStatus.notion,
                icon: "doc.text"
            )
        }
    }
    
    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick Actions")
                .font(.title2)
                .fontWeight(.semibold)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(AutomationTask.allCases, id: \.self) { task in
                    TaskButton(
                        task: task,
                        isLoading: automationService.isLoading && selectedTask == task
                    ) {
                        selectedTask = task
                        await executeTask(task)
                    }
                }
            }
            
            // Notion Page ID input
            VStack(alignment: .leading, spacing: 8) {
                Text("Notion Page ID (Optional)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                TextField("Enter Notion page ID", text: $notionPageId)
                    .textFieldStyle(.roundedBorder)
            }
            .padding(.top)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var recentRunsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recent Runs")
                .font(.title2)
                .fontWeight(.semibold)
            
            if automationService.workflowRuns.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    
                    Text("No recent runs")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text("Execute a task to see it here")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(40)
            } else {
                ForEach(automationService.workflowRuns.suffix(5).reversed(), id: \.id) { run in
                    WorkflowRunRow(run: run)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private func executeTask(_ task: AutomationTask) async {
        guard configManager.configuration.isGitHubConfigured else {
            lastTaskResult = .failure(AutomationError.missingConfiguration("GitHub not configured"))
            showingTaskResult = true
            return
        }
        
        let pageId = notionPageId.isEmpty ? nil : notionPageId
        let result = await automationService.triggerWorkflow(
            task: task,
            notionPageId: pageId,
            configuration: configManager.configuration
        )
        
        lastTaskResult = result
        showingTaskResult = true
    }
    
    private func refreshStatus() async {
        connectionStatus = await automationService.testConnections(
            configuration: configManager.configuration
        )
    }
}

struct StatusCard: View {
    let title: String
    let status: String
    let isConnected: Bool
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(isConnected ? .green : .red)
            
            VStack(alignment: .leading) {
                Text(title)
                    .font(.headline)
                
                Text(status)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Circle()
                .fill(isConnected ? Color.green : Color.red)
                .frame(width: 12, height: 12)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 1)
    }
}

struct TaskButton: View {
    let task: AutomationTask
    let isLoading: Bool
    let action: () async -> Void
    
    var body: some View {
        Button(action: { Task { await action() } }) {
            VStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                } else {
                    Image(systemName: task.icon)
                        .font(.title2)
                }
                
                Text(task.displayName)
                    .font(.caption)
                    .multilineTextAlignment(.center)
            }
            .frame(height: 80)
            .frame(maxWidth: .infinity)
            .background(Color(task.color).opacity(0.1))
            .foregroundColor(Color(task.color))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color(task.color).opacity(0.3), lineWidth: 1)
            )
        }
        .disabled(isLoading)
    }
}

struct WorkflowRunRow: View {
    let run: WorkflowRun
    
    var body: some View {
        HStack {
            Image(systemName: run.task.icon)
                .foregroundColor(Color(run.task.color))
            
            VStack(alignment: .leading) {
                Text(run.task.displayName)
                    .font(.headline)
                
                Text(run.timestamp.formatted(date: .omitted, time: .shortened))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            StatusBadge(status: run.status)
        }
        .padding(.vertical, 8)
    }
}

struct StatusBadge: View {
    let status: WorkflowRun.RunStatus
    
    var body: some View {
        Text(statusText)
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(statusColor.opacity(0.2))
            .foregroundColor(statusColor)
            .cornerRadius(8)
    }
    
    private var statusText: String {
        switch status {
        case .queued: return "Queued"
        case .inProgress: return "Running"
        case .completed: return "Done"
        case .failed: return "Failed"
        }
    }
    
    private var statusColor: Color {
        switch status {
        case .queued: return .orange
        case .inProgress: return .blue
        case .completed: return .green
        case .failed: return .red
        }
    }
}

#Preview {
    DashboardView()
        .environmentObject(ConfigurationManager())
        .environmentObject(AutomationService())
}