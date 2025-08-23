import SwiftUI

struct LogsView: View {
    @EnvironmentObject var automationService: AutomationService
    @State private var searchText = ""
    @State private var selectedLevel: LogEntry.Level? = nil
    @State private var showingExportSheet = false
    @State private var autoScroll = true
    @State private var isFollowingLogs = false
    
    private var filteredLogs: [LogEntry] {
        var logs = automationService.logs
        
        // Filter by level
        if let selectedLevel = selectedLevel {
            logs = logs.filter { $0.level == selectedLevel }
        }
        
        // Filter by search text
        if !searchText.isEmpty {
            logs = logs.filter { 
                $0.message.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        return logs.sorted { $0.timestamp > $1.timestamp }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search and Filter Bar
                searchFilterBar
                
                // Logs Content
                logsContent
            }
            .navigationTitle("📋 Logs")
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Menu("Options") {
                        Button(action: toggleAutoScroll) {
                            HStack {
                                Text("Auto Scroll")
                                if autoScroll {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                        
                        Button("Export Logs") {
                            showingExportSheet = true
                        }
                        
                        Divider()
                        
                        Button("Clear All", role: .destructive) {
                            automationService.clearLogs()
                        }
                    }
                    .foregroundColor(.primary)
                }
            }
            .sheet(isPresented: $showingExportSheet) {
                ExportLogsView(logs: filteredLogs)
            }
        }
    }
    
    private var searchFilterBar: some View {
        VStack(spacing: 12) {
            // Search Bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                
                TextField("Search logs...", text: $searchText)
                    .textFieldStyle(.plain)
                
                if !searchText.isEmpty {
                    Button("Clear") {
                        searchText = ""
                    }
                    .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
            .cornerRadius(10)
            
            // Level Filter
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    FilterChip(
                        title: "All",
                        count: automationService.logs.count,
                        isSelected: selectedLevel == nil,
                        color: .gray
                    ) {
                        selectedLevel = nil
                    }
                    
                    ForEach([LogEntry.Level.info, .success, .warning, .error], id: \.self) { level in
                        let count = automationService.logs.filter { $0.level == level }.count
                        
                        FilterChip(
                            title: levelTitle(level),
                            count: count,
                            isSelected: selectedLevel == level,
                            color: Color(level.color)
                        ) {
                            selectedLevel = selectedLevel == level ? nil : level
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .shadow(radius: 1)
    }
    
    private var logsContent: some View {
        Group {
            if filteredLogs.isEmpty {
                emptyLogsView
            } else {
                logsList
            }
        }
    }
    
    private var emptyLogsView: some View {
        VStack(spacing: 16) {
            Image(systemName: searchText.isEmpty ? "doc.text" : "magnifyingglass")
                .font(.system(size: 50))
                .foregroundColor(.secondary)
            
            Text(searchText.isEmpty ? "No logs yet" : "No matching logs")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text(searchText.isEmpty ? 
                 "Automation logs will appear here as tasks are executed" :
                 "Try adjusting your search or filter criteria")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
    
    private var logsList: some View {
        List {
            ForEach(filteredLogs) { entry in
                LogEntryRow(entry: entry)
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            }
        }
        .listStyle(.plain)
        .refreshable {
            // In a real implementation, this might refresh logs from server
        }
    }
    
    private func levelTitle(_ level: LogEntry.Level) -> String {
        switch level {
        case .info: return "Info"
        case .success: return "Success"
        case .warning: return "Warning"
        case .error: return "Error"
        }
    }
    
    private func toggleAutoScroll() {
        autoScroll.toggle()
        isFollowingLogs = autoScroll
    }
}

struct LogEntryRow: View {
    let entry: LogEntry
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                // Level indicator
                Image(systemName: entry.level.icon)
                    .font(.title3)
                    .foregroundColor(Color(entry.level.color))
                    .frame(width: 24)
                
                VStack(alignment: .leading, spacing: 4) {
                    // Message
                    Text(entry.message)
                        .font(.body)
                        .lineLimit(isExpanded ? nil : 3)
                        .animation(.easeInOut, value: isExpanded)
                    
                    // Timestamp
                    Text(entry.timestamp.formatted(date: .omitted, time: .standard))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Expand button (if needed)
                if entry.message.count > 100 {
                    Button(action: { isExpanded.toggle() }) {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            // Expanded details
            if isExpanded && entry.message.count > 100 {
                VStack(alignment: .leading, spacing: 8) {
                    Divider()
                    
                    Text("Full Message:")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)
                    
                    Text(entry.message)
                        .font(.caption)
                        .padding(.leading)
                }
            }
        }
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(entry.level.color).opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(entry.level.color).opacity(0.2), lineWidth: 1)
        )
    }
}

struct FilterChip: View {
    let title: String
    let count: Int
    let isSelected: Bool
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
                
                if count > 0 {
                    Text("\(count)")
                        .font(.caption2)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(isSelected ? Color.white : color.opacity(0.2))
                        )
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(isSelected ? color : Color(.systemGray6))
            )
            .foregroundColor(isSelected ? .white : color)
            .overlay(
                Capsule()
                    .stroke(color.opacity(0.3), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

struct ExportLogsView: View {
    let logs: [LogEntry]
    @Environment(\.presentationMode) var presentationMode
    @State private var selectedFormat: ExportFormat = .text
    @State private var includeTimestamps = true
    @State private var includeLevels = true
    @State private var showingShareSheet = false
    @State private var exportedContent = ""
    
    enum ExportFormat: String, CaseIterable {
        case text = "Plain Text"
        case json = "JSON"
        case csv = "CSV"
        
        var fileExtension: String {
            switch self {
            case .text: return "txt"
            case .json: return "json"
            case .csv: return "csv"
            }
        }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Export Options")
                        .font(.headline)
                    
                    Picker("Format", selection: $selectedFormat) {
                        ForEach(ExportFormat.allCases, id: \.self) { format in
                            Text(format.rawValue).tag(format)
                        }
                    }
                    .pickerStyle(.segmented)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Include Timestamps", isOn: $includeTimestamps)
                        Toggle("Include Log Levels", isOn: $includeLevels)
                    }
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Export Summary")
                        .font(.headline)
                    
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Total Logs")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(logs.count)")
                                .font(.title2)
                                .fontWeight(.bold)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing) {
                            Text("Format")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(selectedFormat.rawValue)
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                }
                
                Spacer()
                
                Button("Export Logs") {
                    exportedContent = generateExportContent()
                    showingShareSheet = true
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                .disabled(logs.isEmpty)
            }
            .padding()
            .navigationTitle("Export Logs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                ShareSheet(
                    activityItems: [exportedContent],
                    fileName: "automation-logs.\(selectedFormat.fileExtension)"
                )
            }
        }
    }
    
    private func generateExportContent() -> String {
        switch selectedFormat {
        case .text:
            return logs.map { entry in
                var components: [String] = []
                
                if includeTimestamps {
                    components.append("[\(entry.timestamp.formatted())]")
                }
                
                if includeLevels {
                    components.append("[\(levelTitle(entry.level))]")
                }
                
                components.append(entry.message)
                
                return components.joined(separator: " ")
            }.joined(separator: "\n")
            
        case .json:
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            
            let exportData = logs.map { entry in
                LogExportData(
                    timestamp: includeTimestamps ? entry.timestamp : nil,
                    level: includeLevels ? levelTitle(entry.level) : nil,
                    message: entry.message
                )
            }
            
            if let data = try? encoder.encode(exportData),
               let string = String(data: data, encoding: .utf8) {
                return string
            }
            return "Export failed"
            
        case .csv:
            var lines = ["Message"]
            
            if includeTimestamps {
                lines.insert("Timestamp", at: 0)
            }
            if includeLevels {
                lines.insert("Level", at: includeTimestamps ? 1 : 0)
            }
            
            let header = lines.joined(separator: ",")
            let rows = logs.map { entry in
                var fields: [String] = []
                
                if includeTimestamps {
                    fields.append("\"\(entry.timestamp.formatted())\"")
                }
                if includeLevels {
                    fields.append("\"\(levelTitle(entry.level))\"")
                }
                fields.append("\"\(entry.message.replacingOccurrences(of: "\"", with: "\"\""))\"")
                
                return fields.joined(separator: ",")
            }
            
            return ([header] + rows).joined(separator: "\n")
        }
    }
    
    private func levelTitle(_ level: LogEntry.Level) -> String {
        switch level {
        case .info: return "INFO"
        case .success: return "SUCCESS"
        case .warning: return "WARNING"
        case .error: return "ERROR"
        }
    }
}

struct LogExportData: Codable {
    let timestamp: Date?
    let level: String?
    let message: String
}

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let fileName: String?
    
    init(activityItems: [Any], fileName: String? = nil) {
        self.activityItems = activityItems
        self.fileName = fileName
    }
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: nil
        )
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    LogsView()
        .environmentObject(AutomationService())
}