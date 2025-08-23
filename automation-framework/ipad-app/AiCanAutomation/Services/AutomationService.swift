import Foundation
import Combine

class AutomationService: ObservableObject {
    @Published var workflowRuns: [WorkflowRun] = []
    @Published var logs: [LogEntry] = []
    @Published var isLoading = false
    
    private var cancellables = Set<AnyCancellable>()
    
    func triggerWorkflow(
        task: AutomationTask,
        notionPageId: String? = nil,
        configuration: Configuration
    ) async -> Result<Void, Error> {
        guard configuration.isGitHubConfigured else {
            return .failure(AutomationError.missingConfiguration("GitHub not configured"))
        }
        
        let workflowRun = WorkflowRun(
            task: task,
            timestamp: Date(),
            status: .queued,
            notionPageId: notionPageId
        )
        
        await MainActor.run {
            workflowRuns.append(workflowRun)
            log(.info, "Triggering \(task.displayName)...")
            isLoading = true
        }
        
        defer {
            Task { @MainActor in
                isLoading = false
            }
        }
        
        do {
            try await triggerGitHubWorkflow(
                task: task,
                notionPageId: notionPageId,
                configuration: configuration
            )
            
            await MainActor.run {
                log(.success, "\(task.displayName) triggered successfully")
                updateWorkflowStatus(workflowRun.id, status: .inProgress)
            }
            
            return .success(())
            
        } catch {
            await MainActor.run {
                log(.error, "Failed to trigger \(task.displayName): \(error.localizedDescription)")
                updateWorkflowStatus(workflowRun.id, status: .failed)
            }
            
            return .failure(error)
        }
    }
    
    func updateNotionStatus(
        pageId: String,
        status: NotionStatus,
        configuration: Configuration
    ) async -> Result<Void, Error> {
        guard configuration.isNotionConfigured else {
            return .failure(AutomationError.missingConfiguration("Notion not configured"))
        }
        
        await MainActor.run {
            log(.info, "Updating Notion page status to \(status.rawValue)...")
            isLoading = true
        }
        
        defer {
            Task { @MainActor in
                isLoading = false
            }
        }
        
        do {
            try await updateNotionPage(
                pageId: pageId,
                status: status,
                configuration: configuration
            )
            
            await MainActor.run {
                log(.success, "Notion page updated successfully")
            }
            
            return .success(())
            
        } catch {
            await MainActor.run {
                log(.error, "Failed to update Notion: \(error.localizedDescription)")
            }
            
            return .failure(error)
        }
    }
    
    private func triggerGitHubWorkflow(
        task: AutomationTask,
        notionPageId: String?,
        configuration: Configuration
    ) async throws {
        let url = URL(string: "https://api.github.com/repos/\(configuration.repositoryOwner)/\(configuration.repositoryName)/actions/workflows/dispatch.yml/dispatches")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("token \(configuration.githubToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/vnd.github.v3+json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        var inputs: [String: String] = ["task": task.rawValue]
        if let pageId = notionPageId, !pageId.isEmpty {
            inputs["notion_page"] = pageId
        }
        
        let payload: [String: Any] = [
            "ref": "main",
            "inputs": inputs
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AutomationError.invalidResponse
        }
        
        guard httpResponse.statusCode == 204 else {
            throw AutomationError.httpError(httpResponse.statusCode)
        }
    }
    
    private func updateNotionPage(
        pageId: String,
        status: NotionStatus,
        configuration: Configuration
    ) async throws {
        let url = URL(string: "https://api.notion.com/v1/pages/\(pageId)")!
        
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("Bearer \(configuration.notionToken)", forHTTPHeaderField: "Authorization")
        request.setValue("2022-06-28", forHTTPHeaderField: "Notion-Version")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let payload: [String: Any] = [
            "properties": [
                "Status": [
                    "status": [
                        "name": status.rawValue
                    ]
                ]
            ]
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AutomationError.invalidResponse
        }
        
        guard 200...299 ~= httpResponse.statusCode else {
            throw AutomationError.httpError(httpResponse.statusCode)
        }
    }
    
    func testConnections(configuration: Configuration) async -> (github: Bool, notion: Bool) {
        var githubOk = false
        var notionOk = false
        
        // Test GitHub API
        if configuration.isGitHubConfigured {
            do {
                let url = URL(string: "https://api.github.com/repos/\(configuration.repositoryOwner)/\(configuration.repositoryName)")!
                var request = URLRequest(url: url)
                request.setValue("token \(configuration.githubToken)", forHTTPHeaderField: "Authorization")
                
                let (_, response) = try await URLSession.shared.data(for: request)
                githubOk = (response as? HTTPURLResponse)?.statusCode == 200
            } catch {
                githubOk = false
            }
        }
        
        // Test Notion API
        if configuration.isNotionConfigured {
            do {
                let url = URL(string: "https://api.notion.com/v1/users/me")!
                var request = URLRequest(url: url)
                request.setValue("Bearer \(configuration.notionToken)", forHTTPHeaderField: "Authorization")
                request.setValue("2022-06-28", forHTTPHeaderField: "Notion-Version")
                
                let (_, response) = try await URLSession.shared.data(for: request)
                notionOk = (response as? HTTPURLResponse)?.statusCode == 200
            } catch {
                notionOk = false
            }
        }
        
        await MainActor.run {
            log(.info, "Connection test - GitHub: \(githubOk ? "✅" : "❌"), Notion: \(notionOk ? "✅" : "❌")")
        }
        
        return (githubOk, notionOk)
    }
    
    private func log(_ level: LogEntry.Level, _ message: String) {
        let entry = LogEntry(timestamp: Date(), level: level, message: message)
        logs.append(entry)
        
        // Keep only last 100 log entries
        if logs.count > 100 {
            logs.removeFirst(logs.count - 100)
        }
    }
    
    private func updateWorkflowStatus(_ id: UUID, status: WorkflowRun.RunStatus) {
        if let index = workflowRuns.firstIndex(where: { $0.id == id }) {
            // Note: In a real implementation, we'd need to make WorkflowRun mutable
            // For now, this is a placeholder for the concept
        }
    }
    
    func clearLogs() {
        logs.removeAll()
        log(.info, "Logs cleared")
    }
}

enum AutomationError: Error, LocalizedError {
    case missingConfiguration(String)
    case invalidResponse
    case httpError(Int)
    
    var errorDescription: String? {
        switch self {
        case .missingConfiguration(let message):
            return message
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "HTTP error: \(code)"
        }
    }
}