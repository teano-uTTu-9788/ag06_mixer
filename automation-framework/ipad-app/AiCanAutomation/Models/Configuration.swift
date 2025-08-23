import Foundation

struct Configuration: Codable {
    var githubToken: String = ""
    var githubRepository: String = ""
    var notionToken: String = ""
    var defaultNotionPageId: String = ""
    
    var isGitHubConfigured: Bool {
        !githubToken.isEmpty && !githubRepository.isEmpty
    }
    
    var isNotionConfigured: Bool {
        !notionToken.isEmpty
    }
    
    var repositoryOwner: String {
        githubRepository.components(separatedBy: "/").first ?? ""
    }
    
    var repositoryName: String {
        githubRepository.components(separatedBy: "/").last ?? ""
    }
}

enum AutomationTask: String, CaseIterable {
    case doctor = "doctor"
    case ci = "ci"
    case test = "test"
    case lint = "lint"
    case format = "fmt"
    case bootstrap = "bootstrap"
    
    var displayName: String {
        switch self {
        case .doctor: return "System Health Check"
        case .ci: return "Run CI Suite"
        case .test: return "Run Tests"
        case .lint: return "Lint Code"
        case .format: return "Format Code"
        case .bootstrap: return "Bootstrap Environment"
        }
    }
    
    var icon: String {
        switch self {
        case .doctor: return "stethoscope"
        case .ci: return "gearshape.2"
        case .test: return "checkmark.circle"
        case .lint: return "magnifyingglass"
        case .format: return "textformat"
        case .bootstrap: return "arrow.up.circle"
        }
    }
    
    var color: String {
        switch self {
        case .doctor: return "blue"
        case .ci: return "purple"
        case .test: return "green"
        case .lint: return "orange"
        case .format: return "indigo"
        case .bootstrap: return "red"
        }
    }
}

enum NotionStatus: String, CaseIterable {
    case notStarted = "Not started"
    case inProgress = "In progress"
    case completed = "Completed"
    case blocked = "Blocked"
    case cancelled = "Cancelled"
    
    var color: String {
        switch self {
        case .notStarted: return "gray"
        case .inProgress: return "blue"
        case .completed: return "green"
        case .blocked: return "red"
        case .cancelled: return "orange"
        }
    }
}

struct WorkflowRun: Identifiable {
    let id = UUID()
    let task: AutomationTask
    let timestamp: Date
    let status: RunStatus
    let notionPageId: String?
    
    enum RunStatus {
        case queued
        case inProgress
        case completed
        case failed
    }
}

struct LogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let level: Level
    let message: String
    
    enum Level {
        case info
        case success
        case warning
        case error
        
        var color: String {
            switch self {
            case .info: return "blue"
            case .success: return "green"
            case .warning: return "orange"
            case .error: return "red"
            }
        }
        
        var icon: String {
            switch self {
            case .info: return "info.circle"
            case .success: return "checkmark.circle"
            case .warning: return "exclamationmark.triangle"
            case .error: return "xmark.circle"
            }
        }
    }
}