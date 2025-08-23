import XCTest
@testable import AiCanAutomation

class AiCanAutomationTests: XCTestCase {
    
    var configManager: ConfigurationManager!
    var automationService: AutomationService!
    
    override func setUp() {
        super.setUp()
        configManager = ConfigurationManager()
        automationService = AutomationService()
    }
    
    override func tearDown() {
        configManager = nil
        automationService = nil
        super.tearDown()
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationInitialization() {
        let config = Configuration()
        
        XCTAssertTrue(config.githubToken.isEmpty)
        XCTAssertTrue(config.githubRepository.isEmpty)
        XCTAssertTrue(config.notionToken.isEmpty)
        XCTAssertTrue(config.defaultNotionPageId.isEmpty)
        XCTAssertFalse(config.isGitHubConfigured)
        XCTAssertFalse(config.isNotionConfigured)
    }
    
    func testGitHubConfiguration() {
        configManager.updateGitHubSettings(token: "ghp_test123", repository: "user/repo")
        
        XCTAssertTrue(configManager.configuration.isGitHubConfigured)
        XCTAssertEqual(configManager.configuration.repositoryOwner, "user")
        XCTAssertEqual(configManager.configuration.repositoryName, "repo")
    }
    
    func testNotionConfiguration() {
        configManager.updateNotionSettings(token: "secret_test123", defaultPageId: "page123")
        
        XCTAssertTrue(configManager.configuration.isNotionConfigured)
        XCTAssertEqual(configManager.configuration.defaultNotionPageId, "page123")
    }
    
    func testConfigurationValidation() {
        // Test empty configuration
        var errors = configManager.validateConfiguration()
        XCTAssertTrue(errors.contains("GitHub token is required"))
        XCTAssertTrue(errors.contains("GitHub repository is required"))
        
        // Test invalid GitHub token format
        configManager.updateGitHubSettings(token: "invalid_token", repository: "user/repo")
        errors = configManager.validateConfiguration()
        XCTAssertTrue(errors.contains("GitHub token format appears invalid"))
        
        // Test invalid repository format
        configManager.updateGitHubSettings(token: "ghp_valid", repository: "invalid")
        errors = configManager.validateConfiguration()
        XCTAssertTrue(errors.contains("Repository should be in format 'owner/repo'"))
        
        // Test valid configuration
        configManager.updateGitHubSettings(token: "ghp_valid", repository: "user/repo")
        errors = configManager.validateConfiguration()
        XCTAssertFalse(errors.contains("GitHub token is required"))
        XCTAssertFalse(errors.contains("GitHub repository is required"))
    }
    
    // MARK: - Automation Task Tests
    
    func testAutomationTaskProperties() {
        let doctorTask = AutomationTask.doctor
        
        XCTAssertEqual(doctorTask.displayName, "System Health Check")
        XCTAssertEqual(doctorTask.icon, "stethoscope")
        XCTAssertEqual(doctorTask.color, "blue")
        XCTAssertEqual(doctorTask.rawValue, "doctor")
    }
    
    func testAllAutomationTasks() {
        let allTasks = AutomationTask.allCases
        
        XCTAssertEqual(allTasks.count, 6)
        XCTAssertTrue(allTasks.contains(.doctor))
        XCTAssertTrue(allTasks.contains(.ci))
        XCTAssertTrue(allTasks.contains(.test))
        XCTAssertTrue(allTasks.contains(.lint))
        XCTAssertTrue(allTasks.contains(.format))
        XCTAssertTrue(allTasks.contains(.bootstrap))
    }
    
    // MARK: - Notion Status Tests
    
    func testNotionStatusProperties() {
        let inProgressStatus = NotionStatus.inProgress
        
        XCTAssertEqual(inProgressStatus.rawValue, "In progress")
        XCTAssertEqual(inProgressStatus.color, "blue")
    }
    
    func testAllNotionStatuses() {
        let allStatuses = NotionStatus.allCases
        
        XCTAssertEqual(allStatuses.count, 5)
        XCTAssertTrue(allStatuses.contains(.notStarted))
        XCTAssertTrue(allStatuses.contains(.inProgress))
        XCTAssertTrue(allStatuses.contains(.completed))
        XCTAssertTrue(allStatuses.contains(.blocked))
        XCTAssertTrue(allStatuses.contains(.cancelled))
    }
    
    // MARK: - WorkflowRun Tests
    
    func testWorkflowRunCreation() {
        let run = WorkflowRun(
            task: .doctor,
            timestamp: Date(),
            status: .queued,
            notionPageId: "test-page"
        )
        
        XCTAssertEqual(run.task, .doctor)
        XCTAssertEqual(run.status, .queued)
        XCTAssertEqual(run.notionPageId, "test-page")
        XCTAssertNotNil(run.id)
    }
    
    // MARK: - LogEntry Tests
    
    func testLogEntryCreation() {
        let entry = LogEntry(
            timestamp: Date(),
            level: .info,
            message: "Test log message"
        )
        
        XCTAssertEqual(entry.level, .info)
        XCTAssertEqual(entry.message, "Test log message")
        XCTAssertEqual(entry.level.color, "blue")
        XCTAssertEqual(entry.level.icon, "info.circle")
        XCTAssertNotNil(entry.id)
    }
    
    func testLogLevelProperties() {
        XCTAssertEqual(LogEntry.Level.success.color, "green")
        XCTAssertEqual(LogEntry.Level.warning.color, "orange")
        XCTAssertEqual(LogEntry.Level.error.color, "red")
        
        XCTAssertEqual(LogEntry.Level.success.icon, "checkmark.circle")
        XCTAssertEqual(LogEntry.Level.warning.icon, "exclamationmark.triangle")
        XCTAssertEqual(LogEntry.Level.error.icon, "xmark.circle")
    }
    
    // MARK: - AutomationService Tests
    
    func testAutomationServiceInitialization() {
        XCTAssertTrue(automationService.workflowRuns.isEmpty)
        XCTAssertTrue(automationService.logs.isEmpty)
        XCTAssertFalse(automationService.isLoading)
    }
    
    func testLogManagement() {
        // Add logs and verify they're stored
        let expectation = XCTestExpectation(description: "Logs added")
        
        Task {
            // Trigger a workflow to generate logs
            let config = Configuration()
            let result = await automationService.triggerWorkflow(
                task: .doctor,
                configuration: config
            )
            
            await MainActor.run {
                // Should have error log due to missing configuration
                XCTAssertFalse(automationService.logs.isEmpty)
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    // MARK: - Integration Tests
    
    func testConfigurationPersistence() {
        let originalToken = "ghp_test123"
        let originalRepo = "user/repo"
        
        configManager.updateGitHubSettings(token: originalToken, repository: originalRepo)
        
        // Create new config manager to test persistence
        let newConfigManager = ConfigurationManager()
        newConfigManager.loadConfiguration()
        
        XCTAssertEqual(newConfigManager.configuration.githubToken, originalToken)
        XCTAssertEqual(newConfigManager.configuration.githubRepository, originalRepo)
        
        // Clean up
        configManager.clearConfiguration()
    }
    
    func testWorkflowExecutionFlow() {
        let expectation = XCTestExpectation(description: "Workflow execution")
        
        Task {
            let config = Configuration()
            
            // Test with unconfigured GitHub (should fail)
            let result = await automationService.triggerWorkflow(
                task: .doctor,
                configuration: config
            )
            
            await MainActor.run {
                switch result {
                case .failure(let error as AutomationError):
                    XCTAssertEqual(error.localizedDescription, "GitHub not configured")
                case .failure:
                    XCTFail("Unexpected error type")
                case .success:
                    XCTFail("Should have failed with unconfigured GitHub")
                }
                
                expectation.fulfill()
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    // MARK: - Performance Tests
    
    func testLogPerformance() {
        measure {
            let automationService = AutomationService()
            
            // Add many log entries
            for i in 0..<1000 {
                let entry = LogEntry(
                    timestamp: Date(),
                    level: .info,
                    message: "Performance test log \(i)"
                )
                automationService.logs.append(entry)
            }
            
            // Test log management (should keep only 100)
            if automationService.logs.count > 100 {
                automationService.logs.removeFirst(automationService.logs.count - 100)
            }
            
            XCTAssertLesssThanOrEqual(automationService.logs.count, 100)
        }
    }
}

// MARK: - Mock Data Extensions

extension Configuration {
    static func mockGitHubConfigured() -> Configuration {
        var config = Configuration()
        config.githubToken = "ghp_mock123456789"
        config.githubRepository = "test/repo"
        return config
    }
    
    static func mockFullyConfigured() -> Configuration {
        var config = Configuration.mockGitHubConfigured()
        config.notionToken = "secret_mock123456789"
        config.defaultNotionPageId = "mock-page-id"
        return config
    }
}

extension AutomationError {
    var localizedDescription: String {
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