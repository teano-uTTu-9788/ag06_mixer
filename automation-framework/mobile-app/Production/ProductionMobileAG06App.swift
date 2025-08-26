import SwiftUI
import BackgroundTasks
import UserNotifications

@main
struct ProductionMobileAG06App: App {
    @StateObject private var mixerService = ProductionMixerService()
    @StateObject private var configManager = ConfigurationManager()
    @StateObject private var automationService = AutomationService()
    
    // Production services
    @StateObject private var logger = StructuredLogger()
    @StateObject private var performanceMonitor = PerformanceMonitor()
    @StateObject private var crashReporter = CrashReporter()
    @StateObject private var abTestManager = ABTestManager()
    @StateObject private var featureFlags = FeatureFlagManager()
    @StateObject private var sreObservability = SREObservability()
    @StateObject private var alertManager = AlertManager()
    @StateObject private var healthChecker = HealthChecker()
    
    @Environment(\.scenePhase) private var scenePhase
    
    init() {
        setupProductionServices()
        registerBackgroundTasks()
        requestNotificationPermissions()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(mixerService)
                .environmentObject(configManager)
                .environmentObject(automationService)
                .environmentObject(abTestManager)
                .environmentObject(featureFlags)
                .onAppear {
                    startProductionMonitoring()
                }
                .onChange(of: scenePhase) { oldPhase, newPhase in
                    handleScenePhaseChange(from: oldPhase, to: newPhase)
                }
        }
    }
    
    // MARK: - Production Setup
    private func setupProductionServices() {
        // Initialize crash reporting
        crashReporter.initialize(apiKey: ProcessInfo.processInfo.environment["CRASH_REPORTER_KEY"] ?? "")
        
        // Setup structured logging
        logger.configure(
            projectId: ProcessInfo.processInfo.environment["GCP_PROJECT_ID"] ?? "ag06-mixer",
            environment: ProcessInfo.processInfo.environment["ENVIRONMENT"] ?? "production"
        )
        
        // Initialize feature flags
        featureFlags.initialize(
            apiKey: ProcessInfo.processInfo.environment["FEATURE_FLAG_KEY"] ?? "",
            environment: ProcessInfo.processInfo.environment["ENVIRONMENT"] ?? "production"
        )
        
        // Setup A/B testing
        abTestManager.initialize(
            userId: getUserId(),
            attributes: [
                "platform": "ios",
                "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0",
                "device_model": UIDevice.current.model
            ]
        )
        
        // Configure SRE observability
        sreObservability.configure(
            serviceName: "mobile-ag06-mixer",
            environment: ProcessInfo.processInfo.environment["ENVIRONMENT"] ?? "production"
        )
        
        // Setup health checks
        setupHealthChecks()
        
        // Log app launch
        logger.log(
            severity: .info,
            message: "App launched",
            labels: [
                "version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0",
                "build": Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1",
                "environment": ProcessInfo.processInfo.environment["ENVIRONMENT"] ?? "production"
            ]
        )
        
        // Track launch event
        performanceMonitor.startTrace(name: "app_launch")
    }
    
    private func startProductionMonitoring() {
        // Record app launch completion
        performanceMonitor.stopTrace(name: "app_launch")
        
        // Start continuous monitoring
        performanceMonitor.startMonitoring()
        healthChecker.startMonitoring(interval: 30)
        
        // Register SLIs
        sreObservability.registerSLI(name: "app_availability", target: 0.999)
        sreObservability.registerSLI(name: "ui_responsiveness", target: 0.95)
        sreObservability.registerSLI(name: "crash_free_rate", target: 0.999)
        
        // Track app start event
        let startupTime = ProcessInfo.processInfo.systemUptime
        performanceMonitor.recordMetric(
            name: "app_startup_time",
            value: startupTime,
            unit: .seconds,
            tags: ["cold_start": "true"]
        )
        
        // Check for A/B test assignments
        checkABTestAssignments()
        
        // Load remote configuration
        Task {
            await loadRemoteConfiguration()
        }
    }
    
    // MARK: - Health Checks
    private func setupHealthChecks() {
        // App health check
        healthChecker.registerCheck(name: "app_responsive") { 
            return true // This runs on main thread, so if it executes, app is responsive
        }
        
        // Memory health check
        healthChecker.registerCheck(name: "memory_pressure") {
            let memoryInfo = ProcessInfo.processInfo.physicalMemory
            let memoryUsage = getMemoryUsage()
            return memoryUsage < (Double(memoryInfo) * 0.8) // Alert if using >80% memory
        }
        
        // Disk space check
        healthChecker.registerCheck(name: "disk_space") {
            return getDiskSpaceAvailable() > 100_000_000 // 100MB minimum
        }
        
        // Network connectivity check
        healthChecker.registerCheck(name: "network_connectivity") { [weak mixerService] in
            return mixerService?.connectionStatus.isConnected ?? false
        }
    }
    
    // MARK: - Scene Phase Handling
    private func handleScenePhaseChange(from oldPhase: ScenePhase, to newPhase: ScenePhase) {
        logger.log(
            severity: .info,
            message: "Scene phase changed",
            labels: ["from": String(describing: oldPhase), "to": String(describing: newPhase)]
        )
        
        switch newPhase {
        case .active:
            handleAppBecameActive()
        case .inactive:
            handleAppBecameInactive()
        case .background:
            handleAppEnteredBackground()
        @unknown default:
            break
        }
    }
    
    private func handleAppBecameActive() {
        // Resume monitoring
        performanceMonitor.resumeMonitoring()
        
        // Record activation
        sreObservability.recordSLI(name: "app_availability", value: 1.0)
        
        // Sync feature flags
        Task {
            await featureFlags.refresh()
        }
        
        // Check for updates
        checkForUpdates()
        
        // Resume mixer if it was running
        if mixerService.isRunning {
            mixerService.enterForegroundMode()
        }
    }
    
    private func handleAppBecameInactive() {
        // Prepare for background
        performanceMonitor.pauseMonitoring()
        
        // Save state
        saveApplicationState()
    }
    
    private func handleAppEnteredBackground() {
        // Switch to battery-saving mode
        if mixerService.isRunning {
            mixerService.enterBackgroundMode()
        }
        
        // Schedule background tasks
        scheduleBackgroundRefresh()
        
        // Flush analytics
        performanceMonitor.flush()
        crashReporter.flush()
        
        // Log background entry
        logger.log(severity: .info, message: "App entered background")
    }
    
    // MARK: - Background Tasks
    private func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.ag06.mixer.refresh",
            using: nil
        ) { task in
            self.handleBackgroundRefresh(task: task as! BGAppRefreshTask)
        }
        
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.ag06.mixer.analytics",
            using: nil
        ) { task in
            self.handleAnalyticsUpload(task: task as! BGProcessingTask)
        }
    }
    
    private func scheduleBackgroundRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: "com.ag06.mixer.refresh")
        request.earliestBeginDate = Date(timeIntervalSinceNow: 3600) // 1 hour
        
        do {
            try BGTaskScheduler.shared.submit(request)
            logger.log(severity: .debug, message: "Scheduled background refresh")
        } catch {
            logger.log(severity: .error, message: "Failed to schedule background refresh", labels: ["error": error.localizedDescription])
        }
    }
    
    private func handleBackgroundRefresh(task: BGAppRefreshTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }
        
        Task {
            // Refresh feature flags
            await featureFlags.refresh()
            
            // Check for critical alerts
            let hasAlerts = await alertManager.checkForCriticalAlerts()
            
            if hasAlerts {
                sendLocalNotification(
                    title: "AG06 Mixer Alert",
                    body: "Critical system alert requires attention"
                )
            }
            
            task.setTaskCompleted(success: true)
        }
        
        // Schedule next refresh
        scheduleBackgroundRefresh()
    }
    
    private func handleAnalyticsUpload(task: BGProcessingTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }
        
        Task {
            // Upload analytics
            await performanceMonitor.uploadMetrics()
            await crashReporter.uploadReports()
            
            task.setTaskCompleted(success: true)
        }
    }
    
    // MARK: - Notifications
    private func requestNotificationPermissions() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: [.alert, .sound, .badge]
        ) { granted, error in
            if granted {
                logger.log(severity: .info, message: "Notification permissions granted")
            } else if let error = error {
                logger.log(severity: .warning, message: "Notification permissions denied", labels: ["error": error.localizedDescription])
            }
        }
    }
    
    private func sendLocalNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                logger.log(severity: .error, message: "Failed to send notification", labels: ["error": error.localizedDescription])
            }
        }
    }
    
    // MARK: - A/B Testing
    private func checkABTestAssignments() {
        // Check for active experiments
        let experiments = [
            "new_mixer_ui",
            "advanced_eq_controls",
            "ai_powered_suggestions",
            "gesture_controls"
        ]
        
        for experiment in experiments {
            if abTestManager.isInExperiment(experiment) {
                let variant = abTestManager.getVariant(experiment) ?? "control"
                logger.log(
                    severity: .info,
                    message: "User assigned to experiment",
                    labels: ["experiment": experiment, "variant": variant]
                )
                
                // Apply experiment-specific configuration
                applyExperimentConfiguration(experiment: experiment, variant: variant)
            }
        }
    }
    
    private func applyExperimentConfiguration(experiment: String, variant: String) {
        switch experiment {
        case "new_mixer_ui":
            featureFlags.setOverride(feature: "use_new_ui", enabled: variant == "treatment")
        case "advanced_eq_controls":
            featureFlags.setOverride(feature: "show_advanced_eq", enabled: variant == "treatment")
        case "ai_powered_suggestions":
            featureFlags.setOverride(feature: "enable_ai_suggestions", enabled: variant == "treatment")
        case "gesture_controls":
            featureFlags.setOverride(feature: "enable_gestures", enabled: variant == "treatment")
        default:
            break
        }
    }
    
    // MARK: - Remote Configuration
    private func loadRemoteConfiguration() async {
        do {
            // Fetch remote config
            let configUrl = URL(string: "https://api.ag06mixer.com/config/mobile")!
            let (data, _) = try await URLSession.shared.data(from: configUrl)
            
            if let config = try? JSONDecoder().decode(RemoteConfig.self, from: data) {
                applyRemoteConfiguration(config)
            }
        } catch {
            logger.log(severity: .warning, message: "Failed to load remote config", labels: ["error": error.localizedDescription])
        }
    }
    
    private func applyRemoteConfiguration(_ config: RemoteConfig) {
        // Apply remote settings
        if let minVersion = config.minimumVersion {
            checkMinimumVersion(minVersion)
        }
        
        if let maintenanceMode = config.maintenanceMode, maintenanceMode {
            showMaintenanceMode()
        }
        
        // Update feature flags from remote
        for (feature, enabled) in config.featureFlags ?? [:] {
            featureFlags.setOverride(feature: feature, enabled: enabled)
        }
    }
    
    // MARK: - Utility Methods
    private func getUserId() -> String {
        if let userId = UserDefaults.standard.string(forKey: "user_id") {
            return userId
        } else {
            let newUserId = UUID().uuidString
            UserDefaults.standard.set(newUserId, forKey: "user_id")
            return newUserId
        }
    }
    
    private func getMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        return result == KERN_SUCCESS ? Double(info.resident_size) : 0
    }
    
    private func getDiskSpaceAvailable() -> Int64 {
        let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
        do {
            let values = try fileURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            if let capacity = values.volumeAvailableCapacityForImportantUsage {
                return capacity
            }
        } catch {
            logger.log(severity: .error, message: "Error retrieving disk space", labels: ["error": error.localizedDescription])
        }
        return 0
    }
    
    private func saveApplicationState() {
        // Save current state for restoration
        let state = ApplicationState(
            mixerRunning: mixerService.isRunning,
            lastSettings: mixerService.settings,
            timestamp: Date()
        )
        
        if let encoded = try? JSONEncoder().encode(state) {
            UserDefaults.standard.set(encoded, forKey: "app_state")
        }
    }
    
    private func checkForUpdates() {
        // Check if update is available
        Task {
            // Implementation would check App Store for updates
            logger.log(severity: .debug, message: "Checking for app updates")
        }
    }
    
    private func checkMinimumVersion(_ minVersion: String) {
        let currentVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
        
        if currentVersion.compare(minVersion, options: .numeric) == .orderedAscending {
            // Force update required
            showForceUpdateAlert()
        }
    }
    
    private func showMaintenanceMode() {
        // Show maintenance mode UI
        logger.log(severity: .warning, message: "App in maintenance mode")
    }
    
    private func showForceUpdateAlert() {
        // Show force update alert
        logger.log(severity: .critical, message: "Force update required")
    }
}

// MARK: - Supporting Types
struct ApplicationState: Codable {
    let mixerRunning: Bool
    let lastSettings: MixerSettings
    let timestamp: Date
}

struct RemoteConfig: Codable {
    let minimumVersion: String?
    let maintenanceMode: Bool?
    let featureFlags: [String: Bool]?
}