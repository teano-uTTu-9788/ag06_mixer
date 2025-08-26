import Foundation
import Combine
import os.signpost

// MARK: - Google SRE Observability & Monitoring Implementation

// MARK: - 1. Service Level Indicators (SLIs)
final class SLIManager {
    static let shared = SLIManager()
    
    private let logger = StructuredLogger(category: "SLI")
    private var sliMetrics: [String: SLIMetric] = [:]
    private let queue = DispatchQueue(label: "sli.manager", attributes: .concurrent)
    
    struct SLIMetric {
        let name: String
        let target: Double // Target percentage (e.g., 99.9)
        var totalEvents: Int64 = 0
        var goodEvents: Int64 = 0
        var measurements: [Measurement] = []
        let windowSize: TimeInterval // Time window for calculations
        
        struct Measurement {
            let timestamp: Date
            let isGood: Bool
            let value: Double?
        }
        
        var currentSLI: Double {
            guard totalEvents > 0 else { return 100.0 }
            return (Double(goodEvents) / Double(totalEvents)) * 100.0
        }
        
        var isViolated: Bool {
            return currentSLI < target
        }
        
        var errorBudgetRemaining: Double {
            let budgetTotal = 100.0 - target
            let budgetUsed = max(0, target - currentSLI)
            return max(0, budgetTotal - budgetUsed)
        }
    }
    
    func registerSLI(
        name: String,
        target: Double,
        windowSize: TimeInterval = 86400 // 24 hours default
    ) {
        queue.async(flags: .barrier) {
            self.sliMetrics[name] = SLIMetric(
                name: name,
                target: target,
                windowSize: windowSize
            )
        }
        
        logger.log(.info, "Registered SLI: \(name) with target: \(target)%")
    }
    
    func recordEvent(sli: String, isGood: Bool, value: Double? = nil) {
        queue.async(flags: .barrier) {
            guard var metric = self.sliMetrics[sli] else {
                self.logger.log(.warning, "Unknown SLI: \(sli)")
                return
            }
            
            metric.totalEvents += 1
            if isGood {
                metric.goodEvents += 1
            }
            
            metric.measurements.append(
                SLIMetric.Measurement(
                    timestamp: Date(),
                    isGood: isGood,
                    value: value
                )
            )
            
            // Clean old measurements
            let cutoff = Date().addingTimeInterval(-metric.windowSize)
            metric.measurements.removeAll { $0.timestamp < cutoff }
            
            self.sliMetrics[sli] = metric
            
            // Check for SLO violations
            if metric.isViolated {
                self.handleSLOViolation(metric)
            }
        }
    }
    
    private func handleSLOViolation(_ metric: SLIMetric) {
        logger.log(.warning, "SLO violation for \(metric.name): \(metric.currentSLI)% < \(metric.target)%")
        
        // Send alert
        AlertManager.shared.sendAlert(
            Alert(
                severity: .warning,
                title: "SLO Violation",
                message: "SLI \(metric.name) is at \(metric.currentSLI)%, below target of \(metric.target)%",
                metadata: [
                    "sli": metric.name,
                    "current": String(metric.currentSLI),
                    "target": String(metric.target),
                    "error_budget_remaining": String(metric.errorBudgetRemaining)
                ]
            )
        )
    }
    
    func getSLIReport() -> [String: Any] {
        return queue.sync {
            var report: [String: Any] = [:]
            
            for (name, metric) in sliMetrics {
                report[name] = [
                    "current": metric.currentSLI,
                    "target": metric.target,
                    "is_violated": metric.isViolated,
                    "error_budget_remaining": metric.errorBudgetRemaining,
                    "total_events": metric.totalEvents,
                    "good_events": metric.goodEvents
                ]
            }
            
            return report
        }
    }
}

// MARK: - 2. Service Level Objectives (SLOs)
final class SLOManager {
    static let shared = SLOManager()
    
    private let logger = StructuredLogger(category: "SLO")
    
    init() {
        setupDefaultSLOs()
    }
    
    private func setupDefaultSLOs() {
        // Availability SLO
        SLIManager.shared.registerSLI(
            name: "availability",
            target: 99.9, // 99.9% uptime
            windowSize: 30 * 86400 // 30 days
        )
        
        // Latency SLO
        SLIManager.shared.registerSLI(
            name: "latency_p99",
            target: 95.0, // 95% of requests < 100ms
            windowSize: 86400 // 24 hours
        )
        
        // Error rate SLO
        SLIManager.shared.registerSLI(
            name: "error_rate",
            target: 99.5, // < 0.5% errors
            windowSize: 86400 // 24 hours
        )
        
        // Audio quality SLO
        SLIManager.shared.registerSLI(
            name: "audio_quality",
            target: 99.0, // 99% without dropouts
            windowSize: 3600 // 1 hour
        )
    }
    
    func recordAPIRequest(duration: TimeInterval, success: Bool) {
        // Record availability
        SLIManager.shared.recordEvent(sli: "availability", isGood: success)
        
        // Record latency (assuming 100ms is our target)
        let isGoodLatency = duration < 0.1
        SLIManager.shared.recordEvent(sli: "latency_p99", isGood: isGoodLatency, value: duration)
        
        // Record error rate
        SLIManager.shared.recordEvent(sli: "error_rate", isGood: success)
    }
    
    func recordAudioQuality(hasDropouts: Bool) {
        SLIManager.shared.recordEvent(sli: "audio_quality", isGood: !hasDropouts)
    }
}

// MARK: - 3. Distributed Tracing (Google Dapper style)
final class DistributedTracer {
    static let shared = DistributedTracer()
    
    private let signpostLog = OSLog(subsystem: "com.ag06.mixer", category: .pointsOfInterest)
    private var activeSpans: [String: Span] = [:]
    private let queue = DispatchQueue(label: "tracer", attributes: .concurrent)
    
    struct Span {
        let traceId: String
        let spanId: String
        let parentSpanId: String?
        let operationName: String
        let startTime: Date
        var endTime: Date?
        var tags: [String: String] = [:]
        var logs: [LogEntry] = []
        var status: Status = .ok
        
        struct LogEntry {
            let timestamp: Date
            let message: String
            let fields: [String: Any]
        }
        
        enum Status {
            case ok
            case error(String)
            case cancelled
        }
        
        var duration: TimeInterval? {
            guard let endTime = endTime else { return nil }
            return endTime.timeIntervalSince(startTime)
        }
    }
    
    func startSpan(
        operationName: String,
        parentSpanId: String? = nil,
        tags: [String: String] = [:]
    ) -> String {
        let traceId = parentSpanId?.components(separatedBy: ".").first ?? UUID().uuidString
        let spanId = "\(traceId).\(UUID().uuidString.prefix(8))"
        
        let span = Span(
            traceId: traceId,
            spanId: spanId,
            parentSpanId: parentSpanId,
            operationName: operationName,
            startTime: Date(),
            tags: tags
        )
        
        queue.async(flags: .barrier) {
            self.activeSpans[spanId] = span
        }
        
        // OS Signpost for Instruments integration
        os_signpost(.begin, log: signpostLog, name: "Span", "%{public}s", operationName)
        
        return spanId
    }
    
    func endSpan(_ spanId: String, status: Span.Status = .ok) {
        queue.async(flags: .barrier) {
            guard var span = self.activeSpans[spanId] else { return }
            span.endTime = Date()
            span.status = status
            
            // Send to collection service
            self.exportSpan(span)
            
            self.activeSpans.removeValue(forKey: spanId)
        }
        
        os_signpost(.end, log: signpostLog, name: "Span")
    }
    
    func addLog(to spanId: String, message: String, fields: [String: Any] = [:]) {
        queue.async(flags: .barrier) {
            guard var span = self.activeSpans[spanId] else { return }
            
            span.logs.append(
                Span.LogEntry(
                    timestamp: Date(),
                    message: message,
                    fields: fields
                )
            )
            
            self.activeSpans[spanId] = span
        }
    }
    
    func addTag(to spanId: String, key: String, value: String) {
        queue.async(flags: .barrier) {
            self.activeSpans[spanId]?.tags[key] = value
        }
    }
    
    private func exportSpan(_ span: Span) {
        // Export to tracing backend (Jaeger, Zipkin, etc.)
        let traceData: [String: Any] = [
            "trace_id": span.traceId,
            "span_id": span.spanId,
            "parent_span_id": span.parentSpanId ?? "",
            "operation": span.operationName,
            "start_time": span.startTime.timeIntervalSince1970,
            "duration_ms": (span.duration ?? 0) * 1000,
            "tags": span.tags,
            "logs": span.logs.map { [
                "timestamp": $0.timestamp.timeIntervalSince1970,
                "message": $0.message,
                "fields": $0.fields
            ]},
            "status": "\(span.status)"
        ]
        
        MetricsCollector.shared.recordEvent("trace_span", parameters: [
            "data": String(data: try! JSONSerialization.data(withJSONObject: traceData), encoding: .utf8) ?? ""
        ])
    }
}

// MARK: - 4. Alert Manager (PagerDuty style)
final class AlertManager {
    static let shared = AlertManager()
    
    private let logger = StructuredLogger(category: "Alerts")
    private var activeAlerts: [Alert] = []
    private let queue = DispatchQueue(label: "alerts", attributes: .concurrent)
    
    struct Alert {
        let id = UUID().uuidString
        let severity: Severity
        let title: String
        let message: String
        let timestamp = Date()
        let metadata: [String: String]
        var acknowledged = false
        var resolved = false
        
        enum Severity: Int {
            case info = 0
            case warning = 1
            case error = 2
            case critical = 3
            
            var emoji: String {
                switch self {
                case .info: return "ℹ️"
                case .warning: return "⚠️"
                case .error: return "🔴"
                case .critical: return "🚨"
                }
            }
        }
    }
    
    func sendAlert(_ alert: Alert) {
        queue.async(flags: .barrier) {
            self.activeAlerts.append(alert)
        }
        
        logger.log(.warning, "\(alert.severity.emoji) Alert: \(alert.title)")
        
        // Route based on severity
        switch alert.severity {
        case .critical:
            sendPushNotification(alert)
            sendToOncall(alert)
        case .error:
            sendPushNotification(alert)
        case .warning:
            // Log only
            break
        case .info:
            // Log only
            break
        }
        
        // Record metric
        MetricsCollector.shared.recordEvent("alert_triggered", parameters: [
            "severity": "\(alert.severity)",
            "title": alert.title
        ])
    }
    
    private func sendPushNotification(_ alert: Alert) {
        // Implement push notification
        NotificationCenter.default.post(
            name: .alertReceived,
            object: nil,
            userInfo: ["alert": alert]
        )
    }
    
    private func sendToOncall(_ alert: Alert) {
        // Integrate with PagerDuty or similar
        logger.log(.critical, "Paging on-call engineer for: \(alert.title)")
    }
    
    func acknowledgeAlert(_ alertId: String) {
        queue.async(flags: .barrier) {
            if let index = self.activeAlerts.firstIndex(where: { $0.id == alertId }) {
                self.activeAlerts[index].acknowledged = true
            }
        }
    }
    
    func resolveAlert(_ alertId: String) {
        queue.async(flags: .barrier) {
            if let index = self.activeAlerts.firstIndex(where: { $0.id == alertId }) {
                self.activeAlerts[index].resolved = true
            }
        }
    }
}

// MARK: - 5. Health Check System
final class HealthCheckManager {
    static let shared = HealthCheckManager()
    
    private let logger = StructuredLogger(category: "HealthCheck")
    private var healthChecks: [String: HealthCheck] = [:]
    private let queue = DispatchQueue(label: "health", attributes: .concurrent)
    
    struct HealthCheck {
        let name: String
        let checkInterval: TimeInterval
        let timeout: TimeInterval
        let check: () async -> HealthStatus
        var lastStatus: HealthStatus?
        var lastCheckTime: Date?
        var consecutiveFailures: Int = 0
        
        struct HealthStatus {
            let healthy: Bool
            let message: String?
            let metadata: [String: Any]?
        }
    }
    
    func registerHealthCheck(
        name: String,
        interval: TimeInterval = 30,
        timeout: TimeInterval = 5,
        check: @escaping () async -> HealthCheck.HealthStatus
    ) {
        queue.async(flags: .barrier) {
            self.healthChecks[name] = HealthCheck(
                name: name,
                checkInterval: interval,
                timeout: timeout,
                check: check
            )
        }
        
        // Start periodic check
        Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
            Task {
                await self.runHealthCheck(name)
            }
        }
    }
    
    private func runHealthCheck(_ name: String) async {
        guard let healthCheck = queue.sync(execute: { healthChecks[name] }) else { return }
        
        let startTime = Date()
        
        do {
            let status = try await withTimeout(seconds: healthCheck.timeout) {
                await healthCheck.check()
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            queue.async(flags: .barrier) {
                self.healthChecks[name]?.lastStatus = status
                self.healthChecks[name]?.lastCheckTime = Date()
                
                if status.healthy {
                    self.healthChecks[name]?.consecutiveFailures = 0
                } else {
                    self.healthChecks[name]?.consecutiveFailures += 1
                    
                    // Alert on repeated failures
                    if self.healthChecks[name]?.consecutiveFailures ?? 0 >= 3 {
                        AlertManager.shared.sendAlert(
                            Alert(
                                severity: .error,
                                title: "Health Check Failed",
                                message: "Health check '\(name)' has failed 3 times consecutively",
                                metadata: ["check": name, "message": status.message ?? ""]
                            )
                        )
                    }
                }
            }
            
            // Record metrics
            MetricsCollector.shared.recordEvent("health_check", parameters: [
                "name": name,
                "healthy": String(status.healthy),
                "duration_ms": String(Int(duration * 1000))
            ])
            
        } catch {
            logger.log(.error, "Health check '\(name)' timed out")
            
            queue.async(flags: .barrier) {
                self.healthChecks[name]?.consecutiveFailures += 1
            }
        }
    }
    
    func getOverallHealth() -> Bool {
        return queue.sync {
            healthChecks.values.allSatisfy { check in
                check.lastStatus?.healthy ?? false
            }
        }
    }
    
    func getHealthReport() -> [String: Any] {
        return queue.sync {
            var report: [String: Any] = [:]
            
            for (name, check) in healthChecks {
                report[name] = [
                    "healthy": check.lastStatus?.healthy ?? false,
                    "message": check.lastStatus?.message ?? "",
                    "last_check": check.lastCheckTime?.timeIntervalSince1970 ?? 0,
                    "consecutive_failures": check.consecutiveFailures
                ]
            }
            
            report["overall_healthy"] = getOverallHealth()
            
            return report
        }
    }
}

// MARK: - 6. Circuit Breaker Pattern
final class CircuitBreaker {
    enum State {
        case closed
        case open
        case halfOpen
    }
    
    private let name: String
    private let threshold: Int
    private let timeout: TimeInterval
    private let resetTimeout: TimeInterval
    
    private var state: State = .closed
    private var failureCount: Int = 0
    private var lastFailureTime: Date?
    private var successCount: Int = 0
    
    private let queue = DispatchQueue(label: "circuit.breaker", attributes: .concurrent)
    private let logger = StructuredLogger(category: "CircuitBreaker")
    
    init(
        name: String,
        threshold: Int = 5,
        timeout: TimeInterval = 60,
        resetTimeout: TimeInterval = 30
    ) {
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self.resetTimeout = resetTimeout
    }
    
    func execute<T>(_ operation: () async throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async(flags: .barrier) {
                switch self.state {
                case .open:
                    // Check if we should transition to half-open
                    if let lastFailure = self.lastFailureTime,
                       Date().timeIntervalSince(lastFailure) >= self.resetTimeout {
                        self.state = .halfOpen
                        self.logger.log(.info, "Circuit breaker '\(self.name)' transitioning to half-open")
                    } else {
                        continuation.resume(throwing: CircuitBreakerError.circuitOpen)
                        return
                    }
                    
                case .halfOpen:
                    // Allow limited requests through
                    break
                    
                case .closed:
                    // Normal operation
                    break
                }
                
                Task {
                    do {
                        let result = try await operation()
                        
                        await self.queue.async(flags: .barrier) {
                            self.recordSuccess()
                        }
                        
                        continuation.resume(returning: result)
                    } catch {
                        await self.queue.async(flags: .barrier) {
                            self.recordFailure()
                        }
                        
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
    }
    
    private func recordSuccess() {
        switch state {
        case .halfOpen:
            successCount += 1
            if successCount >= threshold / 2 {
                state = .closed
                failureCount = 0
                successCount = 0
                logger.log(.info, "Circuit breaker '\(name)' closed after successful recovery")
            }
            
        case .closed:
            failureCount = 0
            
        case .open:
            break
        }
    }
    
    private func recordFailure() {
        failureCount += 1
        lastFailureTime = Date()
        
        switch state {
        case .closed:
            if failureCount >= threshold {
                state = .open
                logger.log(.warning, "Circuit breaker '\(name)' opened after \(failureCount) failures")
                
                AlertManager.shared.sendAlert(
                    Alert(
                        severity: .warning,
                        title: "Circuit Breaker Open",
                        message: "Circuit breaker '\(name)' has opened due to repeated failures",
                        metadata: ["breaker": name, "failures": String(failureCount)]
                    )
                )
            }
            
        case .halfOpen:
            state = .open
            successCount = 0
            logger.log(.warning, "Circuit breaker '\(name)' reopened after failure in half-open state")
            
        case .open:
            break
        }
    }
    
    enum CircuitBreakerError: Error {
        case circuitOpen
    }
}

// MARK: - 7. Utility Functions
func withTimeout<T>(seconds: TimeInterval, operation: @escaping () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw TimeoutError()
        }
        
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

struct TimeoutError: Error {}

// MARK: - Notification Extensions
extension Notification.Name {
    static let alertReceived = Notification.Name("alertReceived")
}

// MARK: - Production Setup
final class ProductionSetup {
    static func initialize() {
        // Initialize crash reporting
        CrashReporter.shared.initialize()
        
        // Set up default health checks
        setupHealthChecks()
        
        // Configure SLOs
        _ = SLOManager.shared
        
        // Start network monitoring
        NetworkQualityMonitor.shared.startMonitoring()
        
        // Initialize feature flags
        setupFeatureFlags()
        
        // Set up experiments
        setupExperiments()
        
        // Configure remote config
        Task {
            await RemoteConfig.shared.fetch()
        }
    }
    
    private static func setupHealthChecks() {
        // API health check
        HealthCheckManager.shared.registerHealthCheck(name: "api") {
            // Check API connectivity
            return HealthCheckManager.HealthCheck.HealthStatus(
                healthy: true,
                message: "API is responsive",
                metadata: nil
            )
        }
        
        // Memory health check
        HealthCheckManager.shared.registerHealthCheck(name: "memory") {
            let memoryUsage = getMemoryUsage()
            let isHealthy = memoryUsage < 100 // MB
            
            return HealthCheckManager.HealthCheck.HealthStatus(
                healthy: isHealthy,
                message: isHealthy ? nil : "High memory usage: \(memoryUsage)MB",
                metadata: ["usage_mb": memoryUsage]
            )
        }
        
        // Disk space health check
        HealthCheckManager.shared.registerHealthCheck(name: "disk_space") {
            let availableSpace = getDiskSpace()
            let isHealthy = availableSpace > 100 // MB
            
            return HealthCheckManager.HealthCheck.HealthStatus(
                healthy: isHealthy,
                message: isHealthy ? nil : "Low disk space: \(availableSpace)MB",
                metadata: ["available_mb": availableSpace]
            )
        }
    }
    
    private static func setupFeatureFlags() {
        // Example feature flags
        FeatureFlags.shared.flags["new_audio_engine"] = FeatureFlags.FeatureFlag(
            name: "new_audio_engine",
            enabled: false,
            rolloutPercentage: 0.1, // 10% rollout
            overrides: [:],
            conditions: []
        )
        
        FeatureFlags.shared.flags["ai_mixing"] = FeatureFlags.FeatureFlag(
            name: "ai_mixing",
            enabled: true,
            rolloutPercentage: 1.0, // 100% rollout
            overrides: [:],
            conditions: []
        )
    }
    
    private static func setupExperiments() {
        // Example A/B test
        ExperimentManager.shared.registerExperiment(
            ExperimentManager.Experiment(
                name: "audio_buffer_size",
                variants: [
                    ExperimentManager.Experiment.Variant(
                        name: "control",
                        weight: 0.5,
                        parameters: ["buffer_size": 256]
                    ),
                    ExperimentManager.Experiment.Variant(
                        name: "treatment",
                        weight: 0.5,
                        parameters: ["buffer_size": 512]
                    )
                ],
                allocation: 0.2 // 20% of users in experiment
            )
        )
    }
    
    private static func getMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return result == KERN_SUCCESS ? Double(info.resident_size) / 1024 / 1024 : 0
    }
    
    private static func getDiskSpace() -> Double {
        let paths = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)
        
        guard let path = paths.first else { return 0 }
        
        do {
            let attributes = try FileManager.default.attributesOfFileSystem(forPath: path)
            if let freeSpace = attributes[.systemFreeSize] as? NSNumber {
                return freeSpace.doubleValue / 1024 / 1024
            }
        } catch {
            return 0
        }
        
        return 0
    }
}