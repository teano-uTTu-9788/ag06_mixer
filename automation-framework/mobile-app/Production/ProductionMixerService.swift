import Foundation
import Combine
import Network

/// Production-ready mixer service with Google/Meta best practices integrated
@MainActor
final class ProductionMixerService: ObservableObject {
    // MARK: - Published Properties
    @Published private(set) var audioMetrics = AudioMetrics()
    @Published private(set) var connectionStatus = ConnectionStatus()
    @Published private(set) var isRunning = false
    @Published private(set) var settings = MixerSettings()
    @Published private(set) var batteryMode: BatteryMode = .balanced
    @Published private(set) var subscriptionTier: SubscriptionTier = .free
    
    // MARK: - Production Services
    private let logger: StructuredLogger
    private let performanceMonitor: PerformanceMonitor
    private let crashReporter: CrashReporter
    private let abTestManager: ABTestManager
    private let featureFlags: FeatureFlagManager
    private let networkQuality: NetworkQualityMonitor
    private let metricsCollector: MetricsCollector
    private let sreObservability: SREObservability
    private let distributedTracing: DistributedTracing
    private let alertManager: AlertManager
    private let healthChecker: HealthChecker
    private let circuitBreaker: CircuitBreaker
    
    // MARK: - Private Properties
    private var updateTimer: Timer?
    private let session: URLSession
    private var eventSource: URLSessionDataTask?
    private var cancellables = Set<AnyCancellable>()
    private let monitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "com.ag06.networkmonitor")
    
    // MARK: - Configuration
    private let baseURL: String
    private let apiKey: String?
    
    // MARK: - Initialization
    init(configuration: MixerConfiguration = MixerConfiguration()) {
        self.baseURL = configuration.serverURL
        self.apiKey = configuration.apiKey
        self.subscriptionTier = configuration.subscriptionTier
        
        // Initialize production services
        self.logger = StructuredLogger()
        self.performanceMonitor = PerformanceMonitor()
        self.crashReporter = CrashReporter()
        self.abTestManager = ABTestManager()
        self.featureFlags = FeatureFlagManager()
        self.networkQuality = NetworkQualityMonitor()
        self.metricsCollector = MetricsCollector()
        self.sreObservability = SREObservability()
        self.distributedTracing = DistributedTracing()
        self.alertManager = AlertManager()
        self.healthChecker = HealthChecker()
        self.circuitBreaker = CircuitBreaker()
        
        // Configure URLSession with production settings
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = true
        config.tlsMinimumSupportedProtocolVersion = .TLSv12
        
        // Add distributed tracing headers
        if let traceHeaders = distributedTracing.getCurrentTraceHeaders() {
            config.httpAdditionalHeaders = traceHeaders
        }
        
        self.session = URLSession(configuration: config)
        
        setupNetworkMonitoring()
        setupProductionMonitoring()
        
        logger.log(
            severity: .info,
            message: "ProductionMixerService initialized",
            labels: ["version": "1.0.0", "tier": subscriptionTier.rawValue]
        )
    }
    
    // MARK: - Production Monitoring Setup
    private func setupProductionMonitoring() {
        // Register SLIs
        sreObservability.registerSLI(name: "mixer_availability", target: 0.999)
        sreObservability.registerSLI(name: "api_latency_p99", target: 0.5) // 500ms
        sreObservability.registerSLI(name: "audio_quality", target: 0.95)
        
        // Setup health checks
        healthChecker.registerCheck(name: "api_connectivity") { [weak self] in
            guard let self = self else { return false }
            return self.connectionStatus.isConnected
        }
        
        healthChecker.registerCheck(name: "audio_processing") { [weak self] in
            guard let self = self else { return false }
            return self.isRunning
        }
        
        // Start health monitoring
        healthChecker.startMonitoring(interval: 60)
        
        // Setup circuit breaker
        circuitBreaker.onOpen = { [weak self] in
            self?.logger.log(severity: .error, message: "Circuit breaker opened - API failures detected")
            self?.alertManager.fireAlert(
                severity: .critical,
                message: "Mixer API circuit breaker opened",
                labels: ["component": "mixer_service"]
            )
        }
        
        circuitBreaker.onHalfOpen = { [weak self] in
            self?.logger.log(severity: .warning, message: "Circuit breaker half-open - testing API")
        }
        
        circuitBreaker.onClose = { [weak self] in
            self?.logger.log(severity: .info, message: "Circuit breaker closed - API recovered")
            self?.alertManager.resolveAlert(labels: ["component": "mixer_service"])
        }
    }
    
    // MARK: - Network Monitoring
    private func setupNetworkMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                
                self.connectionStatus.isConnected = path.status == .satisfied
                self.connectionStatus.isExpensive = path.isExpensive
                self.connectionStatus.isConstrained = path.isConstrained
                
                // Update network quality metrics
                self.networkQuality.updateNetworkStatus(
                    isConnected: path.status == .satisfied,
                    isExpensive: path.isExpensive,
                    connectionType: self.getConnectionType(from: path)
                )
                
                // Record SLI
                self.sreObservability.recordSLI(
                    name: "mixer_availability",
                    value: path.status == .satisfied ? 1.0 : 0.0
                )
                
                // Adjust behavior based on network quality
                if path.isExpensive || path.isConstrained {
                    self.enterLowDataMode()
                }
                
                self.logger.log(
                    severity: .info,
                    message: "Network status changed",
                    labels: [
                        "connected": String(path.status == .satisfied),
                        "expensive": String(path.isExpensive),
                        "constrained": String(path.isConstrained)
                    ]
                )
            }
        }
        monitor.start(queue: monitorQueue)
    }
    
    // MARK: - Public Methods
    func startMixer() async throws {
        let span = distributedTracing.startSpan(name: "start_mixer")
        defer { distributedTracing.endSpan(span) }
        
        // Track performance
        let startTime = CFAbsoluteTimeGetCurrent()
        
        do {
            // Check circuit breaker
            guard await circuitBreaker.call() else {
                throw MixerError.connectionFailed
            }
            
            // Check feature flag
            guard featureFlags.isEnabled("mixer_start") else {
                logger.log(severity: .warning, message: "Mixer start disabled by feature flag")
                throw MixerError.featureDisabled
            }
            
            let url = URL(string: "\(baseURL)/api/start")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            // Add distributed tracing headers
            if let traceId = span.traceId {
                request.setValue(traceId, forHTTPHeaderField: "X-Trace-Id")
            }
            
            let (data, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                circuitBreaker.recordFailure()
                throw MixerError.invalidResponse
            }
            
            // Record latency
            let latency = CFAbsoluteTimeGetCurrent() - startTime
            sreObservability.recordSLI(name: "api_latency_p99", value: latency)
            performanceMonitor.recordMetric(name: "mixer_start_latency", value: latency, unit: .seconds)
            
            if httpResponse.statusCode == 200 {
                isRunning = true
                circuitBreaker.recordSuccess()
                startRealTimeUpdates()
                
                // Log success with metrics
                logger.log(
                    severity: .info,
                    message: "Mixer started successfully",
                    labels: [
                        "latency_ms": String(Int(latency * 1000)),
                        "status_code": String(httpResponse.statusCode)
                    ]
                )
                
                // Track custom event
                metricsCollector.trackEvent(
                    name: "mixer_started",
                    properties: ["method": "manual", "tier": subscriptionTier.rawValue]
                )
                
                // A/B test tracking
                if abTestManager.isInExperiment("new_mixer_ui") {
                    metricsCollector.trackEvent(
                        name: "mixer_started_experiment",
                        properties: ["variant": abTestManager.getVariant("new_mixer_ui") ?? "control"]
                    )
                }
            } else {
                circuitBreaker.recordFailure()
                throw MixerError.serverError(httpResponse.statusCode)
            }
            
        } catch {
            // Record failure metrics
            crashReporter.recordError(error, context: ["action": "start_mixer"])
            sreObservability.recordSLI(name: "mixer_availability", value: 0.0)
            
            // Alert if critical
            if error is MixerError {
                alertManager.fireAlert(
                    severity: .high,
                    message: "Failed to start mixer",
                    labels: ["error": error.localizedDescription]
                )
            }
            
            throw error
        }
    }
    
    func stopMixer() async throws {
        let span = distributedTracing.startSpan(name: "stop_mixer")
        defer { distributedTracing.endSpan(span) }
        
        do {
            let url = URL(string: "\(baseURL)/api/stop")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            // Add tracing
            if let traceId = span.traceId {
                request.setValue(traceId, forHTTPHeaderField: "X-Trace-Id")
            }
            
            let (_, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw MixerError.invalidResponse
            }
            
            isRunning = false
            stopRealTimeUpdates()
            
            logger.log(severity: .info, message: "Mixer stopped")
            metricsCollector.trackEvent(name: "mixer_stopped")
            
        } catch {
            crashReporter.recordError(error, context: ["action": "stop_mixer"])
            throw error
        }
    }
    
    // MARK: - Real-time Updates
    private func startRealTimeUpdates() {
        // Determine update frequency based on battery mode and network quality
        let updateInterval = getOptimalUpdateInterval()
        
        updateTimer?.invalidate()
        updateTimer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            Task { [weak self] in
                await self?.fetchMetrics()
            }
        }
        
        // Start SSE connection for real-time events
        if featureFlags.isEnabled("realtime_sse") && subscriptionTier != .free {
            startEventStream()
        }
        
        logger.log(
            severity: .info,
            message: "Started real-time updates",
            labels: ["interval": String(updateInterval), "sse": String(featureFlags.isEnabled("realtime_sse"))]
        )
    }
    
    private func getOptimalUpdateInterval() -> TimeInterval {
        // Adjust based on battery mode
        var interval: TimeInterval = batteryMode.updateInterval
        
        // Further adjust based on network quality
        if connectionStatus.isExpensive || connectionStatus.isConstrained {
            interval *= 2 // Reduce frequency on expensive connections
        }
        
        // Apply subscription limits
        if subscriptionTier == .free {
            interval = max(interval, 2.0) // Minimum 2 second interval for free tier
        }
        
        return interval
    }
    
    private func fetchMetrics() async {
        let span = distributedTracing.startSpan(name: "fetch_metrics")
        defer { distributedTracing.endSpan(span) }
        
        do {
            let url = URL(string: "\(baseURL)/api/status")!
            let (data, _) = try await session.data(from: url)
            
            if let metrics = try? JSONDecoder().decode(AudioMetrics.self, from: data) {
                await MainActor.run {
                    self.audioMetrics = metrics
                    
                    // Record audio quality SLI
                    let quality = calculateAudioQuality(metrics)
                    sreObservability.recordSLI(name: "audio_quality", value: quality)
                    
                    // Check for anomalies
                    if metrics.clipping {
                        alertManager.fireAlert(
                            severity: .warning,
                            message: "Audio clipping detected",
                            labels: ["channel": "master"]
                        )
                    }
                }
            }
        } catch {
            logger.log(severity: .error, message: "Failed to fetch metrics", labels: ["error": error.localizedDescription])
        }
    }
    
    private func calculateAudioQuality(_ metrics: AudioMetrics) -> Double {
        var quality = 1.0
        
        // Deduct for clipping
        if metrics.clipping { quality -= 0.3 }
        
        // Deduct for extreme levels
        if metrics.leftLevel > 0.95 || metrics.rightLevel > 0.95 { quality -= 0.1 }
        if metrics.leftLevel < 0.05 && metrics.rightLevel < 0.05 { quality -= 0.1 }
        
        return max(0.0, quality)
    }
    
    // MARK: - Low Data Mode
    private func enterLowDataMode() {
        batteryMode = .aggressive
        stopEventStream()
        
        logger.log(severity: .info, message: "Entered low data mode")
        metricsCollector.trackEvent(name: "low_data_mode_activated")
    }
    
    // MARK: - Event Stream
    private func startEventStream() {
        guard let url = URL(string: "\(baseURL)/api/stream") else { return }
        
        let request = URLRequest(url: url)
        eventSource = session.dataTask(with: request) { [weak self] data, response, error in
            guard let data = data, error == nil else { return }
            
            // Process SSE data
            if let eventString = String(data: data, encoding: .utf8) {
                self?.processServerSentEvent(eventString)
            }
        }
        eventSource?.resume()
    }
    
    private func stopEventStream() {
        eventSource?.cancel()
        eventSource = nil
    }
    
    private func processServerSentEvent(_ event: String) {
        // Parse and process SSE events
        logger.log(severity: .debug, message: "Received SSE event", labels: ["event": event])
    }
    
    private func stopRealTimeUpdates() {
        updateTimer?.invalidate()
        updateTimer = nil
        stopEventStream()
    }
    
    private func getConnectionType(from path: NWPath) -> String {
        if path.usesInterfaceType(.wifi) {
            return "wifi"
        } else if path.usesInterfaceType(.cellular) {
            return "cellular"
        } else if path.usesInterfaceType(.wiredEthernet) {
            return "ethernet"
        } else {
            return "unknown"
        }
    }
    
    deinit {
        updateTimer?.invalidate()
        eventSource?.cancel()
        monitor.cancel()
    }
}

// MARK: - MixerError Extension
extension MixerError {
    static let featureDisabled = MixerError.custom("Feature disabled by flag")
}