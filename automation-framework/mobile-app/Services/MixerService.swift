import Foundation
import Combine
import Network

// MARK: - Battery-Optimized Mixer Service
@MainActor
class MixerService: ObservableObject {
    // MARK: - Published Properties
    @Published var audioMetrics = AudioMetrics()
    @Published var mixerSettings = MixerSettings()
    @Published var connectionStatus = ConnectionStatus()
    @Published var isLoading = false
    @Published var availableDevices: [AudioDevice] = []
    @Published var logs: [LogEntry] = []
    
    // MARK: - Private Properties
    private var configuration: MixerConfiguration
    private var cancellables = Set<AnyCancellable>()
    private var metricsTimer: Timer?
    private var reconnectTimer: Timer?
    private let networkMonitor = NWPathMonitor()
    private let networkQueue = DispatchQueue(label: "NetworkMonitor")
    private var urlSession: URLSession
    
    // Battery optimization
    private var isBackgroundMode = false
    private var lastUpdateTime = Date()
    
    init(configuration: MixerConfiguration) {
        self.configuration = configuration
        
        // Configure URLSession for battery optimization
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 5.0
        config.timeoutIntervalForResource = 10.0
        config.allowsCellularAccess = true
        // Support both HTTP for local and HTTPS for production
        config.tlsMinimumSupportedProtocolVersion = .TLSv12
        self.urlSession = URLSession(configuration: config)
        
        setupNetworkMonitoring()
        
        if configuration.isAutoConnectEnabled {
            startMonitoring()
        }
    }
    
    deinit {
        stopMonitoring()
        networkMonitor.cancel()
    }
    
    // MARK: - Public API
    func updateConfiguration(_ newConfig: MixerConfiguration) {
        self.configuration = newConfig
        
        if newConfig.isAutoConnectEnabled && !isMonitoring {
            startMonitoring()
        } else if !newConfig.isAutoConnectEnabled && isMonitoring {
            stopMonitoring()
        }
    }
    
    func startMixer() async -> Result<Void, MixerError> {
        guard configuration.isConfigured else {
            return .failure(.notConfigured)
        }
        
        // Check subscription limits for concurrent streams
        guard await checkSubscriptionLimits() else {
            return .failure(.subscriptionRequired("Multiple concurrent streams"))
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            let url = URL(string: "\(configuration.serverURL)/api/start")!
            let (_, response) = try await urlSession.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw MixerError.connectionFailed("Start failed")
            }
            
            log(.success, "Mixer started successfully")
            await refreshStatus()
            return .success(())
            
        } catch {
            let mixerError = MixerError.connectionFailed(error.localizedDescription)
            log(.error, mixerError.localizedDescription ?? "Unknown error")
            return .failure(mixerError)
        }
    }
    
    func stopMixer() async -> Result<Void, MixerError> {
        guard configuration.isConfigured else {
            return .failure(.notConfigured)
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            let url = URL(string: "\(configuration.serverURL)/api/stop")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            
            let (_, response) = try await urlSession.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw MixerError.connectionFailed("Stop failed")
            }
            
            log(.success, "Mixer stopped successfully")
            await refreshStatus()
            return .success(())
            
        } catch {
            let mixerError = MixerError.connectionFailed(error.localizedDescription)
            log(.error, mixerError.localizedDescription ?? "Unknown error")
            return .failure(mixerError)
        }
    }
    
    func updateSettings(_ settings: MixerSettings) async -> Result<Void, MixerError> {
        guard configuration.isConfigured else {
            return .failure(.notConfigured)
        }
        
        guard settings.isValid else {
            return .failure(.audioEngineError("Invalid settings"))
        }
        
        // Check if advanced features require subscription
        if configuration.subscriptionTier == .free && settings.targetLUFS != -14.0 {
            return .failure(.subscriptionRequired("Custom LUFS targeting"))
        }
        
        isLoading = true
        defer { isLoading = false }
        
        do {
            let url = URL(string: "\(configuration.serverURL)/api/config")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let payload: [String: Any] = [
                "ai_mix": settings.aiMix,
                "target_lufs": settings.targetLUFS,
                "blocksize": settings.blockSize,
                "samplerate": settings.sampleRate
            ]
            
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (_, response) = try await urlSession.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw MixerError.connectionFailed("Config update failed")
            }
            
            self.mixerSettings = settings
            log(.success, "Settings updated successfully")
            return .success(())
            
        } catch {
            let mixerError = MixerError.connectionFailed(error.localizedDescription)
            log(.error, mixerError.localizedDescription ?? "Unknown error")
            return .failure(mixerError)
        }
    }
    
    func testConnection() async -> (success: Bool, latency: TimeInterval?) {
        guard configuration.isConfigured else {
            return (false, nil)
        }
        
        let startTime = Date()
        
        do {
            // Support both HTTP for local development and HTTPS for production
            var urlString = configuration.serverURL
            if !urlString.hasPrefix("http://") && !urlString.hasPrefix("https://") {
                urlString = "https://\(urlString)"
            }
            let url = URL(string: "\(urlString)/healthz")!
            let (_, response) = try await urlSession.data(from: url)
            
            let latency = Date().timeIntervalSince(startTime)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                return (false, nil)
            }
            
            connectionStatus = ConnectionStatus(
                isConnected: true,
                latency: latency,
                lastUpdate: Date()
            )
            
            log(.success, String(format: "Connection test successful (%.0fms)", latency * 1000))
            return (true, latency)
            
        } catch {
            connectionStatus = ConnectionStatus(
                isConnected: false,
                latency: nil,
                lastUpdate: Date()
            )
            
            log(.warning, "Connection test failed: \(error.localizedDescription)")
            return (false, nil)
        }
    }
    
    func refreshStatus() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.fetchAudioMetrics()
            }
            
            group.addTask {
                await self.fetchMixerSettings()
            }
            
            group.addTask {
                _ = await self.testConnection()
            }
        }
    }
    
    // MARK: - Battery Optimization
    func enterBackgroundMode() {
        isBackgroundMode = true
        
        // Reduce update frequency in background
        if configuration.subscriptionTier.batteryOptimization == .aggressive {
            stopMonitoring()
        } else {
            // Reduce frequency but keep monitoring
            setupMetricsTimer(interval: configuration.subscriptionTier.batteryOptimization.updateInterval * 3)
        }
        
        log(.info, "Entered background mode - reduced monitoring")
    }
    
    func enterForegroundMode() {
        isBackgroundMode = false
        
        // Restore full monitoring
        if configuration.isAutoConnectEnabled {
            startMonitoring()
        }
        
        // Immediate refresh when returning to foreground
        Task {
            await refreshStatus()
        }
        
        log(.info, "Entered foreground mode - full monitoring restored")
    }
    
    // MARK: - Private Implementation
    private var isMonitoring: Bool {
        metricsTimer != nil
    }
    
    private func startMonitoring() {
        guard !isMonitoring else { return }
        
        let interval = isBackgroundMode ? 
            configuration.subscriptionTier.batteryOptimization.updateInterval * 3 :
            configuration.subscriptionTier.batteryOptimization.updateInterval
            
        setupMetricsTimer(interval: interval)
        log(.info, "Started monitoring at \(String(format: "%.1f", 1.0/interval))Hz")
    }
    
    private func stopMonitoring() {
        metricsTimer?.invalidate()
        metricsTimer = nil
        reconnectTimer?.invalidate()
        reconnectTimer = nil
        
        log(.info, "Stopped monitoring")
    }
    
    private func setupMetricsTimer(interval: TimeInterval) {
        metricsTimer?.invalidate()
        
        metricsTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
            Task { @MainActor in
                await self.fetchAudioMetrics()
                
                // Periodic connection check
                if Date().timeIntervalSince(self.lastUpdateTime) > 10.0 {
                    _ = await self.testConnection()
                }
            }
        }
    }
    
    private func fetchAudioMetrics() async {
        guard configuration.isConfigured else { return }
        
        do {
            let url = URL(string: "\(configuration.serverURL)/api/status")!
            let (data, response) = try await urlSession.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                handleConnectionError()
                return
            }
            
            let apiResponse = try JSONDecoder().decode(APIStatusResponse.self, from: data)
            
            audioMetrics = AudioMetrics(
                rmsDB: apiResponse.metrics.rms_db,
                peakDB: apiResponse.metrics.peak_db,
                lufsEst: apiResponse.metrics.lufs_est,
                isClipping: apiResponse.metrics.clipping,
                dropouts: apiResponse.metrics.dropouts,
                deviceIn: apiResponse.metrics.device_in,
                deviceOut: apiResponse.metrics.device_out,
                isRunning: apiResponse.metrics.running,
                error: apiResponse.metrics.err,
                timestamp: Date()
            )
            
            lastUpdateTime = Date()
            
            // Update connection status
            if !connectionStatus.isConnected {
                connectionStatus = ConnectionStatus(
                    isConnected: true,
                    latency: connectionStatus.latency,
                    lastUpdate: Date()
                )
            }
            
        } catch {
            handleConnectionError()
        }
    }
    
    private func fetchMixerSettings() async {
        guard configuration.isConfigured else { return }
        
        do {
            let url = URL(string: "\(configuration.serverURL)/api/status")!
            let (data, _) = try await urlSession.data(from: url)
            let apiResponse = try JSONDecoder().decode(APIStatusResponse.self, from: data)
            
            mixerSettings = MixerSettings(
                aiMix: apiResponse.config.ai_mix,
                targetLUFS: apiResponse.config.target_lufs,
                blockSize: apiResponse.config.blocksize,
                sampleRate: apiResponse.config.samplerate
            )
            
        } catch {
            log(.warning, "Failed to fetch mixer settings: \(error.localizedDescription)")
        }
    }
    
    private func handleConnectionError() {
        if connectionStatus.isConnected {
            connectionStatus = ConnectionStatus(
                isConnected: false,
                latency: nil,
                lastUpdate: Date()
            )
            
            log(.error, "Lost connection to mixer")
            
            // Auto-reconnect after delay
            scheduleReconnect()
        }
    }
    
    private func scheduleReconnect() {
        guard reconnectTimer == nil else { return }
        
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            Task { @MainActor in
                self.reconnectTimer = nil
                let (success, _) = await self.testConnection()
                
                if success {
                    self.log(.success, "Reconnected to mixer")
                } else {
                    self.scheduleReconnect() // Try again
                }
            }
        }
    }
    
    private func setupNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                guard let self = self else { return }
                
                if path.status == .satisfied && !self.connectionStatus.isConnected {
                    Task {
                        _ = await self.testConnection()
                    }
                } else if path.status != .satisfied {
                    self.connectionStatus = ConnectionStatus(
                        isConnected: false,
                        latency: nil,
                        lastUpdate: Date()
                    )
                }
            }
        }
        
        networkMonitor.start(queue: networkQueue)
    }
    
    private func checkSubscriptionLimits() async -> Bool {
        // In a real app, this would check with subscription service
        // For now, just validate based on tier
        return configuration.subscriptionTier.maxConcurrentStreams > 0
    }
    
    private func log(_ level: LogEntry.Level, _ message: String) {
        let entry = LogEntry(timestamp: Date(), level: level, message: message)
        logs.append(entry)
        
        // Battery optimization: Limit log retention
        let maxLogs = configuration.subscriptionTier.batteryOptimization == .aggressive ? 50 : 100
        if logs.count > maxLogs {
            logs.removeFirst(logs.count - maxLogs)
        }
    }
}

// MARK: - API Response Models
private struct APIStatusResponse: Codable {
    let metrics: APIMetrics
    let config: APIConfig
}

private struct APIMetrics: Codable {
    let rms_db: Float
    let peak_db: Float
    let lufs_est: Float
    let clipping: Bool
    let dropouts: Int
    let device_in: String?
    let device_out: String?
    let running: Bool
    let err: String?
}

private struct APIConfig: Codable {
    let ai_mix: Float
    let target_lufs: Float
    let blocksize: Int
    let samplerate: Int
}