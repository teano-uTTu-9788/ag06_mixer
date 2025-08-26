import Foundation

// MARK: - AG06 Mixer Configuration Models
struct MixerConfiguration: Codable {
    var serverURL: String = "http://127.0.0.1:8080"
    var apiKey: String = ""
    var isAutoConnectEnabled: Bool = true
    var subscriptionTier: SubscriptionTier = .free
    
    var isConfigured: Bool {
        !serverURL.isEmpty
    }
}

// MARK: - Subscription Management
enum SubscriptionTier: String, CaseIterable, Codable {
    case free = "free"
    case pro = "pro"
    case studio = "studio"
    
    var displayName: String {
        switch self {
        case .free: return "Free"
        case .pro: return "Pro"
        case .studio: return "Studio"
        }
    }
    
    var maxConcurrentStreams: Int {
        switch self {
        case .free: return 1
        case .pro: return 4
        case .studio: return 16
        }
    }
    
    var hasAdvancedEQ: Bool {
        switch self {
        case .free: return false
        case .pro: return true
        case .studio: return true
        }
    }
    
    var hasAIProcessing: Bool {
        switch self {
        case .free: return false
        case .pro: return true
        case .studio: return true
        }
    }
    
    var batteryOptimization: BatteryMode {
        switch self {
        case .free: return .aggressive
        case .pro: return .balanced
        case .studio: return .performance
        }
    }
}

// MARK: - Battery Optimization
enum BatteryMode: String, CaseIterable, Codable {
    case aggressive = "aggressive"
    case balanced = "balanced"
    case performance = "performance"
    
    var displayName: String {
        switch self {
        case .aggressive: return "Battery Saver"
        case .balanced: return "Balanced"
        case .performance: return "Performance"
        }
    }
    
    var updateInterval: TimeInterval {
        switch self {
        case .aggressive: return 2.0  // 0.5Hz
        case .balanced: return 0.5    // 2Hz
        case .performance: return 0.1 // 10Hz
        }
    }
    
    var enableBackgroundProcessing: Bool {
        switch self {
        case .aggressive: return false
        case .balanced: return true
        case .performance: return true
        }
    }
}

// MARK: - Real-time Audio Metrics
struct AudioMetrics: Codable, Equatable {
    let rmsDB: Float
    let peakDB: Float
    let lufsEst: Float
    let isClipping: Bool
    let dropouts: Int
    let deviceIn: String?
    let deviceOut: String?
    let isRunning: Bool
    let error: String?
    let timestamp: Date
    
    init(
        rmsDB: Float = -60.0,
        peakDB: Float = -60.0,
        lufsEst: Float = -60.0,
        isClipping: Bool = false,
        dropouts: Int = 0,
        deviceIn: String? = nil,
        deviceOut: String? = nil,
        isRunning: Bool = false,
        error: String? = nil,
        timestamp: Date = Date()
    ) {
        self.rmsDB = rmsDB
        self.peakDB = peakDB
        self.lufsEst = lufsEst
        self.isClipping = isClipping
        self.dropouts = dropouts
        self.deviceIn = deviceIn
        self.deviceOut = deviceOut
        self.isRunning = isRunning
        self.error = error
        self.timestamp = timestamp
    }
}

// MARK: - Mixer Control Parameters
struct MixerSettings: Codable, Equatable {
    var aiMix: Float = 0.7          // 0.0-1.0
    var targetLUFS: Float = -14.0   // Target loudness
    var blockSize: Int = 256        // Audio block size
    var sampleRate: Int = 44100     // Sample rate
    
    var isValid: Bool {
        aiMix >= 0.0 && aiMix <= 1.0 &&
        targetLUFS >= -60.0 && targetLUFS <= 0.0 &&
        blockSize >= 64 && blockSize <= 2048 &&
        [22050, 44100, 48000, 96000].contains(sampleRate)
    }
}

// MARK: - Connection Status
struct ConnectionStatus {
    let isConnected: Bool
    let latency: TimeInterval?
    let lastUpdate: Date
    let serverVersion: String?
    
    init(
        isConnected: Bool = false,
        latency: TimeInterval? = nil,
        lastUpdate: Date = Date(),
        serverVersion: String? = nil
    ) {
        self.isConnected = isConnected
        self.latency = latency
        self.lastUpdate = lastUpdate
        self.serverVersion = serverVersion
    }
}

// MARK: - Audio Device Info
struct AudioDevice: Identifiable, Codable, Equatable {
    let id: Int
    let name: String
    let isInput: Bool
    let isOutput: Bool
    let isAG06: Bool
    
    var displayName: String {
        return isAG06 ? "🎚️ \(name)" : name
    }
}

// MARK: - Error Types
enum MixerError: Error, LocalizedError {
    case notConfigured
    case connectionFailed(String)
    case audioEngineError(String)
    case subscriptionRequired(String)
    case batteryOptimizationActive
    
    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "Mixer not configured"
        case .connectionFailed(let message):
            return "Connection failed: \(message)"
        case .audioEngineError(let message):
            return "Audio error: \(message)"
        case .subscriptionRequired(let feature):
            return "\(feature) requires Pro or Studio subscription"
        case .batteryOptimizationActive:
            return "Feature disabled for battery optimization"
        }
    }
}