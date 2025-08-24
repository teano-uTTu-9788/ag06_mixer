# Mobile SDKs for AI Mixing Studio

## Overview

Cross-platform mobile SDKs providing native iOS and Android access to the AI-powered real-time audio mixing capabilities. Built on a shared C++ core for consistent performance across platforms while providing native language bindings and platform-specific optimizations.

## Architecture

```
┌─────────────────────────────────────┐
│           Mobile Applications       │
├─────────────────┬───────────────────┤
│     iOS SDK     │    Android SDK    │
│   (Swift API)   │   (Kotlin API)    │
├─────────────────┼───────────────────┤
│         Shared C++ Core             │
│      (ai_mixer_core.h/cpp)         │
├─────────────────────────────────────┤
│        AI Mixing Engine             │
│   • Genre Detection                 │
│   • DSP Processing                  │
│   • Feature Extraction             │
│   • Performance Monitoring         │
└─────────────────────────────────────┘
```

## Features

### 🎯 Cross-Platform Consistency
- **Unified API**: Consistent interface across iOS and Android
- **Shared Core**: Same C++ engine ensures identical processing results
- **Platform Optimizations**: Native iOS/Android integrations for best performance

### 🎵 Professional Audio Processing
- **Real-Time DSP**: <20ms latency with professional audio chain
- **AI Genre Detection**: Speech, Rock, Jazz, Electronic, Classical classification
- **Adaptive Mixing**: Genre-specific parameter adjustments
- **Studio Effects**: Noise Gate, Compressor, EQ, Limiter

### 📱 Mobile-Optimized
- **Battery Efficient**: Optimized for mobile power consumption
- **Background Processing**: Continuous processing support
- **Hardware Integration**: AVAudioEngine (iOS), AudioTrack/Record (Android)
- **Modern Async**: Swift async/await, Kotlin coroutines

## Quick Start

### iOS (Swift)

```swift
import AIMixerSDK

class AudioViewController: UIViewController, AIMixerDelegate {
    private var mixer: AIMixerSDK!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Initialize mixer
        mixer = AIMixerSDK()
        mixer.delegate = self
        
        Task {
            try await mixer.initialize()
            try await mixer.startProcessing()
        }
    }
    
    // Delegate methods
    func mixerDidDetectGenre(_ genre: Genre, confidence: Float) {
        print("Detected genre: \\(genre) (\\(confidence))")
    }
    
    func mixerDidEncounterError(_ error: AIMixerError) {
        print("Mixer error: \\(error)")
    }
}
```

### Android (Kotlin)

```kotlin
class AudioActivity : AppCompatActivity(), AIMixerCallback {
    private lateinit var mixer: AIMixerSDK
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        lifecycleScope.launch {
            try {
                mixer = this@AudioActivity.createAIMixer()
                    .withCallback(this@AudioActivity)
                    .build()
                    
                mixer.startProcessing()
            } catch (e: AIMixerError) {
                Log.e("AudioActivity", "Mixer error", e)
            }
        }
    }
    
    // Callback methods
    override fun onGenreDetected(genre: Genre, confidence: Float) {
        Log.i("AudioActivity", "Detected genre: $genre ($confidence)")
    }
    
    override fun onError(error: AIMixerError) {
        Log.e("AudioActivity", "Mixer error: $error")
    }
}
```

## API Reference

### Common Types

#### Genre Enumeration
```
SPEECH = 0      // Voice, podcasts, spoken content
ROCK = 1        // Rock, metal, punk music
JAZZ = 2        // Jazz, blues, swing music  
ELECTRONIC = 3  // EDM, techno, synthetic music
CLASSICAL = 4   // Classical, orchestral music
UNKNOWN = 5     // Unclassified or mixed content
```

#### DSP Configuration
```swift
// iOS Swift
struct DSPConfiguration {
    var gateThresholdDB: Float = -50.0
    var gateRatio: Float = 4.0
    var compThresholdDB: Float = -18.0
    var compRatio: Float = 3.0
    var eqLowGainDB: Float = 0.0
    var limiterThresholdDB: Float = -3.0
    // ... additional parameters
}
```

```kotlin
// Android Kotlin
data class DSPConfiguration(
    var gateThresholdDB: Float = -50.0f,
    var gateRatio: Float = 4.0f,
    var compThresholdDB: Float = -18.0f,
    var compRatio: Float = 3.0f,
    var eqLowGainDB: Float = 0.0f,
    var limiterThresholdDB: Float = -3.0f
    // ... additional parameters
)
```

### Core Methods

#### Initialization
- `initialize(config:)` - Initialize mixer with optional DSP configuration
- `shutdown()` - Clean up resources and stop processing

#### Audio Processing
- `startProcessing()` - Begin real-time audio processing
- `stopProcessing()` - Stop real-time audio processing
- `processBuffer(buffer:)` - Process single audio buffer (manual mode)

#### Configuration
- `updateConfiguration(config:)` - Update DSP settings at runtime
- `setManualGenre(genre:bypass:)` - Override AI detection with manual genre
- `loadCustomModel(data:)` - Load custom TensorFlow Lite model

#### Monitoring
- `getPerformanceMetrics()` - Get current performance statistics

### Performance Specifications

#### Audio Requirements
- **Sample Rate**: 48kHz (CD quality)
- **Frame Size**: 960 samples (20ms)
- **Channels**: Mono/Stereo support
- **Bit Depth**: 32-bit float internal processing

#### Performance Targets
- **Latency**: <20ms processing time
- **CPU Usage**: <25% on mid-range devices
- **Memory**: <100MB total footprint
- **Battery**: Optimized for continuous processing

#### Mobile Optimizations
- **iOS**: AVAudioEngine integration, Core Audio optimizations
- **Android**: AudioTrack/Record integration, AAudio support
- **Both**: Hardware acceleration detection, adaptive quality

## Integration Guide

### iOS Integration

#### Xcode Project Setup
1. **Add Files**: Import SDK files to your Xcode project
2. **Build Settings**: Add C++ compilation support
3. **Frameworks**: Link AVFoundation, Accelerate
4. **Permissions**: Add microphone usage description

```xml
<!-- Info.plist -->
<key>NSMicrophoneUsageDescription</key>
<string>AI Mixer needs microphone access for real-time audio processing</string>
```

#### Swift Package Manager
```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/your-org/aimixer-ios-sdk", from: "1.0.0")
]
```

### Android Integration

#### Gradle Setup
```kotlin
// app/build.gradle
android {
    compileSdk 34
    
    defaultConfig {
        minSdk 24  // Android 7.0+
        targetSdk 34
    }
    
    ndkVersion "25.1.8937393"
}

dependencies {
    implementation 'com.aimixer:sdk:1.0.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

#### Permissions
```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
```

### Native Library Build

#### iOS Build
```bash
# Build universal iOS library
xcodebuild -project AIMixerSDK.xcodeproj \
           -scheme AIMixerSDK \
           -configuration Release \
           -destination "generic/platform=iOS" \
           -archivePath AIMixerSDK-iOS.xcarchive \
           archive

# Create XCFramework
xcodebuild -create-xcframework \
           -archive AIMixerSDK-iOS.xcarchive -framework AIMixerSDK.framework \
           -archive AIMixerSDK-Simulator.xcarchive -framework AIMixerSDK.framework \
           -output AIMixerSDK.xcframework
```

#### Android Build
```bash
# Build native library with NDK
cd android/jni
ndk-build
```

## Testing

### Comprehensive Test Suite
```bash
# Run all mobile SDK tests
cd mobile_sdks
python test_mobile_sdks.py
```

**Test Categories:**
- Directory Structure ✅
- Shared Core Headers ✅  
- iOS SDK Structure ✅
- Android SDK Structure ✅
- API Consistency ✅
- Genre Enum Consistency ✅
- DSP Config Completeness ✅
- Audio Constants Consistency ✅
- Error Handling Completeness ✅
- Callback Interfaces ✅
- Performance Monitoring ✅
- Mobile Specific Features ⚠️
- Integration Readiness ✅

**Current Status: 12/13 tests passing (92.3%)**

### Device Testing Recommendations

#### iOS Testing
- **iPhone 12+**: Primary development target
- **iPad Air/Pro**: Tablet optimizations
- **iPhone SE**: Lower-end performance validation
- **TestFlight**: Beta distribution and crash reporting

#### Android Testing
- **Pixel 6+**: Primary development target  
- **Samsung Galaxy S21+**: Popular manufacturer
- **OnePlus/Xiaomi**: International market coverage
- **Mid-range devices**: Performance validation
- **Internal Testing**: Play Console beta distribution

## Deployment

### Production Checklist

#### Pre-Deployment
- [ ] All 13/13 tests passing
- [ ] Performance benchmarking on target devices
- [ ] Memory leak testing (24+ hour sessions)
- [ ] Battery usage optimization validation
- [ ] Crash reporting integration

#### iOS App Store
- [ ] Code signing certificates
- [ ] App Store Connect configuration
- [ ] Privacy manifest (iOS 17+)
- [ ] TestFlight beta testing
- [ ] App Review submission

#### Google Play Store
- [ ] App Bundle (.aab) generation
- [ ] Play App Signing
- [ ] Internal testing track
- [ ] Staged rollout (10% → 50% → 100%)
- [ ] Play Console crash reporting

### Monitoring & Analytics

#### Performance Metrics
- **Processing Latency**: Target <20ms, alert >30ms
- **CPU Usage**: Target <25%, alert >40%
- **Memory Usage**: Target <100MB, alert >150MB
- **Battery Drain**: Measure mAh per hour usage

#### User Experience Metrics
- **Genre Detection Accuracy**: Track user corrections
- **Crash Rate**: Target <0.1% sessions
- **ANR Rate**: Android target <0.1% sessions
- **Load Time**: Target <2s initialization

## Advanced Features

### Custom Model Integration
```swift
// iOS - Load custom TensorFlow Lite model
let modelURL = Bundle.main.url(forResource: "custom_genre_model", withExtension: "tflite")!
try await mixer.loadCustomModel(from: modelURL)
```

```kotlin
// Android - Load custom model from assets
val modelData = assets.open("custom_genre_model.tflite").readBytes()
mixer.loadCustomModel(modelData)
```

### Background Processing
```swift
// iOS - Enable background audio processing
try AVAudioSession.sharedInstance().setCategory(.playAndRecord, 
                                                mode: .default, 
                                                options: [.allowBluetooth, .defaultToSpeaker])
```

```kotlin
// Android - Foreground service for background processing
class AudioProcessingService : Service() {
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, createNotification())
        // Continue processing in background
        return START_STICKY
    }
}
```

### Real-Time Visualization
```swift
// iOS - Real-time audio visualization
func mixerDidUpdateMetrics(_ metrics: PerformanceMetrics) {
    DispatchQueue.main.async {
        self.updateVisualization(
            rmsLevel: metrics.rmsLevelDB,
            peakLevel: metrics.peakLevelDB,
            genre: metrics.detectedGenre
        )
    }
}
```

## Architecture Decisions

### Why C++ Core?
- **Performance**: Native code for real-time audio processing
- **Consistency**: Identical behavior across platforms
- **Optimization**: Hardware-specific optimizations
- **Integration**: Easy binding to mobile platforms

### Why Platform-Specific APIs?
- **Native Feel**: Platform conventions (delegate vs callback)
- **Modern Patterns**: async/await, coroutines
- **Framework Integration**: AVAudioEngine, AudioTrack
- **Developer Experience**: Familiar patterns for each platform

### Memory Management
- **iOS**: ARC (Automatic Reference Counting) with C++ bridge
- **Android**: Garbage collection with JNI lifecycle management
- **Shared**: RAII patterns in C++ core for deterministic cleanup

## Troubleshooting

### Common Issues

#### iOS
- **Audio Session Conflicts**: Configure audio session before initialization
- **Background Processing**: Enable background modes capability
- **Xcode Build Errors**: Ensure C++ compilation flags are set

#### Android
- **NDK Compatibility**: Use compatible NDK version (25.x)
- **Audio Permission**: Request RECORD_AUDIO permission at runtime
- **ProGuard Issues**: Add keep rules for native methods

#### Both Platforms
- **High CPU Usage**: Check frame size configuration (should be 960)
- **Audio Dropouts**: Increase buffer sizes for older devices
- **Memory Leaks**: Ensure proper shutdown() calls

### Performance Tuning

#### Optimization Strategies
1. **Reduce Frame Size**: 480 samples (10ms) for lower latency
2. **Hardware Acceleration**: Enable when available
3. **Thread Priorities**: Use real-time threads for audio processing
4. **Memory Pools**: Pre-allocate buffers to avoid GC pressure

#### Device-Specific Tuning
- **High-end devices**: Enable all features, lowest latency
- **Mid-range devices**: Balance features with performance
- **Low-end devices**: Simplified processing, higher latency

## License

Licensed under the MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_mobile_sdks.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## Support

- **Documentation**: [https://docs.aimixer.com/mobile-sdks](https://docs.aimixer.com/mobile-sdks)
- **GitHub Issues**: Report bugs and request features
- **Discord Community**: Real-time support and discussions
- **Email Support**: support@aimixer.com for enterprise customers