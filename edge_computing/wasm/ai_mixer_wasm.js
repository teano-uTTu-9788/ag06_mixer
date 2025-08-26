/**
 * AI Mixer WebAssembly JavaScript Interface
 * 
 * High-level JavaScript API for the AI Mixer WebAssembly module.
 * Provides WebAudio API integration and real-time processing.
 * 
 * Features:
 * - WebAudio API integration with AudioWorklet
 * - Real-time audio processing
 * - Genre detection and visualization
 * - Performance monitoring
 */

class AIMixerWASM {
    constructor() {
        this.wasmModule = null;
        this.mixer = null;
        this.audioContext = null;
        this.audioWorklet = null;
        this.isInitialized = false;
        this.isProcessing = false;
        
        // Audio configuration
        this.sampleRate = 48000;
        this.frameSize = 960; // 20ms at 48kHz
        this.featureSize = 13;
        
        // Performance monitoring
        this.stats = {
            framesProcessed: 0,
            avgProcessingTime: 0,
            peakProcessingTime: 0,
            currentGenre: 'UNKNOWN',
            confidence: 0
        };
        
        // Event callbacks
        this.onGenreDetected = null;
        this.onError = null;
        this.onMetricsUpdated = null;
    }
    
    /**
     * Initialize the AI Mixer with WebAssembly module
     * @param {string} wasmPath - Path to the WebAssembly file
     * @param {Object} config - DSP configuration
     */
    async initialize(wasmPath = './ai_mixer_wasm.wasm', config = null) {
        try {
            // Load WebAssembly module
            this.wasmModule = await this.loadWASM(wasmPath);
            
            // Create mixer instance
            this.mixer = new this.wasmModule.AIMixerWASM();
            
            // Initialize with configuration
            const dspConfig = config || this.getDefaultConfig();
            if (!this.mixer.initialize(dspConfig)) {
                throw new Error('Failed to initialize AI Mixer');
            }
            
            // Initialize Web Audio API
            await this.initializeAudioContext();
            
            this.isInitialized = true;
            console.log('AI Mixer WebAssembly initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize AI Mixer:', error);
            if (this.onError) {
                this.onError(error);
            }
            throw error;
        }
    }
    
    /**
     * Load WebAssembly module
     */
    async loadWASM(wasmPath) {
        try {
            // Check if Module is already loaded (from Emscripten)
            if (typeof Module !== 'undefined' && Module.ready) {
                await Module.ready;
                return Module;
            }
            
            // Load module dynamically
            const wasmModule = await import(wasmPath);
            await wasmModule.ready;
            return wasmModule;
            
        } catch (error) {
            console.error('Failed to load WASM module:', error);
            throw error;
        }
    }
    
    /**
     * Initialize Web Audio API context
     */
    async initializeAudioContext() {
        try {
            // Create AudioContext with optimal settings
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate,
                latencyHint: 'interactive' // Optimize for low latency
            });
            
            // Wait for user activation if needed
            if (this.audioContext.state === 'suspended') {
                console.log('AudioContext suspended - waiting for user activation');
                // Will be resumed when user interacts
            }
            
            // Load AudioWorklet processor
            await this.audioContext.audioWorklet.addModule('./ai-mixer-processor.js');
            
            console.log('Web Audio API initialized');
            
        } catch (error) {
            console.error('Failed to initialize Web Audio API:', error);
            throw error;
        }
    }
    
    /**
     * Start real-time audio processing
     */
    async startProcessing() {
        if (!this.isInitialized) {
            throw new Error('AI Mixer not initialized');
        }
        
        if (this.isProcessing) {
            console.log('Already processing');
            return;
        }
        
        try {
            // Resume AudioContext if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            // Get user media
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: false,
                    autoGainControl: false,
                    noiseSuppression: false
                }
            });
            
            // Create audio nodes
            const inputNode = this.audioContext.createMediaStreamSource(stream);
            const outputNode = this.audioContext.createGain();
            
            // Create AudioWorklet node for processing
            this.audioWorklet = new AudioWorkletNode(this.audioContext, 'ai-mixer-processor', {
                processorOptions: {
                    frameSize: this.frameSize
                }
            });
            
            // Set up message handling
            this.audioWorklet.port.onmessage = (event) => {
                this.handleWorkletMessage(event.data);
            };
            
            // Connect audio graph
            inputNode.connect(this.audioWorklet);
            this.audioWorklet.connect(outputNode);
            outputNode.connect(this.audioContext.destination);
            
            this.isProcessing = true;
            console.log('Started real-time audio processing');
            
        } catch (error) {
            console.error('Failed to start processing:', error);
            if (this.onError) {
                this.onError(error);
            }
            throw error;
        }
    }
    
    /**
     * Stop real-time audio processing
     */
    stopProcessing() {
        if (!this.isProcessing) {
            return;
        }
        
        try {
            if (this.audioWorklet) {
                this.audioWorklet.disconnect();
                this.audioWorklet = null;
            }
            
            this.isProcessing = false;
            console.log('Stopped real-time audio processing');
            
        } catch (error) {
            console.error('Failed to stop processing:', error);
        }
    }
    
    /**
     * Process single audio buffer (manual mode)
     * @param {Float32Array} inputBuffer - Input audio samples
     * @returns {Object} - {outputBuffer, metadata}
     */
    processBuffer(inputBuffer) {
        if (!this.isInitialized) {
            throw new Error('AI Mixer not initialized');
        }
        
        if (inputBuffer.length !== this.frameSize) {
            throw new Error(`Invalid buffer size: expected ${this.frameSize}, got ${inputBuffer.length}`);
        }
        
        try {
            // Create output buffer
            const outputBuffer = new Float32Array(this.frameSize);
            
            // Convert to JavaScript arrays for WASM interface
            const inputArray = Array.from(inputBuffer);
            const outputArray = new Array(this.frameSize);
            
            // Process through WebAssembly
            const metadata = this.mixer.processFrame(inputArray, outputArray);
            
            // Convert back to Float32Array
            outputBuffer.set(outputArray);
            
            // Update statistics
            this.updateStats(metadata);
            
            return { outputBuffer, metadata };
            
        } catch (error) {
            console.error('Failed to process buffer:', error);
            throw error;
        }
    }
    
    /**
     * Update DSP configuration
     * @param {Object} config - DSP configuration object
     */
    updateConfiguration(config) {
        if (!this.isInitialized) {
            throw new Error('AI Mixer not initialized');
        }
        
        try {
            this.mixer.updateConfig(config);
            
            // Send config to AudioWorklet if running
            if (this.audioWorklet) {
                this.audioWorklet.port.postMessage({
                    type: 'updateConfig',
                    config: config
                });
            }
            
        } catch (error) {
            console.error('Failed to update configuration:', error);
            throw error;
        }
    }
    
    /**
     * Extract audio features without processing
     * @param {Float32Array} inputBuffer - Input audio samples
     * @returns {Float32Array} - Feature vector
     */
    extractFeatures(inputBuffer) {
        if (!this.isInitialized) {
            throw new Error('AI Mixer not initialized');
        }
        
        if (inputBuffer.length !== this.frameSize) {
            throw new Error(`Invalid buffer size: expected ${this.frameSize}, got ${inputBuffer.length}`);
        }
        
        try {
            const inputArray = Array.from(inputBuffer);
            const featuresArray = this.mixer.extractFeatures(inputArray);
            return new Float32Array(featuresArray);
            
        } catch (error) {
            console.error('Failed to extract features:', error);
            throw error;
        }
    }
    
    /**
     * Classify genre from features
     * @param {Float32Array} features - Feature vector
     * @returns {string} - Genre name
     */
    classifyGenre(features) {
        if (!this.isInitialized) {
            throw new Error('AI Mixer not initialized');
        }
        
        if (features.length !== this.featureSize) {
            throw new Error(`Invalid feature size: expected ${this.featureSize}, got ${features.length}`);
        }
        
        try {
            const featuresArray = Array.from(features);
            const genre = this.mixer.classifyGenre(featuresArray);
            return this.genreToString(genre);
            
        } catch (error) {
            console.error('Failed to classify genre:', error);
            return 'UNKNOWN';
        }
    }
    
    /**
     * Get performance statistics
     * @returns {Object} - Performance metrics
     */
    getStats() {
        const wasmStats = this.mixer ? this.mixer.getStats() : {};
        
        return {
            ...this.stats,
            wasmFrameCount: wasmStats.frameCount || 0,
            sampleRate: this.sampleRate,
            frameSize: this.frameSize,
            isInitialized: this.isInitialized,
            isProcessing: this.isProcessing
        };
    }
    
    /**
     * Get default DSP configuration
     */
    getDefaultConfig() {
        return {
            // Noise Gate
            gateThresholdDB: -50.0,
            gateRatio: 4.0,
            gateAttackMS: 1.0,
            gateReleaseMS: 100.0,
            
            // Compressor
            compThresholdDB: -18.0,
            compRatio: 3.0,
            compAttackMS: 5.0,
            compReleaseMS: 50.0,
            compKneeDB: 2.0,
            
            // EQ
            eqLowGainDB: 0.0,
            eqLowFreq: 100.0,
            eqMidGainDB: 0.0,
            eqMidFreq: 1000.0,
            eqHighGainDB: 0.0,
            eqHighFreq: 8000.0,
            
            // Limiter
            limiterThresholdDB: -3.0,
            limiterReleaseMS: 10.0
        };
    }
    
    /**
     * Handle messages from AudioWorklet
     */
    handleWorkletMessage(data) {
        switch (data.type) {
            case 'metadata':
                this.updateStats(data.metadata);
                
                // Trigger genre detection callback
                if (this.onGenreDetected && data.metadata.detectedGenre !== this.stats.currentGenre) {
                    this.onGenreDetected(
                        this.genreToString(data.metadata.detectedGenre),
                        data.metadata.confidence
                    );
                }
                break;
                
            case 'error':
                console.error('AudioWorklet error:', data.error);
                if (this.onError) {
                    this.onError(new Error(data.error));
                }
                break;
                
            case 'metrics':
                if (this.onMetricsUpdated) {
                    this.onMetricsUpdated(data.metrics);
                }
                break;
        }
    }
    
    /**
     * Update internal statistics
     */
    updateStats(metadata) {
        this.stats.framesProcessed++;
        
        // Update processing time averages
        const alpha = 0.1; // Smoothing factor
        this.stats.avgProcessingTime = this.stats.avgProcessingTime * (1 - alpha) + metadata.processingTimeMS * alpha;
        this.stats.peakProcessingTime = Math.max(this.stats.peakProcessingTime, metadata.processingTimeMS);
        
        // Update current genre and confidence
        this.stats.currentGenre = this.genreToString(metadata.detectedGenre);
        this.stats.confidence = metadata.confidence;
    }
    
    /**
     * Convert genre enum to string
     */
    genreToString(genre) {
        const genreMap = {
            0: 'SPEECH',
            1: 'ROCK', 
            2: 'JAZZ',
            3: 'ELECTRONIC',
            4: 'CLASSICAL',
            5: 'UNKNOWN'
        };
        
        return genreMap[genre] || 'UNKNOWN';
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        try {
            this.stopProcessing();
            
            if (this.audioContext) {
                this.audioContext.close();
                this.audioContext = null;
            }
            
            if (this.mixer) {
                // WebAssembly cleanup is handled automatically
                this.mixer = null;
            }
            
            this.isInitialized = false;
            console.log('AI Mixer destroyed');
            
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }
}

// AudioWorklet processor for real-time processing
const AUDIO_WORKLET_PROCESSOR = `
class AIMixerProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        
        this.frameSize = options.processorOptions?.frameSize || 960;
        this.inputBuffer = new Float32Array(this.frameSize);
        this.outputBuffer = new Float32Array(this.frameSize);
        this.bufferIndex = 0;
        
        // Will be set when WASM is available
        this.mixer = null;
        this.isReady = false;
        
        // Listen for messages from main thread
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };
        
        // Request WASM instance from main thread
        this.port.postMessage({ type: 'requestWasm' });
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'setWasm':
                this.mixer = data.mixer;
                this.isReady = true;
                break;
                
            case 'updateConfig':
                if (this.mixer) {
                    this.mixer.updateConfig(data.config);
                }
                break;
        }
    }
    
    process(inputs, outputs) {
        const input = inputs[0];
        const output = outputs[0];
        
        if (!input || !input[0] || !this.isReady) {
            return true;
        }
        
        const inputChannel = input[0];
        const outputChannel = output[0];
        
        // Accumulate samples until we have a full frame
        for (let i = 0; i < inputChannel.length; i++) {
            this.inputBuffer[this.bufferIndex] = inputChannel[i];
            this.bufferIndex++;
            
            if (this.bufferIndex >= this.frameSize) {
                // Process full frame
                try {
                    const inputArray = Array.from(this.inputBuffer);
                    const outputArray = new Array(this.frameSize);
                    
                    const metadata = this.mixer.processFrame(inputArray, outputArray);
                    
                    this.outputBuffer.set(outputArray);
                    
                    // Send metadata back to main thread
                    this.port.postMessage({
                        type: 'metadata',
                        metadata: metadata
                    });
                    
                } catch (error) {
                    this.port.postMessage({
                        type: 'error',
                        error: error.message
                    });
                }
                
                this.bufferIndex = 0;
            }
            
            // Output processed audio (with buffering consideration)
            const outputIndex = Math.min(i, this.outputBuffer.length - 1);
            outputChannel[i] = this.outputBuffer[outputIndex];
        }
        
        return true;
    }
}

registerProcessor('ai-mixer-processor', AIMixerProcessor);
`;

// Export AudioWorklet processor as blob URL
const createAudioWorkletURL = () => {
    const blob = new Blob([AUDIO_WORKLET_PROCESSOR], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
};

// Module exports
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AIMixerWASM, createAudioWorkletURL };
} else if (typeof window !== 'undefined') {
    window.AIMixerWASM = AIMixerWASM;
    window.createAudioWorkletURL = createAudioWorkletURL;
}