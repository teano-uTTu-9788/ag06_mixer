/**
 * Audio Worklet Processor for Real-time Karaoke Processing
 * Runs in separate thread for low-latency audio processing
 */

class KaraokeProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    this.bufferSize = options.processorOptions?.bufferSize || 256;
    this.sampleRate = sampleRate; // Global AudioWorklet variable
    
    // Auto-tune parameters
    this.autoTuneEnabled = true;
    this.autoTuneStrength = 0.7;
    this.targetPitch = 440; // A4
    
    // Pitch detection
    this.autocorrelationBuffer = new Float32Array(this.bufferSize * 2);
    this.pitchHistory = [];
    this.maxPitchHistory = 10;
    
    // Level monitoring
    this.levelSmoothing = 0.95;
    this.currentLevel = 0;
    this.peakLevel = 0;
    this.peakHoldTime = 3000; // ms
    this.lastPeakTime = 0;
    
    // Clipping detection
    this.clippingThreshold = 0.99;
    this.clippingCount = 0;
    
    // Setup message handling
    this.port.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }
  
  handleMessage(data) {
    switch (data.type) {
      case 'update-effects':
        if (data.effects.autoTune) {
          this.autoTuneEnabled = data.effects.autoTune.enabled;
          this.autoTuneStrength = data.effects.autoTune.strength;
        }
        break;
      case 'reset-peak':
        this.peakLevel = 0;
        break;
    }
  }
  
  /**
   * Main processing function - called for each audio buffer
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    
    if (!input || !input[0]) {
      return true;
    }
    
    // Process each channel
    for (let channel = 0; channel < output.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];
      
      if (!inputChannel) {
        continue;
      }
      
      // Copy input to output
      outputChannel.set(inputChannel);
      
      // Apply processing
      this.processChannel(outputChannel, channel === 0);
    }
    
    return true; // Keep processor alive
  }
  
  processChannel(samples, isPrimaryChannel) {
    // Level monitoring (only on primary channel)
    if (isPrimaryChannel) {
      this.updateLevels(samples);
    }
    
    // Pitch detection and correction
    if (this.autoTuneEnabled && isPrimaryChannel) {
      const pitch = this.detectPitch(samples);
      if (pitch > 0) {
        this.applyPitchCorrection(samples, pitch);
      }
    }
    
    // Apply subtle harmonic enhancement
    this.applyHarmonicEnhancement(samples);
    
    // Soft clipping to prevent distortion
    this.applySoftClipping(samples);
    
    // Check for clipping
    this.detectClipping(samples);
  }
  
  /**
   * Detect pitch using autocorrelation
   */
  detectPitch(samples) {
    // Simple autocorrelation-based pitch detection
    const minPeriod = Math.floor(this.sampleRate / 800); // 800 Hz max
    const maxPeriod = Math.floor(this.sampleRate / 80);  // 80 Hz min
    
    let maxCorrelation = 0;
    let bestPeriod = 0;
    
    for (let period = minPeriod; period < maxPeriod; period++) {
      let correlation = 0;
      for (let i = 0; i < samples.length - period; i++) {
        correlation += samples[i] * samples[i + period];
      }
      
      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        bestPeriod = period;
      }
    }
    
    if (bestPeriod > 0 && maxCorrelation > 0.3) {
      const frequency = this.sampleRate / bestPeriod;
      
      // Smooth pitch detection
      this.pitchHistory.push(frequency);
      if (this.pitchHistory.length > this.maxPitchHistory) {
        this.pitchHistory.shift();
      }
      
      // Return median of recent pitches
      const sorted = [...this.pitchHistory].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      
      // Send pitch to main thread
      if (currentFrame % 2048 === 0) { // Throttle messages
        this.port.postMessage({
          type: 'pitch',
          frequency: median,
          note: this.frequencyToNote(median)
        });
      }
      
      return median;
    }
    
    return 0;
  }
  
  /**
   * Apply pitch correction
   */
  applyPitchCorrection(samples, detectedPitch) {
    // Find nearest musical note
    const targetPitch = this.getNearestNote(detectedPitch);
    
    if (Math.abs(detectedPitch - targetPitch) < 50) { // Within 50 cents
      const pitchRatio = targetPitch / detectedPitch;
      const correctionAmount = 1 + (pitchRatio - 1) * this.autoTuneStrength;
      
      // Simple pitch shift using phase vocoder technique
      // This is simplified - real implementation would use FFT
      if (correctionAmount !== 1) {
        this.applyPitchShift(samples, correctionAmount);
      }
    }
  }
  
  /**
   * Simplified pitch shifting
   */
  applyPitchShift(samples, ratio) {
    // This is a very simplified version
    // Real implementation would use PSOLA or phase vocoder
    const shifted = new Float32Array(samples.length);
    
    for (let i = 0; i < samples.length; i++) {
      const sourceIndex = i / ratio;
      const index = Math.floor(sourceIndex);
      const fraction = sourceIndex - index;
      
      if (index < samples.length - 1) {
        // Linear interpolation
        shifted[i] = samples[index] * (1 - fraction) + 
                    samples[index + 1] * fraction;
      } else {
        shifted[i] = samples[samples.length - 1];
      }
    }
    
    // Crossfade with original
    for (let i = 0; i < samples.length; i++) {
      samples[i] = samples[i] * (1 - this.autoTuneStrength) + 
                   shifted[i] * this.autoTuneStrength;
    }
  }
  
  /**
   * Add harmonic enhancement for brightness
   */
  applyHarmonicEnhancement(samples) {
    const enhancementAmount = 0.05; // Subtle enhancement
    
    for (let i = 0; i < samples.length; i++) {
      // Generate harmonics using soft clipping
      const enhanced = Math.tanh(samples[i] * 3) * enhancementAmount;
      samples[i] += enhanced;
    }
  }
  
  /**
   * Soft clipping to prevent harsh distortion
   */
  applySoftClipping(samples) {
    const threshold = 0.95;
    
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) > threshold) {
        samples[i] = Math.sign(samples[i]) * 
                     (threshold + (1 - threshold) * Math.tanh((Math.abs(samples[i]) - threshold) * 10));
      }
    }
  }
  
  /**
   * Update level meters
   */
  updateLevels(samples) {
    let sum = 0;
    let max = 0;
    
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      sum += abs * abs;
      if (abs > max) {
        max = abs;
      }
    }
    
    const rms = Math.sqrt(sum / samples.length);
    
    // Smooth RMS level
    this.currentLevel = this.currentLevel * this.levelSmoothing + 
                        rms * (1 - this.levelSmoothing);
    
    // Update peak
    if (max > this.peakLevel) {
      this.peakLevel = max;
      this.lastPeakTime = currentTime;
    } else if (currentTime - this.lastPeakTime > this.peakHoldTime) {
      this.peakLevel *= 0.95; // Slow decay
    }
    
    // Send level update (throttled)
    if (currentFrame % 512 === 0) {
      this.port.postMessage({
        type: 'level',
        level: this.currentLevel,
        peak: this.peakLevel
      });
    }
  }
  
  /**
   * Detect clipping
   */
  detectClipping(samples) {
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) > this.clippingThreshold) {
        this.clippingCount++;
        
        if (this.clippingCount > 10) {
          this.port.postMessage({ type: 'clipping' });
          this.clippingCount = 0;
        }
        break;
      }
    }
  }
  
  /**
   * Get nearest musical note frequency
   */
  getNearestNote(frequency) {
    const A4 = 440;
    const noteNumber = 12 * Math.log2(frequency / A4) + 49;
    const nearestNote = Math.round(noteNumber);
    return A4 * Math.pow(2, (nearestNote - 49) / 12);
  }
  
  /**
   * Convert frequency to note name
   */
  frequencyToNote(frequency) {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const A4 = 440;
    const noteNumber = Math.round(12 * Math.log2(frequency / A4) + 49);
    const noteName = notes[(noteNumber - 1) % 12];
    const octave = Math.floor((noteNumber + 8) / 12);
    return `${noteName}${octave}`;
  }
}

// Register the processor
registerProcessor('karaoke-processor', KaraokeProcessor);