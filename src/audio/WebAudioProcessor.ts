/**
 * Web Audio API Processor for Real-time Karaoke
 * Following Google Chrome team's best practices for Web Audio
 */

export interface AudioProcessorConfig {
  sampleRate: number;
  bufferSize: number;
  channels: number;
  latencyHint: 'interactive' | 'balanced' | 'playback';
}

export interface EffectsConfig {
  autoTune: {
    enabled: boolean;
    strength: number; // 0-1
    speed: number; // ms
    key: string;
    scale: 'major' | 'minor' | 'chromatic';
  };
  reverb: {
    enabled: boolean;
    mix: number; // 0-1
    roomSize: number; // 0-1
    damping: number; // 0-1
  };
  compression: {
    enabled: boolean;
    threshold: number; // dB
    ratio: number;
    attack: number; // ms
    release: number; // ms
  };
  eq: {
    enabled: boolean;
    lowCut: number; // Hz
    presence: { freq: number; gain: number };
    warmth: { freq: number; gain: number };
    air: { freq: number; gain: number };
  };
}

export class WebAudioProcessor {
  private audioContext: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  private convolver: ConvolverNode | null = null;
  private filters: Map<string, BiquadFilterNode> = new Map();
  private gainNode: GainNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private isProcessing = false;
  private config: AudioProcessorConfig;
  private effects: EffectsConfig;

  constructor(config?: Partial<AudioProcessorConfig>) {
    this.config = {
      sampleRate: 48000,
      bufferSize: 256,
      channels: 2,
      latencyHint: 'interactive',
      ...config,
    };

    this.effects = this.getDefaultEffects();
  }

  private getDefaultEffects(): EffectsConfig {
    return {
      autoTune: {
        enabled: true,
        strength: 0.7,
        speed: 20,
        key: 'C',
        scale: 'major',
      },
      reverb: {
        enabled: true,
        mix: 0.25,
        roomSize: 0.7,
        damping: 0.5,
      },
      compression: {
        enabled: true,
        threshold: -20,
        ratio: 3,
        attack: 5,
        release: 100,
      },
      eq: {
        enabled: true,
        lowCut: 80,
        presence: { freq: 3000, gain: 3 },
        warmth: { freq: 300, gain: 1.5 },
        air: { freq: 12000, gain: 2 },
      },
    };
  }

  async initialize(): Promise<void> {
    try {
      // Request microphone permission
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
        },
      });

      // Create AudioContext with optimal settings
      this.audioContext = new AudioContext({
        sampleRate: this.config.sampleRate,
        latencyHint: this.config.latencyHint,
      });

      // Load audio worklet for custom processing
      await this.loadAudioWorklet();

      // Build audio processing chain
      await this.buildProcessingChain();

      this.isProcessing = true;
      console.log('✅ Web Audio Processor initialized');
    } catch (error) {
      console.error('Failed to initialize audio processor:', error);
      throw error;
    }
  }

  private async loadAudioWorklet(): Promise<void> {
    if (!this.audioContext) return;

    try {
      await this.audioContext.audioWorklet.addModule('/audio-worklet.js');
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'karaoke-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2],
          processorOptions: {
            bufferSize: this.config.bufferSize,
          },
        }
      );

      // Handle messages from worklet
      this.workletNode.port.onmessage = (event) => {
        this.handleWorkletMessage(event.data);
      };
    } catch (error) {
      console.warn('Audio worklet not supported, falling back to ScriptProcessor');
    }
  }

  private async buildProcessingChain(): Promise<void> {
    if (!this.audioContext || !this.stream) return;

    // Create source from microphone
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);

    // Create analyser for visualization
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.8;

    // Create compressor
    this.compressor = this.audioContext.createDynamicsCompressor();
    this.updateCompressor();

    // Create EQ filters
    this.createEQFilters();

    // Create reverb
    await this.createReverb();

    // Create gain node
    this.gainNode = this.audioContext.createGain();
    this.gainNode.gain.value = 1.0;

    // Connect the chain
    this.connectAudioChain();
  }

  private createEQFilters(): void {
    if (!this.audioContext) return;

    // High-pass filter (remove rumble)
    const highPass = this.audioContext.createBiquadFilter();
    highPass.type = 'highpass';
    highPass.frequency.value = this.effects.eq.lowCut;
    highPass.Q.value = 0.7;
    this.filters.set('highpass', highPass);

    // Presence boost
    const presence = this.audioContext.createBiquadFilter();
    presence.type = 'peaking';
    presence.frequency.value = this.effects.eq.presence.freq;
    presence.Q.value = 1;
    presence.gain.value = this.effects.eq.presence.gain;
    this.filters.set('presence', presence);

    // Warmth
    const warmth = this.audioContext.createBiquadFilter();
    warmth.type = 'peaking';
    warmth.frequency.value = this.effects.eq.warmth.freq;
    warmth.Q.value = 0.7;
    warmth.gain.value = this.effects.eq.warmth.gain;
    this.filters.set('warmth', warmth);

    // Air
    const air = this.audioContext.createBiquadFilter();
    air.type = 'highshelf';
    air.frequency.value = this.effects.eq.air.freq;
    air.gain.value = this.effects.eq.air.gain;
    this.filters.set('air', air);
  }

  private async createReverb(): Promise<void> {
    if (!this.audioContext) return;

    this.convolver = this.audioContext.createConvolver();

    // Generate impulse response for reverb
    const length = this.audioContext.sampleRate * 2; // 2 second reverb
    const impulse = this.audioContext.createBuffer(
      2,
      length,
      this.audioContext.sampleRate
    );

    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        channelData[i] = 
          (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2);
      }
    }

    this.convolver.buffer = impulse;
  }

  private updateCompressor(): void {
    if (!this.compressor) return;

    const comp = this.effects.compression;
    this.compressor.threshold.value = comp.threshold;
    this.compressor.ratio.value = comp.ratio;
    this.compressor.attack.value = comp.attack / 1000;
    this.compressor.release.value = comp.release / 1000;
  }

  private connectAudioChain(): void {
    if (!this.audioContext || !this.sourceNode) return;

    let currentNode: AudioNode = this.sourceNode;

    // Connect through EQ filters
    if (this.effects.eq.enabled) {
      this.filters.forEach((filter) => {
        currentNode.connect(filter);
        currentNode = filter;
      });
    }

    // Connect compressor
    if (this.effects.compression.enabled && this.compressor) {
      currentNode.connect(this.compressor);
      currentNode = this.compressor;
    }

    // Connect worklet or analyzer
    if (this.workletNode) {
      currentNode.connect(this.workletNode);
      currentNode = this.workletNode;
    }

    // Connect analyser for visualization
    if (this.analyser) {
      currentNode.connect(this.analyser);
    }

    // Connect reverb (parallel processing)
    if (this.effects.reverb.enabled && this.convolver && this.gainNode) {
      const wetGain = this.audioContext.createGain();
      const dryGain = this.audioContext.createGain();

      wetGain.gain.value = this.effects.reverb.mix;
      dryGain.gain.value = 1 - this.effects.reverb.mix;

      currentNode.connect(dryGain);
      currentNode.connect(this.convolver);
      this.convolver.connect(wetGain);

      dryGain.connect(this.gainNode);
      wetGain.connect(this.gainNode);

      currentNode = this.gainNode;
    } else if (this.gainNode) {
      currentNode.connect(this.gainNode);
      currentNode = this.gainNode;
    }

    // Connect to output
    currentNode.connect(this.audioContext.destination);
  }

  private handleWorkletMessage(data: any): void {
    switch (data.type) {
      case 'pitch':
        this.onPitchDetected(data.frequency, data.note);
        break;
      case 'level':
        this.onLevelUpdate(data.level, data.peak);
        break;
      case 'clipping':
        console.warn('Audio clipping detected!');
        break;
    }
  }

  private onPitchDetected(frequency: number, note: string): void {
    // Emit pitch detection event
    window.dispatchEvent(
      new CustomEvent('pitch-detected', {
        detail: { frequency, note },
      })
    );
  }

  private onLevelUpdate(level: number, peak: number): void {
    // Emit level update event
    window.dispatchEvent(
      new CustomEvent('level-update', {
        detail: { level, peak },
      })
    );
  }

  updateEffects(effects: Partial<EffectsConfig>): void {
    this.effects = { ...this.effects, ...effects };

    // Update compressor
    if (effects.compression) {
      this.updateCompressor();
    }

    // Update EQ
    if (effects.eq) {
      this.filters.get('highpass')?.frequency.setValueAtTime(
        effects.eq.lowCut || this.effects.eq.lowCut,
        0
      );
      // Update other filters...
    }

    // Update to worklet
    if (this.workletNode && effects.autoTune) {
      this.workletNode.port.postMessage({
        type: 'update-effects',
        effects: this.effects,
      });
    }
  }

  getAnalyserData(): { frequencies: Uint8Array; waveform: Uint8Array } | null {
    if (!this.analyser) return null;

    const frequencies = new Uint8Array(this.analyser.frequencyBinCount);
    const waveform = new Uint8Array(this.analyser.frequencyBinCount);

    this.analyser.getByteFrequencyData(frequencies);
    this.analyser.getByteTimeDomainData(waveform);

    return { frequencies, waveform };
  }

  async startRecording(): Promise<MediaRecorder> {
    if (!this.audioContext || !this.gainNode) {
      throw new Error('Audio processor not initialized');
    }

    const destination = this.audioContext.createMediaStreamDestination();
    this.gainNode.connect(destination);

    const recorder = new MediaRecorder(destination.stream, {
      mimeType: 'audio/webm',
    });

    return recorder;
  }

  stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
    }

    if (this.audioContext) {
      this.audioContext.close();
    }

    this.isProcessing = false;
    console.log('Audio processor stopped');
  }

  getStatus(): {
    isProcessing: boolean;
    sampleRate: number;
    latency: number;
    effects: EffectsConfig;
  } {
    return {
      isProcessing: this.isProcessing,
      sampleRate: this.audioContext?.sampleRate || 0,
      latency: this.audioContext?.baseLatency || 0,
      effects: this.effects,
    };
  }
}