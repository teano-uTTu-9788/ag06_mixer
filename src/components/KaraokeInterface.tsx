import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Slider,
  Switch,
  FormControlLabel,
  Grid,
  Paper,
  Alert,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Collapse,
} from '@mui/material';
import {
  Mic,
  MicOff,
  PlayArrow,
  Stop,
  Settings,
  GraphicEq,
  Tune,
  WaterDrop,
  Compress,
  ExpandMore,
  ExpandLess,
  RecordVoiceOver,
  MusicNote,
  VolumeUp,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { WebAudioProcessor, EffectsConfig } from '../audio/WebAudioProcessor';
import { motion, AnimatePresence } from 'framer-motion';

// Type definitions for status change callback
export interface KaraokeActiveStatus {
  active: true;
  effects: EffectsConfig;
}

export interface KaraokeInactiveStatus {
  active: false;
}

export type KaraokeStatus = KaraokeActiveStatus | KaraokeInactiveStatus;

// Custom event types
export interface PitchDetectedEvent extends CustomEvent {
  detail: {
    frequency: number;
    note: string;
  };
}

export interface LevelUpdateEvent extends CustomEvent {
  detail: {
    level: number;
    peak: number;
  };
}

// Effect value types based on effect category
type AutoTuneValue = boolean | number | string | 'major' | 'minor' | 'chromatic';
type ReverbValue = boolean | number;
type CompressionValue = boolean | number;
type EQValue = boolean | number | { freq: number; gain: number };
type EffectValue = AutoTuneValue | ReverbValue | CompressionValue | EQValue;

// Styled components
const StyledCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  color: 'white',
  marginBottom: theme.spacing(2),
}));

const VUMeter = styled(LinearProgress)(({ theme }) => ({
  height: 20,
  borderRadius: 10,
  '& .MuiLinearProgress-bar': {
    background: 'linear-gradient(90deg, #00ff00 0%, #ffff00 50%, #ff0000 100%)',
  },
}));

const PitchIndicator = styled(Box)(({ theme }) => ({
  width: '100%',
  height: 40,
  position: 'relative',
  background: 'rgba(0,0,0,0.2)',
  borderRadius: theme.shape.borderRadius,
  marginTop: theme.spacing(1),
}));

interface KaraokeInterfaceProps {
  onStatusChange?: (status: KaraokeStatus) => void;
}

export const KaraokeInterface: React.FC<KaraokeInterfaceProps> = ({
  onStatusChange,
}) => {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentLevel, setCurrentLevel] = useState(0);
  const [peakLevel, setPeakLevel] = useState(0);
  const [currentPitch, setCurrentPitch] = useState<string | null>(null);
  const [pitchFrequency, setPitchFrequency] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  const audioProcessor = useRef<WebAudioProcessor | null>(null);
  const animationFrame = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [effects, setEffects] = useState<EffectsConfig>({
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
  });

  // Initialize audio processor
  const initializeAudio = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      audioProcessor.current = new WebAudioProcessor({
        sampleRate: 48000,
        bufferSize: 256,
        channels: 2,
        latencyHint: 'interactive',
      });

      await audioProcessor.current.initialize();
      audioProcessor.current.updateEffects(effects);

      // Setup event listeners
      window.addEventListener('pitch-detected', handlePitchDetected);
      window.addEventListener('level-update', handleLevelUpdate);

      setIsActive(true);
      startVisualization();

      if (onStatusChange) {
        onStatusChange({ active: true, effects });
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to initialize audio';
      setError(errorMessage);
      console.error('Audio initialization error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [effects, onStatusChange]);

  // Stop audio processor
  const stopAudio = useCallback(() => {
    if (audioProcessor.current) {
      audioProcessor.current.stop();
      audioProcessor.current = null;
    }

    if (animationFrame.current) {
      cancelAnimationFrame(animationFrame.current);
    }

    window.removeEventListener('pitch-detected', handlePitchDetected);
    window.removeEventListener('level-update', handleLevelUpdate);

    setIsActive(false);
    setCurrentLevel(0);
    setPeakLevel(0);
    setCurrentPitch(null);
    setPitchFrequency(null);

    if (onStatusChange) {
      onStatusChange({ active: false });
    }
  }, [onStatusChange]);

  // Handle pitch detection
  const handlePitchDetected = useCallback((event: Event) => {
    const customEvent = event as PitchDetectedEvent;
    const { frequency, note } = customEvent.detail;
    setPitchFrequency(frequency);
    setCurrentPitch(note);
  }, []);

  // Handle level updates
  const handleLevelUpdate = useCallback((event: Event) => {
    const customEvent = event as LevelUpdateEvent;
    const { level, peak } = customEvent.detail;
    setCurrentLevel(level * 100);
    setPeakLevel(peak * 100);
  }, []);

  // Update effects
  const updateEffect = useCallback(
    (category: keyof EffectsConfig, field: string, value: EffectValue) => {
      const newEffects = {
        ...effects,
        [category]: {
          ...effects[category],
          [field]: value,
        },
      };

      setEffects(newEffects);

      if (audioProcessor.current && isActive) {
        audioProcessor.current.updateEffects(newEffects);
      }
    },
    [effects, isActive]
  );

  // Visualization
  const startVisualization = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      if (!audioProcessor.current || !isActive) return;

      const data = audioProcessor.current.getAnalyserData();
      if (!data) return;

      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#00ff00';
      ctx.beginPath();

      const sliceWidth = canvas.width / data.waveform.length;
      let x = 0;

      for (let i = 0; i < data.waveform.length; i++) {
        const v = data.waveform[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.stroke();

      // Draw frequency bars
      const barWidth = (canvas.width / data.frequencies.length) * 2.5;
      let barX = 0;

      for (let i = 0; i < data.frequencies.length; i++) {
        const barHeight = (data.frequencies[i] / 255) * canvas.height;
        const hue = (i / data.frequencies.length) * 360;

        ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        ctx.fillRect(barX, canvas.height - barHeight, barWidth, barHeight);

        barX += barWidth + 1;
      }

      animationFrame.current = requestAnimationFrame(draw);
    };

    draw();
  }, [isActive]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio();
    };
  }, [stopAudio]);

  return (
    <Box sx={{ maxWidth: 1200, margin: '0 auto', padding: 2 }}>
      {/* Header Card */}
      <StyledCard>
        <CardContent>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
          >
            <Box display="flex" alignItems="center" gap={2}>
              <MusicNote fontSize="large" />
              <Box>
                <Typography variant="h4" fontWeight="bold">
                  AI Karaoke Mixer
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  Professional vocal processing in your browser
                </Typography>
              </Box>
            </Box>

            <Button
              variant="contained"
              size="large"
              color={isActive ? 'error' : 'success'}
              startIcon={isActive ? <Stop /> : <Mic />}
              onClick={isActive ? stopAudio : initializeAudio}
              disabled={isLoading}
              sx={{
                minWidth: 150,
                background: isActive
                  ? 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)'
                  : 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              }}
            >
              {isLoading ? 'Loading...' : isActive ? 'Stop' : 'Start'}
            </Button>
          </Box>
        </CardContent>
      </StyledCard>

      {/* Error Alert */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Controls */}
      <Grid container spacing={2}>
        {/* Live Monitoring */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              <RecordVoiceOver sx={{ mr: 1, verticalAlign: 'middle' }} />
              Live Monitoring
            </Typography>

            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="textSecondary">
                Input Level
              </Typography>
              <VUMeter variant="determinate" value={currentLevel} />
              <Box display="flex" justifyContent="space-between" mt={1}>
                <Chip
                  label={`Level: ${currentLevel.toFixed(1)}%`}
                  size="small"
                  color="primary"
                />
                <Chip
                  label={`Peak: ${peakLevel.toFixed(1)}%`}
                  size="small"
                  color="warning"
                />
              </Box>
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" color="textSecondary">
                Pitch Detection
              </Typography>
              <PitchIndicator>
                {currentPitch && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                    }}
                  >
                    <Chip
                      label={`${currentPitch} (${pitchFrequency?.toFixed(1)} Hz)`}
                      color="secondary"
                    />
                  </motion.div>
                )}
              </PitchIndicator>
            </Box>

            {/* Visualization Canvas */}
            <Box sx={{ mt: 3 }}>
              <canvas
                ref={canvasRef}
                width={500}
                height={150}
                style={{
                  width: '100%',
                  height: 150,
                  background: 'rgba(0,0,0,0.1)',
                  borderRadius: 8,
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Effects Controls */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">
                <GraphicEq sx={{ mr: 1, verticalAlign: 'middle' }} />
                Effects
              </Typography>
              <IconButton onClick={() => setShowAdvanced(!showAdvanced)}>
                {showAdvanced ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>

            {/* Auto-Tune */}
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={effects.autoTune.enabled}
                    onChange={(e) =>
                      updateEffect('autoTune', 'enabled', e.target.checked)
                    }
                    color="secondary"
                  />
                }
                label={
                  <Box display="flex" alignItems="center">
                    <Tune sx={{ mr: 1 }} />
                    Auto-Tune
                  </Box>
                }
              />
              {effects.autoTune.enabled && (
                <Box sx={{ pl: 4, mt: 1 }}>
                  <Typography variant="body2" color="textSecondary">
                    Strength: {(effects.autoTune.strength * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={effects.autoTune.strength}
                    onChange={(e, value) =>
                      updateEffect('autoTune', 'strength', value as number)
                    }
                    min={0}
                    max={1}
                    step={0.1}
                    color="secondary"
                  />
                </Box>
              )}
            </Box>

            {/* Reverb */}
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={effects.reverb.enabled}
                    onChange={(e) =>
                      updateEffect('reverb', 'enabled', e.target.checked)
                    }
                    color="primary"
                  />
                }
                label={
                  <Box display="flex" alignItems="center">
                    <WaterDrop sx={{ mr: 1 }} />
                    Reverb
                  </Box>
                }
              />
              {effects.reverb.enabled && (
                <Box sx={{ pl: 4, mt: 1 }}>
                  <Typography variant="body2" color="textSecondary">
                    Mix: {(effects.reverb.mix * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={effects.reverb.mix}
                    onChange={(e, value) =>
                      updateEffect('reverb', 'mix', value as number)
                    }
                    min={0}
                    max={1}
                    step={0.05}
                    color="primary"
                  />
                </Box>
              )}
            </Box>

            {/* Advanced Controls */}
            <Collapse in={showAdvanced}>
              {/* Compression */}
              <Box sx={{ mt: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={effects.compression.enabled}
                      onChange={(e) =>
                        updateEffect('compression', 'enabled', e.target.checked)
                      }
                    />
                  }
                  label={
                    <Box display="flex" alignItems="center">
                      <Compress sx={{ mr: 1 }} />
                      Compression
                    </Box>
                  }
                />
                {effects.compression.enabled && (
                  <Box sx={{ pl: 4, mt: 1 }}>
                    <Typography variant="body2" color="textSecondary">
                      Threshold: {effects.compression.threshold} dB
                    </Typography>
                    <Slider
                      value={effects.compression.threshold}
                      onChange={(e, value) =>
                        updateEffect('compression', 'threshold', value as number)
                      }
                      min={-40}
                      max={0}
                      step={1}
                    />
                  </Box>
                )}
              </Box>

              {/* EQ */}
              <Box sx={{ mt: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={effects.eq.enabled}
                      onChange={(e) =>
                        updateEffect('eq', 'enabled', e.target.checked)
                      }
                    />
                  }
                  label={
                    <Box display="flex" alignItems="center">
                      <GraphicEq sx={{ mr: 1 }} />
                      Equalizer
                    </Box>
                  }
                />
              </Box>
            </Collapse>
          </Paper>
        </Grid>
      </Grid>

      {/* Status Bar */}
      {isActive && (
        <Paper sx={{ mt: 2, p: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Chip
              icon={<VolumeUp />}
              label="Processing Active"
              color="success"
              variant="outlined"
            />
            <Typography variant="body2" color="textSecondary">
              Latency: {audioProcessor.current?.getStatus().latency.toFixed(1)} ms
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Sample Rate: {audioProcessor.current?.getStatus().sampleRate} Hz
            </Typography>
          </Box>
        </Paper>
      )}
    </Box>
  );
};