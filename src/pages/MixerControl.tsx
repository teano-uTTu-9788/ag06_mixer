import React, { useEffect, useCallback, useMemo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Slider,
  IconButton,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  Skeleton,
} from '@mui/material';
import {
  VolumeUp,
  VolumeOff,
  Headset,
  GraphicEq,
  FiberManualRecord,
  Stop,
  PlayArrow,
  Settings,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useSocket } from '../hooks/useSocket';
import { useAudioContext } from '../hooks/useAudioContext';
import { mixerActions } from '../store/mixerSlice';
import { RootState } from '../store';
import { ChannelStrip } from '../components/ChannelStrip';
import { EffectsRack } from '../components/EffectsRack';
import { MasterSection } from '../components/MasterSection';
import { VUMeter } from '../components/VUMeter';
import { PresetManager } from '../components/PresetManager';
import { api } from '../api/client';
// Type definitions
type MixerActionType = 'UPDATE' | 'RESET' | 'LOAD_PRESET';

interface MasterConfig {
  gain: number;
  mute: boolean;
  mono: boolean;
  limiter: boolean;
  levelL?: number;
  levelR?: number;
  peakL?: number;
  peakR?: number;
}

interface MixerState {
  channels: Record<string, ChannelConfig>;
  master: MasterConfig;
  effects: Record<string, number>;
}

interface ChannelConfig {
  gain: number;
  mute: boolean;
  solo: boolean;
  pan: number;
  level?: number;
  peak?: number;
}

// Channel property value type
type ChannelPropertyValue = number | boolean;

const MixerControl: React.FC = () => {
  const dispatch = useDispatch();
  const queryClient = useQueryClient();
  const socket = useSocket();
  const audioContext = useAudioContext();
  
  // Redux state
  const mixerState = useSelector((state: RootState) => state.mixer);
  const isConnected = useSelector((state: RootState) => state.connection.isConnected);
  
  // Fetch initial mixer state
  const { data: initialState, isLoading, error } = useQuery({
    queryKey: ['mixerState'],
    queryFn: api.mixer.getState,
    refetchInterval: isConnected ? false : 5000,
  });
  
  // Mutation for updating mixer state
  const updateMixerMutation = useMutation({
    mutationFn: api.mixer.updateState,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mixerState'] });
    },
    onError: (error) => {
      console.error('Failed to update mixer:', error);
    },
  });
  
  // WebSocket event handlers
  useEffect(() => {
    if (!socket) return;
    
    const handleMixerUpdate = (data: Partial<MixerState>) => {
      dispatch(mixerActions.updateState(data));
    };
    
    const handleChannelUpdate = (data: { channel: number; config: Partial<ChannelConfig> }) => {
      dispatch(mixerActions.updateChannel(data));
    };
    
    const handleEffectUpdate = (data: { effect: string; value: number }) => {
      dispatch(mixerActions.updateEffect(data));
    };
    
    socket.on('mixer:update', handleMixerUpdate);
    socket.on('channel:update', handleChannelUpdate);
    socket.on('effect:update', handleEffectUpdate);
    
    return () => {
      socket.off('mixer:update', handleMixerUpdate);
      socket.off('channel:update', handleChannelUpdate);
      socket.off('effect:update', handleEffectUpdate);
    };
  }, [socket, dispatch]);
  
  // Channel control handlers
  const handleChannelChange = useCallback(
    (channelId: number, property: keyof ChannelConfig, value: ChannelPropertyValue) => {
      const update = { channel: channelId, config: { [property]: value } };

      // Optimistic update
      dispatch(mixerActions.updateChannel(update));

      // Send to server
      socket?.emit('channel:update', update);
      updateMixerMutation.mutate({
        channels: {
          ...mixerState.channels,
          [channelId]: {
            ...mixerState.channels[channelId],
            [property]: value,
          },
        },
      });
    },
    [dispatch, socket, updateMixerMutation, mixerState.channels]
  );
  
  // Master property value type
  type MasterPropertyValue = number | boolean;

  // Master control handlers
  const handleMasterChange = useCallback(
    (property: keyof MasterConfig, value: MasterPropertyValue) => {
      const update = { master: { [property]: value } };

      // Optimistic update
      dispatch(mixerActions.updateMaster(update));

      // Send to server
      socket?.emit('master:update', update);
      updateMixerMutation.mutate({
        master: {
          ...mixerState.master,
          [property]: value,
        },
      });
    },
    [dispatch, socket, updateMixerMutation, mixerState.master]
  );
  
  // Effect control handlers
  const handleEffectChange = useCallback(
    (effect: string, value: number) => {
      const update = { effect, value };
      
      // Optimistic update
      dispatch(mixerActions.updateEffect(update));
      
      // Send to server
      socket?.emit('effect:update', update);
      updateMixerMutation.mutate({
        effects: {
          ...mixerState.effects,
          [effect]: value,
        },
      });
    },
    [dispatch, socket, updateMixerMutation, mixerState.effects]
  );
  
  // Memoized channel strips
  const channelStrips = useMemo(
    () =>
      Object.entries(mixerState.channels).map(([id, config]) => (
        <Grid item xs={12} sm={6} md={3} key={id}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: parseInt(id) * 0.1 }}
          >
            <ChannelStrip
              channelId={parseInt(id)}
              config={config}
              onChange={handleChannelChange}
              disabled={!isConnected}
            />
          </motion.div>
        </Grid>
      )),
    [mixerState.channels, handleChannelChange, isConnected]
  );
  
  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Skeleton variant="rectangular" height={400} />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Failed to load mixer state. Please check your connection.
        </Alert>
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3, bgcolor: 'background.default', minHeight: '100vh' }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold' }}>
          AG06 Mixer Control
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip
            icon={<FiberManualRecord />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            variant="outlined"
          />
          <IconButton color="primary">
            <Settings />
          </IconButton>
        </Box>
      </Box>
      
      {/* Main Mixer Grid */}
      <Grid container spacing={3}>
        {/* Channel Strips */}
        {channelStrips}
        
        {/* Master Section */}
        <Grid item xs={12} md={4}>
          <MasterSection
            config={mixerState.master}
            onChange={handleMasterChange}
            disabled={!isConnected}
          />
        </Grid>
        
        {/* Effects Rack */}
        <Grid item xs={12} md={8}>
          <EffectsRack
            effects={mixerState.effects}
            onChange={handleEffectChange}
            disabled={!isConnected}
          />
        </Grid>
        
        {/* VU Meters */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Level Monitoring
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(mixerState.channels).map(([id, config]) => (
                <Grid item xs={6} sm={3} key={id}>
                  <VUMeter
                    label={`CH ${id}`}
                    level={config.level || 0}
                    peak={config.peak || 0}
                  />
                </Grid>
              ))}
              <Grid item xs={6} sm={3}>
                <VUMeter
                  label="Master L"
                  level={mixerState.master.levelL || 0}
                  peak={mixerState.master.peakL || 0}
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <VUMeter
                  label="Master R"
                  level={mixerState.master.levelR || 0}
                  peak={mixerState.master.peakR || 0}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        {/* Preset Manager */}
        <Grid item xs={12}>
          <PresetManager
            currentState={mixerState}
            onLoad={(preset) => {
              dispatch(mixerActions.loadPreset(preset));
              updateMixerMutation.mutate(preset);
            }}
          />
        </Grid>
      </Grid>
      
      {/* Status Messages */}
      <AnimatePresence>
        {updateMixerMutation.isPending && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            style={{
              position: 'fixed',
              bottom: 20,
              right: 20,
            }}
          >
            <Alert severity="info">Updating mixer settings...</Alert>
          </motion.div>
        )}
      </AnimatePresence>
    </Box>
  );
};

export default MixerControl;