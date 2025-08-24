/**
 * AI Mixer Core - Cross-platform C/C++ interface
 * 
 * Provides unified audio processing and AI-powered mixing capabilities
 * for iOS and Android mobile applications.
 * 
 * Features:
 * - Real-time audio processing with <20ms latency
 * - AI-powered genre detection and adaptive mixing
 * - Professional DSP chain (Gate, Compressor, EQ, Limiter)
 * - Optimized for mobile devices with battery efficiency
 */

#ifndef AI_MIXER_CORE_H
#define AI_MIXER_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ==============================================================================
// CONSTANTS & CONFIGURATION
// ==============================================================================

#define AI_MIXER_VERSION "1.0.0"
#define AI_MIXER_SAMPLE_RATE 48000
#define AI_MIXER_FRAME_SIZE 960  // 20ms at 48kHz
#define AI_MIXER_MAX_CHANNELS 2
#define AI_MIXER_FEATURE_SIZE 13  // MFCC features

// Genre classifications
typedef enum {
    GENRE_SPEECH = 0,
    GENRE_ROCK = 1,
    GENRE_JAZZ = 2,
    GENRE_ELECTRONIC = 3,
    GENRE_CLASSICAL = 4,
    GENRE_UNKNOWN = 5
} ai_mixer_genre_t;

// Processing status codes
typedef enum {
    AI_MIXER_SUCCESS = 0,
    AI_MIXER_ERROR_INVALID_PARAMETER = -1,
    AI_MIXER_ERROR_NOT_INITIALIZED = -2,
    AI_MIXER_ERROR_PROCESSING_FAILED = -3,
    AI_MIXER_ERROR_MEMORY_ALLOCATION = -4,
    AI_MIXER_ERROR_MODEL_LOAD_FAILED = -5
} ai_mixer_result_t;

// DSP configuration
typedef struct {
    // Noise Gate
    float gate_threshold_db;      // -60.0f to 0.0f
    float gate_ratio;             // 1.0f to inf (10.0f typical)
    float gate_attack_ms;         // 0.1f to 100.0f
    float gate_release_ms;        // 10.0f to 5000.0f
    
    // Compressor
    float comp_threshold_db;      // -60.0f to 0.0f
    float comp_ratio;             // 1.0f to 20.0f
    float comp_attack_ms;         // 0.1f to 100.0f
    float comp_release_ms;        // 10.0f to 5000.0f
    float comp_knee_db;           // 0.0f to 10.0f (soft knee)
    
    // Parametric EQ (3-band)
    float eq_low_gain_db;         // -15.0f to 15.0f
    float eq_low_freq;            // 20.0f to 500.0f
    float eq_mid_gain_db;         // -15.0f to 15.0f
    float eq_mid_freq;            // 200.0f to 5000.0f
    float eq_high_gain_db;        // -15.0f to 15.0f
    float eq_high_freq;           // 2000.0f to 20000.0f
    
    // Limiter
    float limiter_threshold_db;   // -20.0f to 0.0f
    float limiter_release_ms;     // 1.0f to 100.0f
    float limiter_lookahead_ms;   // 0.0f to 10.0f
} ai_mixer_dsp_config_t;

// Processing metadata
typedef struct {
    ai_mixer_genre_t detected_genre;
    float confidence;
    float processing_time_ms;
    float cpu_usage_percent;
    uint32_t frame_count;
    
    // Audio analysis
    float rms_level_db;
    float peak_level_db;
    float spectral_centroid;
    float zero_crossing_rate;
    
    // DSP status
    bool gate_active;
    float comp_gain_reduction_db;
    bool limiter_active;
} ai_mixer_metadata_t;

// Mixer handle (opaque)
typedef struct ai_mixer_context ai_mixer_context_t;

// ==============================================================================
// CORE API
// ==============================================================================

/**
 * Initialize AI mixer context
 * 
 * @param config DSP configuration (NULL for defaults)
 * @return Mixer context handle or NULL on failure
 */
ai_mixer_context_t* ai_mixer_create(const ai_mixer_dsp_config_t* config);

/**
 * Destroy mixer context and free resources
 * 
 * @param ctx Mixer context handle
 */
void ai_mixer_destroy(ai_mixer_context_t* ctx);

/**
 * Get default DSP configuration
 * 
 * @param config Output configuration structure
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_get_default_config(ai_mixer_dsp_config_t* config);

/**
 * Update DSP configuration at runtime
 * 
 * @param ctx Mixer context handle
 * @param config New DSP configuration
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_update_config(ai_mixer_context_t* ctx, 
                                        const ai_mixer_dsp_config_t* config);

/**
 * Process audio frame with AI-powered mixing
 * 
 * @param ctx Mixer context handle
 * @param input_samples Input audio samples (interleaved stereo)
 * @param output_samples Output audio samples (interleaved stereo)
 * @param frame_size Number of samples per channel (should be AI_MIXER_FRAME_SIZE)
 * @param metadata Output processing metadata (optional, can be NULL)
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_process_frame(ai_mixer_context_t* ctx,
                                        const float* input_samples,
                                        float* output_samples,
                                        uint32_t frame_size,
                                        ai_mixer_metadata_t* metadata);

/**
 * Get mixer performance metrics
 * 
 * @param ctx Mixer context handle
 * @param avg_processing_time_ms Average processing time over last 100 frames
 * @param peak_processing_time_ms Peak processing time over last 100 frames
 * @param cpu_usage_percent Current CPU usage estimate
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_get_performance_metrics(ai_mixer_context_t* ctx,
                                                  float* avg_processing_time_ms,
                                                  float* peak_processing_time_ms,
                                                  float* cpu_usage_percent);

// ==============================================================================
// ADVANCED API
// ==============================================================================

/**
 * Load custom AI model for genre detection
 * 
 * @param ctx Mixer context handle
 * @param model_data Model binary data (TensorFlow Lite format)
 * @param model_size Size of model data in bytes
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_load_custom_model(ai_mixer_context_t* ctx,
                                            const uint8_t* model_data,
                                            uint32_t model_size);

/**
 * Set genre detection callback for custom processing
 * 
 * @param ctx Mixer context handle
 * @param callback Function to call when genre is detected
 * @param user_data User data passed to callback
 * @return AI_MIXER_SUCCESS on success
 */
typedef void (*ai_mixer_genre_callback_t)(ai_mixer_genre_t genre, 
                                          float confidence, 
                                          void* user_data);

ai_mixer_result_t ai_mixer_set_genre_callback(ai_mixer_context_t* ctx,
                                             ai_mixer_genre_callback_t callback,
                                             void* user_data);

/**
 * Extract audio features for external processing
 * 
 * @param ctx Mixer context handle
 * @param input_samples Input audio samples
 * @param frame_size Number of samples per channel
 * @param features Output feature vector (size AI_MIXER_FEATURE_SIZE)
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_extract_features(ai_mixer_context_t* ctx,
                                           const float* input_samples,
                                           uint32_t frame_size,
                                           float* features);

/**
 * Bypass AI processing and use manual genre setting
 * 
 * @param ctx Mixer context handle
 * @param genre Manual genre setting
 * @param bypass true to bypass AI, false to re-enable
 * @return AI_MIXER_SUCCESS on success
 */
ai_mixer_result_t ai_mixer_set_manual_genre(ai_mixer_context_t* ctx,
                                           ai_mixer_genre_t genre,
                                           bool bypass);

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

/**
 * Convert result code to human-readable string
 * 
 * @param result Result code
 * @return String description of result
 */
const char* ai_mixer_result_to_string(ai_mixer_result_t result);

/**
 * Convert genre enum to string
 * 
 * @param genre Genre enum value
 * @return String name of genre
 */
const char* ai_mixer_genre_to_string(ai_mixer_genre_t genre);

/**
 * Get library version string
 * 
 * @return Version string (e.g., "1.0.0")
 */
const char* ai_mixer_get_version(void);

/**
 * Check if device supports hardware acceleration
 * 
 * @return true if hardware acceleration available
 */
bool ai_mixer_has_hardware_acceleration(void);

#ifdef __cplusplus
}
#endif

#endif // AI_MIXER_CORE_H