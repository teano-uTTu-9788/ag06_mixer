"""
Audio Model Factory for ML Optimization
Creates and trains lightweight models specifically for real-time audio processing
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class AudioModelConfig:
    """Configuration for audio ML models"""
    model_type: str = "genre_classifier"  # genre_classifier, noise_detector, level_analyzer
    input_features: int = 13  # MFCC features
    hidden_units: List[int] = None  # Hidden layer sizes
    output_classes: int = 5  # Number of genres
    sample_rate: int = 48000
    frame_size: int = 960  # 20ms at 48kHz
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [64, 32]

class SimpleGenreClassifier:
    """Lightweight genre classification model for real-time processing"""
    
    def __init__(self, config: AudioModelConfig):
        self.config = config
        self.model = None
        self.feature_scaler = None
        self.label_encoder = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup the neural network model"""
        try:
            # Try scikit-learn first (lightweight alternative)
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(self.config.hidden_units),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
            
            self.feature_scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            
            logger.info("Initialized sklearn-based genre classifier")
            
        except ImportError:
            # Fallback to simple numpy implementation
            logger.warning("sklearn not available, using simple numpy model")
            self._create_numpy_model()
    
    def _create_numpy_model(self):
        """Create a simple numpy-based neural network"""
        # Initialize weights randomly
        input_size = self.config.input_features
        hidden_sizes = self.config.hidden_units
        output_size = self.config.output_classes
        
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        logger.info(f"Created numpy model with layers: {layer_sizes}")
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        try:
            import librosa
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.config.sample_rate,
                n_mfcc=self.config.input_features,
                hop_length=512,
                n_fft=1024
            )
            
            # Take mean across time axis
            features = np.mean(mfccs, axis=1)
            return features
            
        except ImportError:
            # Fallback to simple spectral features
            logger.warning("librosa not available, using simple spectral features")
            return self._extract_simple_features(audio_data)
    
    def _extract_simple_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract simple spectral features without librosa"""
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Simple spectral features
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Divide into bands and compute energy
        bands = np.array_split(magnitude, self.config.input_features)
        features = np.array([np.mean(band) for band in bands])
        
        # Add some temporal features
        if len(features) < self.config.input_features:
            # Add RMS, zero crossing rate, etc.
            rms = np.sqrt(np.mean(audio_data**2))
            zcr = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
            
            additional = [rms, zcr]
            features = np.concatenate([features, additional])
        
        return features[:self.config.input_features]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the genre classifier"""
        if hasattr(self.model, 'fit'):
            # sklearn model
            if self.feature_scaler:
                X = self.feature_scaler.fit_transform(X)
            if self.label_encoder:
                y = self.label_encoder.fit_transform(y)
            
            self.model.fit(X, y)
            logger.info("Training completed with sklearn")
            
        else:
            # numpy model - simple training
            self._train_numpy_model(X, y)
    
    def _train_numpy_model(self, X: np.ndarray, y: np.ndarray):
        """Train the numpy-based model"""
        # Simple gradient descent training
        learning_rate = 0.01
        epochs = 100
        
        # One-hot encode labels
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y.astype(int)]
        
        for epoch in range(epochs):
            # Forward pass
            activations = [X]
            
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = np.dot(activations[-1], w) + b
                if i == len(self.weights) - 1:
                    # Output layer - softmax
                    a = self._softmax(z)
                else:
                    # Hidden layer - ReLU
                    a = np.maximum(0, z)
                activations.append(a)
            
            # Compute loss
            loss = -np.mean(np.sum(y_onehot * np.log(activations[-1] + 1e-8), axis=1))
            
            # Backward pass (simplified)
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def _softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict genre for audio features"""
        if hasattr(self.model, 'predict'):
            # sklearn model
            if self.feature_scaler:
                X = self.feature_scaler.transform(X)
            predictions = self.model.predict(X)
            
            if self.label_encoder:
                predictions = self.label_encoder.inverse_transform(predictions)
            
            return predictions
        
        else:
            # numpy model
            return self._predict_numpy(X)
    
    def _predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """Predict with numpy model"""
        activations = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations, w) + b
            if i == len(self.weights) - 1:
                activations = self._softmax(z)
            else:
                activations = np.maximum(0, z)
        
        return np.argmax(activations, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if hasattr(self.model, 'predict_proba'):
            if self.feature_scaler:
                X = self.feature_scaler.transform(X)
            return self.model.predict_proba(X)
        else:
            # numpy model - return softmax probabilities
            return self._predict_proba_numpy(X)
    
    def _predict_proba_numpy(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with numpy model"""
        activations = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations, w) + b
            if i == len(self.weights) - 1:
                activations = self._softmax(z)
            else:
                activations = np.maximum(0, z)
        
        return activations
    
    def save(self, model_path: str):
        """Save the trained model"""
        import pickle
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.config,
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'weights': getattr(self, 'weights', None),
            'biases': getattr(self, 'biases', None)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str):
        """Load a trained model"""
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.feature_scaler = model_data['feature_scaler']
        instance.label_encoder = model_data['label_encoder']
        
        if model_data['weights'] is not None:
            instance.weights = model_data['weights']
            instance.biases = model_data['biases']
        
        logger.info(f"Model loaded from {model_path}")
        return instance

class AudioDataGenerator:
    """Generate synthetic audio data for model training"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.genres = ['speech', 'rock', 'jazz', 'electronic', 'classical']
    
    def generate_training_data(self, samples_per_genre: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic audio data for different genres"""
        X = []
        y = []
        
        for genre_idx, genre in enumerate(self.genres):
            for _ in range(samples_per_genre):
                # Generate synthetic audio for this genre
                audio = self._generate_genre_audio(genre)
                X.append(audio)
                y.append(genre_idx)
        
        return np.array(X), np.array(y)
    
    def _generate_genre_audio(self, genre: str) -> np.ndarray:
        """Generate synthetic audio that mimics genre characteristics"""
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if genre == 'speech':
            # Speech-like: formants and pauses
            audio = self._generate_speech_like(t)
        elif genre == 'rock':
            # Rock-like: strong bass and distortion
            audio = self._generate_rock_like(t)
        elif genre == 'jazz':
            # Jazz-like: complex harmonics
            audio = self._generate_jazz_like(t)
        elif genre == 'electronic':
            # Electronic-like: synthetic sounds
            audio = self._generate_electronic_like(t)
        elif genre == 'classical':
            # Classical-like: orchestral harmonics
            audio = self._generate_classical_like(t)
        else:
            # Default white noise
            audio = np.random.randn(len(t)) * 0.1
        
        # Add some noise
        audio += np.random.randn(len(audio)) * 0.01
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio.astype(np.float32)
    
    def _generate_speech_like(self, t):
        """Generate speech-like audio"""
        # Formant-like frequencies
        f1, f2, f3 = 800, 1200, 2400
        audio = (np.sin(2 * np.pi * f1 * t) * 0.5 +
                np.sin(2 * np.pi * f2 * t) * 0.3 +
                np.sin(2 * np.pi * f3 * t) * 0.2)
        
        # Add pauses
        pause_mask = (t % 0.5) < 0.3  # 30% speech, 70% pause pattern
        audio *= pause_mask
        
        return audio
    
    def _generate_rock_like(self, t):
        """Generate rock-like audio"""
        # Strong fundamental with harmonics
        fundamental = 220  # A3
        audio = (np.sin(2 * np.pi * fundamental * t) * 0.6 +
                np.sin(2 * np.pi * fundamental * 2 * t) * 0.3 +
                np.sin(2 * np.pi * fundamental * 3 * t) * 0.2)
        
        # Add distortion
        audio = np.tanh(audio * 3)
        
        return audio
    
    def _generate_jazz_like(self, t):
        """Generate jazz-like audio"""
        # Complex chord with 7ths and 9ths
        freqs = [220, 277, 330, 370, 415]  # Complex jazz chord
        audio = sum(np.sin(2 * np.pi * f * t) * (0.8 / len(freqs)) for f in freqs)
        
        # Add swing rhythm
        swing_mask = np.sin(2 * np.pi * 2 * t) > 0  # 2 Hz swing pattern
        audio *= (0.7 + 0.3 * swing_mask)
        
        return audio
    
    def _generate_electronic_like(self, t):
        """Generate electronic-like audio"""
        # Sawtooth wave with filters
        fundamental = 440
        sawtooth = 2 * (t * fundamental - np.floor(t * fundamental + 0.5))
        
        # Add filter sweep
        cutoff = 1000 + 500 * np.sin(2 * np.pi * 0.5 * t)
        # Simple high-pass filter effect
        audio = sawtooth * (1 + np.sin(2 * np.pi * cutoff * t)) * 0.5
        
        return audio
    
    def _generate_classical_like(self, t):
        """Generate classical-like audio"""
        # Rich harmonic content
        fundamental = 261  # C4
        harmonics = [1, 2, 3, 4, 5, 6]
        amplitudes = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        audio = sum(amp * np.sin(2 * np.pi * fundamental * harm * t) 
                   for harm, amp in zip(harmonics, amplitudes))
        
        # Add vibrato
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 6 * t)
        audio *= vibrato
        
        return audio

class AudioModelTrainingPipeline:
    """Complete pipeline for training audio models"""
    
    def __init__(self):
        self.data_generator = AudioDataGenerator()
        self.models = {}
    
    def create_genre_classifier(self, config: AudioModelConfig = None) -> SimpleGenreClassifier:
        """Create and train a genre classifier"""
        if config is None:
            config = AudioModelConfig(
                model_type="genre_classifier",
                input_features=13,
                hidden_units=[32, 16],
                output_classes=5
            )
        
        # Generate training data
        logger.info("Generating synthetic training data...")
        X_audio, y = self.data_generator.generate_training_data(samples_per_genre=50)
        
        # Create classifier
        classifier = SimpleGenreClassifier(config)
        
        # Extract features from audio
        logger.info("Extracting features...")
        X_features = []
        for audio in X_audio:
            features = classifier.extract_features(audio)
            X_features.append(features)
        X_features = np.array(X_features)
        
        # Train the model
        logger.info("Training genre classifier...")
        classifier.train(X_features, y)
        
        self.models['genre_classifier'] = classifier
        return classifier
    
    def save_all_models(self, output_dir: str = "models/audio"):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_path / f"{name}.pkl"
            model.save(str(model_path))
        
        logger.info(f"All models saved to {output_dir}")
    
    def create_optimized_models(self, output_dir: str = "models/optimized"):
        """Create optimized versions of all models"""
        from .model_optimizer import ModelOptimizationPipeline, OptimizationConfig
        
        # This would convert the sklearn/numpy models to TensorFlow/ONNX format
        # and then optimize them - implementation depends on model format
        logger.info("Model optimization would be applied here")
        # TODO: Implement model format conversion and optimization

def demo_audio_model_training():
    """Demonstrate audio model training pipeline"""
    print("🎵 Audio Model Training Demo")
    print("=" * 40)
    
    # Create training pipeline
    pipeline = AudioModelTrainingPipeline()
    
    # Train genre classifier
    classifier = pipeline.create_genre_classifier()
    
    # Test the classifier
    print("\n🧪 Testing genre classifier...")
    test_audio = pipeline.data_generator._generate_genre_audio('jazz')
    features = classifier.extract_features(test_audio)
    prediction = classifier.predict(features.reshape(1, -1))
    probabilities = classifier.predict_proba(features.reshape(1, -1))
    
    genres = pipeline.data_generator.genres
    print(f"Predicted genre: {genres[prediction[0]]}")
    print(f"Probabilities: {dict(zip(genres, probabilities[0]))}")
    
    # Save models
    pipeline.save_all_models()
    
    print("\n✅ Audio model training demo completed!")
    return pipeline

if __name__ == "__main__":
    demo_audio_model_training()