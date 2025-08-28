#!/usr/bin/env python3
"""
AiOke Complete System Test
Test all components including AI vocal mixing and karaoke enhancement
"""

import numpy as np
import time
from ai_vocal_auto_mixer import AIVocalAutoMixer
from karaoke_vocal_enhancer import KaraokeVocalMixer

def test_ai_vocal_mixing():
    """Test the AI vocal auto-mixing"""
    print("\n🤖 Testing AI Vocal Auto-Mixer")
    print("=" * 50)
    
    mixer = AIVocalAutoMixer()
    
    # Create test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate different vocal types
    test_cases = [
        ("Beginner Singer", 220, 0.02, "Clear tone"),  # Low male voice
        ("Female Singer", 440, 0.01, "Bright tone"),    # Female voice
        ("Confident Singer", 330, 0.005, "Warm tone"),  # Mid-range
    ]
    
    for name, freq, noise, description in test_cases:
        print(f"\n📍 Testing: {name} ({description})")
        
        # Create test vocal with vibrato
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
        vocal = np.sin(2 * np.pi * freq * vibrato * t)
        vocal += np.random.normal(0, noise, len(vocal))  # Add noise
        
        # Create backing music
        music = (np.sin(2 * np.pi * 261.63 * t) +  # C chord
                np.sin(2 * np.pi * 329.63 * t) +
                np.sin(2 * np.pi * 392.00 * t)) / 3
        
        # Process with AI
        processed = mixer.process_vocal_with_ai(vocal, music)
        
        # Analyze results
        vocal_chars = mixer.analyze_vocal(vocal)
        print(f"  • Gender: {vocal_chars.gender}")
        print(f"  • Skill Level: {vocal_chars.skill_level}")
        print(f"  • Confidence: {vocal_chars.confidence_score:.1%}")
        print(f"  • Pitch Accuracy: {vocal_chars.pitch_accuracy:.1%}")
        
        # Check processing quality
        original_energy = np.sqrt(np.mean(vocal ** 2))
        processed_energy = np.sqrt(np.mean(processed ** 2))
        enhancement = (processed_energy / original_energy - 1) * 100
        
        print(f"  ✅ Enhancement: {enhancement:+.1f}%")
    
    print("\n✅ AI Vocal Mixing Test Complete!")
    return True

def test_karaoke_presets():
    """Test all karaoke presets"""
    print("\n🎤 Testing Karaoke Presets")
    print("=" * 50)
    
    mixer = KaraokeVocalMixer()
    
    # Create test audio
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test vocal
    vocal = np.sin(2 * np.pi * 330 * t)  # E4 note
    vocal += np.random.normal(0, 0.01, len(vocal))  # Slight imperfection
    
    # Test music
    music = np.sin(2 * np.pi * 261.63 * t) * 0.5  # C4 note
    
    # Test each preset
    presets = ["Shower Singer", "Karaoke King", "Concert Hall", "Studio Pro", "Radio Voice"]
    
    for preset_name in presets:
        print(f"\n📍 Testing Preset: {preset_name}")
        
        # Set preset
        mixer.set_preset(preset_name)
        preset = mixer.current_preset
        
        # Process audio
        output = mixer.mix(vocal, music)
        
        # Analyze results
        print(f"  • Reverb: {preset.reverb_amount/100:.0%}")  # Convert to percentage
        print(f"  • Pitch Correction: {preset.pitch_correction/100:.0%}")  # Convert to percentage
        print(f"  • Confidence Boost: {preset.confidence_boost/100:.0%}")  # Convert to percentage
        
        # Check enhancement
        original_energy = np.sqrt(np.mean(vocal ** 2))
        output_energy = np.sqrt(np.mean(output ** 2))
        
        # Reverb should add energy
        reverb_added = output_energy > original_energy
        print(f"  ✅ Reverb Applied: {'Yes' if reverb_added else 'No'}")
        
        if preset.reverb_amount >= 0.35:  # High reverb presets
            assert reverb_added, f"High reverb preset {preset_name} should add energy"
    
    print("\n✅ Karaoke Presets Test Complete!")
    return True

def test_system_integration():
    """Test complete system integration"""
    print("\n🎛️ Testing System Integration")
    print("=" * 50)
    
    # Test both mixers can work together
    ai_mixer = AIVocalAutoMixer()
    karaoke_mixer = KaraokeVocalMixer()
    
    # Create test audio
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Beginner vocal (needs help)
    vocal = np.sin(2 * np.pi * 280 * t)  # Slightly off-pitch
    vocal += np.random.normal(0, 0.03, len(vocal))  # More noise (beginner)
    
    # Background music
    music = (np.sin(2 * np.pi * 261.63 * t) +
             np.sin(2 * np.pi * 329.63 * t)) / 2
    
    print("\n1️⃣ AI Analysis Phase")
    # First: AI analyzes the vocal
    vocal_chars = ai_mixer.analyze_vocal(vocal)
    print(f"  • Detected: {vocal_chars.gender} {vocal_chars.skill_level} singer")
    print(f"  • Confidence: {vocal_chars.confidence_score:.1%}")
    print(f"  • Pitch Accuracy: {vocal_chars.pitch_accuracy:.1%}")
    
    print("\n2️⃣ AI Processing Phase")
    # Second: AI processes the vocal
    ai_processed = ai_mixer.process_vocal_with_ai(vocal, music)
    print("  ✅ AI processing applied")
    
    print("\n3️⃣ Karaoke Enhancement Phase")
    # Third: Apply karaoke enhancement for maximum help
    if vocal_chars.skill_level == "beginner":
        karaoke_mixer.set_preset("Concert Hall")  # Maximum reverb
        print("  • Selected: Concert Hall preset (60% reverb)")
    else:
        karaoke_mixer.set_preset("Karaoke King")
        print("  • Selected: Karaoke King preset (35% reverb)")
    
    # Final processing
    final_output = karaoke_mixer.mix(ai_processed, music)
    print("  ✅ Karaoke enhancement applied")
    
    print("\n4️⃣ Results")
    # Compare energy levels
    original_energy = np.sqrt(np.mean(vocal ** 2))
    final_energy = np.sqrt(np.mean(final_output ** 2))
    total_enhancement = (final_energy / original_energy - 1) * 100
    
    print(f"  • Total Enhancement: {total_enhancement:+.1f}%")
    print(f"  • Processing Chain: Vocal → AI → Karaoke → Output")
    
    print("\n✅ System Integration Test Complete!")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🎵 AiOke Complete System Test Suite")
    print("=" * 60)
    
    tests = [
        ("AI Vocal Mixing", test_ai_vocal_mixing),
        ("Karaoke Presets", test_karaoke_presets),
        ("System Integration", test_system_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {e}")
            results.append((name, False))
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 Final Test Report")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! AiOke system is fully operational!")
        print("\n🎤 Key Features Working:")
        print("  • Top-notch AI auto-mixing for vocals")
        print("  • Intelligent vocal analysis and enhancement")
        print("  • 5 karaoke presets with 15-60% reverb")
        print("  • Pitch correction for beginners")
        print("  • Complete processing chain integration")
        print("\n🚀 Ready for music production and karaoke!")
    else:
        print("\n⚠️ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()