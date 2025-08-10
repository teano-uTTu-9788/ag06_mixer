"""
Main Application for AG06 Mixer
Follows SOLID principles - Demonstrates proper architecture
"""
import asyncio
from typing import Optional

from core.dependency_container import DependencyContainer, ServiceRegistration, ServiceLocator
from core.workflow_orchestrator import AG06WorkflowOrchestrator, WorkflowTask, TaskType
from interfaces.audio_engine import AudioConfig
from interfaces.midi_controller import MidiMessage, MidiMessageType


class AG06MixerApplication:
    """Main application class - Single Responsibility: Application lifecycle"""
    
    def __init__(self):
        """Initialize application"""
        self._container: Optional[DependencyContainer] = None
        self._orchestrator: Optional[AG06WorkflowOrchestrator] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize application with dependency injection"""
        # Create DI container
        self._container = DependencyContainer()
        
        # Configure services
        ServiceRegistration.configure_services(self._container)
        
        # Set up service locator
        locator = ServiceLocator.get_instance()
        locator.container = self._container
        
        # Create and initialize orchestrator
        self._orchestrator = AG06WorkflowOrchestrator()
        await self._orchestrator.initialize()
        
        print("✅ AG06 Mixer Application initialized with SOLID architecture")
    
    async def start(self) -> None:
        """Start the application"""
        if not self._orchestrator:
            await self.initialize()
        
        self._running = True
        await self._orchestrator.start()
        
        print("✅ AG06 Mixer Application started")
    
    async def stop(self) -> None:
        """Stop the application"""
        self._running = False
        
        if self._orchestrator:
            await self._orchestrator.stop()
        
        print("✅ AG06 Mixer Application stopped")
    
    async def process_audio_task(self, audio_data: bytes) -> None:
        """Process an audio task"""
        task = WorkflowTask(
            type=TaskType.AUDIO_PROCESSING,
            parameters={'audio_data': audio_data},
            priority=1
        )
        
        result = await self._orchestrator.execute_task(task)
        
        if result.success:
            print(f"✅ Audio processed successfully")
        else:
            print(f"❌ Audio processing failed: {result.error}")
    
    async def send_midi_control(self, cc_number: int, value: int) -> None:
        """Send a MIDI control change"""
        message = MidiMessage(
            type=MidiMessageType.CONTROL_CHANGE,
            channel=0,
            data1=cc_number,
            data2=value
        )
        
        task = WorkflowTask(
            type=TaskType.MIDI_CONTROL,
            parameters={'midi_message': message},
            priority=2
        )
        
        result = await self._orchestrator.execute_task(task)
        
        if result.success:
            print(f"✅ MIDI control sent: CC{cc_number} = {value}")
        else:
            print(f"❌ MIDI control failed: {result.error}")
    
    async def load_preset(self, preset_name: str) -> None:
        """Load a preset"""
        task = WorkflowTask(
            type=TaskType.PRESET_MANAGEMENT,
            parameters={
                'action': 'load',
                'preset_name': preset_name
            },
            priority=0
        )
        
        result = await self._orchestrator.execute_task(task)
        
        if result.success:
            print(f"✅ Preset '{preset_name}' loaded")
        else:
            print(f"❌ Preset loading failed: {result.error}")
    
    async def process_karaoke(self, audio_data: bytes, remove_vocals: bool = True) -> None:
        """Process karaoke audio"""
        action = 'remove_vocals' if remove_vocals else 'enhance_vocals'
        
        task = WorkflowTask(
            type=TaskType.KARAOKE_PROCESSING,
            parameters={
                'action': action,
                'audio_data': audio_data,
                'enhancement_level': 1.5
            },
            priority=1
        )
        
        result = await self._orchestrator.execute_task(task)
        
        if result.success:
            print(f"✅ Karaoke processing complete: {action}")
        else:
            print(f"❌ Karaoke processing failed: {result.error}")


async def demonstrate_solid_architecture():
    """Demonstrate the SOLID-compliant architecture"""
    print("\n" + "="*60)
    print("AG06 MIXER - SOLID ARCHITECTURE DEMONSTRATION")
    print("="*60)
    
    # Create application
    app = AG06MixerApplication()
    
    # Initialize and start
    await app.initialize()
    await app.start()
    
    print("\n📋 SOLID Principles Demonstrated:")
    print("✅ Single Responsibility - Each class has one job")
    print("✅ Open/Closed - Extensible via handlers, closed for modification")
    print("✅ Liskov Substitution - All implementations follow interfaces")
    print("✅ Interface Segregation - Small, focused interfaces")
    print("✅ Dependency Inversion - Depend on abstractions, not concretions")
    
    print("\n🎯 Architecture Features:")
    print("• Dependency Injection Container")
    print("• Factory Pattern for object creation")
    print("• Interface-based design")
    print("• Separation of concerns")
    print("• Testable components")
    
    # Demonstrate some operations
    print("\n🔧 Testing Operations:")
    
    # Test audio processing
    test_audio = b'\x00' * 1024  # Mock audio data
    await app.process_audio_task(test_audio)
    
    # Test MIDI control
    await app.send_midi_control(7, 100)  # Volume control
    
    # Test preset loading
    await app.load_preset("Rock Vocals")
    
    # Test karaoke processing
    await app.process_karaoke(test_audio, remove_vocals=True)
    
    # Stop application
    await app.stop()
    
    print("\n" + "="*60)
    print("✅ SOLID ARCHITECTURE DEMONSTRATION COMPLETE")
    print("="*60)


def main():
    """Main entry point"""
    asyncio.run(demonstrate_solid_architecture())


if __name__ == "__main__":
    main()