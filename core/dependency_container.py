"""
Dependency Injection Container for AG06 Mixer
Follows SOLID principles - Dependency Inversion & Inversion of Control
"""
from typing import Dict, Any, Type, Callable, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
import asyncio
from functools import wraps


T = TypeVar('T')


class IServiceProvider(ABC):
    """Service provider interface - Dependency Inversion"""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance by type"""
        pass
    
    @abstractmethod
    def register_service(self, service_type: Type[T], factory: Callable[[], T], singleton: bool = False) -> None:
        """Register a service with factory"""
        pass


class ServiceLifetime:
    """Service lifetime enumeration"""
    TRANSIENT = "transient"  # New instance each time
    SINGLETON = "singleton"  # Single instance for app lifetime
    SCOPED = "scoped"       # Single instance per scope


class ServiceDescriptor:
    """Describes a registered service"""
    
    def __init__(self, 
                 service_type: Type,
                 factory: Callable,
                 lifetime: str = ServiceLifetime.TRANSIENT):
        self.service_type = service_type
        self.factory = factory
        self.lifetime = lifetime
        self.instance = None  # For singleton storage


class DependencyContainer(IServiceProvider):
    """Dependency injection container - Inversion of Control"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_instances: Dict[Type, Any] = {}
    
    def register_service(self, 
                        service_type: Type[T], 
                        factory: Callable[[], T], 
                        singleton: bool = False) -> None:
        """Register a service with its factory"""
        lifetime = ServiceLifetime.SINGLETON if singleton else ServiceLifetime.TRANSIENT
        descriptor = ServiceDescriptor(service_type, factory, lifetime)
        self._services[service_type] = descriptor
    
    def register_transient(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a transient service (new instance each time)"""
        descriptor = ServiceDescriptor(service_type, factory, ServiceLifetime.TRANSIENT)
        self._services[service_type] = descriptor
    
    def register_singleton(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a singleton service (single instance)"""
        descriptor = ServiceDescriptor(service_type, factory, ServiceLifetime.SINGLETON)
        self._services[service_type] = descriptor
    
    def register_scoped(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """Register a scoped service (single instance per scope)"""
        descriptor = ServiceDescriptor(service_type, factory, ServiceLifetime.SCOPED)
        self._services[service_type] = descriptor
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance by type"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        descriptor = self._services[service_type]
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is None:
                descriptor.instance = descriptor.factory()
            return descriptor.instance
        
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self._scoped_instances:
                self._scoped_instances[service_type] = descriptor.factory()
            return self._scoped_instances[service_type]
        
        else:  # TRANSIENT
            return descriptor.factory()
    
    def create_scope(self) -> 'ServiceScope':
        """Create a new service scope"""
        return ServiceScope(self)
    
    def clear_scoped_services(self):
        """Clear all scoped service instances"""
        self._scoped_instances.clear()


class ServiceScope:
    """Scoped service container"""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.scoped_instances: Dict[Type, Any] = {}
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service within this scope"""
        if service_type not in self.container._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        descriptor = self.container._services[service_type]
        
        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self.scoped_instances:
                self.scoped_instances[service_type] = descriptor.factory()
            return self.scoped_instances[service_type]
        else:
            return self.container.get_service(service_type)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scoped_instances.clear()


class ServiceLocator:
    """Service locator pattern - Alternative to constructor injection"""
    
    _instance: Optional['ServiceLocator'] = None
    
    def __init__(self):
        self.container = DependencyContainer()
    
    @classmethod
    def get_instance(cls) -> 'ServiceLocator':
        """Get singleton instance of service locator"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(self, service_type: Type[T], factory: Callable[[], T], lifetime: str = ServiceLifetime.TRANSIENT):
        """Register a service"""
        if lifetime == ServiceLifetime.SINGLETON:
            self.container.register_singleton(service_type, factory)
        elif lifetime == ServiceLifetime.SCOPED:
            self.container.register_scoped(service_type, factory)
        else:
            self.container.register_transient(service_type, factory)
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service"""
        return self.container.get_service(service_type)


def inject(*service_types):
    """Decorator for automatic dependency injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            locator = ServiceLocator.get_instance()
            services = [locator.resolve(st) for st in service_types]
            return func(*args, *services, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            locator = ServiceLocator.get_instance()
            services = [locator.resolve(st) for st in service_types]
            return await func(*args, *services, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


class ServiceRegistration:
    """Service registration helper"""
    
    @staticmethod
    def configure_services(container: DependencyContainer):
        """Configure all services for AG06 mixer"""
        from factories.component_factory import AG06ComponentFactory, IComponentFactory
        from interfaces.audio_engine import IAudioEngine
        from interfaces.midi_controller import IMidiController
        from interfaces.preset_manager import IPresetManager
        from interfaces.karaoke_integration import IVocalProcessor
        
        # Register component factory as singleton
        factory_instance = AG06ComponentFactory()
        container.register_singleton(
            IComponentFactory,
            lambda: factory_instance
        )
        
        # Register audio engine as singleton
        container.register_singleton(
            IAudioEngine,
            lambda: container.get_service(IComponentFactory).create_audio_engine()
        )
        
        # Register MIDI controller as singleton
        container.register_singleton(
            IMidiController,
            lambda: container.get_service(IComponentFactory).create_midi_controller()
        )
        
        # Register preset manager as singleton
        container.register_singleton(
            IPresetManager,
            lambda: container.get_service(IComponentFactory).create_preset_manager()
        )
        
        # Register vocal processor as transient
        container.register_transient(
            IVocalProcessor,
            lambda: container.get_service(IComponentFactory).create_vocal_processor()
        )