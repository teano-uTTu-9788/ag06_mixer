#!/usr/bin/env python3
"""
Security-First Framework 2025
Microsoft Zero Trust Inspired Security Implementation

Core Principles:
- Never trust, always verify
- Assume breach mentality
- Principle of least privilege
- Verify explicitly
- Use least privileged access
- Secure by design, secure by default

Based on Microsoft's Secure Future Initiative and Zero Trust architecture.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipaddress

# Security Configuration
@dataclass
class SecurityConfig:
    """Security configuration parameters"""
    # Authentication
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    jwt_expiry_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Encryption
    encryption_key: bytes = field(default_factory=Fernet.generate_key)
    password_min_length: int = 12
    require_mfa: bool = True
    
    # Network Security
    allowed_ips: List[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])
    require_https: bool = True
    tls_min_version: str = "1.2"
    
    # Audit
    audit_retention_days: int = 90
    log_failed_attempts: bool = True
    sensitive_data_masking: bool = True
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_per_ip: int = 60
    
    # Zero Trust
    verify_every_request: bool = True
    context_aware_access: bool = True
    continuous_validation: bool = True

# Identity and Access Management
class SecurityRole(Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"

class Permission(Enum):
    # Audio Operations
    AUDIO_PROCESS = "audio:process"
    AUDIO_RECORD = "audio:record"
    AUDIO_EXPORT = "audio:export"
    
    # Hardware Control
    HARDWARE_CONFIGURE = "hardware:configure"
    HARDWARE_DIAGNOSTICS = "hardware:diagnostics"
    HARDWARE_RESET = "hardware:reset"
    
    # System Operations
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIGURE = "system:configure"
    SYSTEM_SHUTDOWN = "system:shutdown"
    
    # User Management
    USER_CREATE = "user:create"
    USER_DELETE = "user:delete"
    USER_MODIFY = "user:modify"

@dataclass
class SecurityContext:
    """Security context for request evaluation"""
    user_id: str
    roles: Set[SecurityRole]
    permissions: Set[Permission]
    ip_address: str
    user_agent: str
    timestamp: datetime
    session_id: str
    device_fingerprint: Optional[str] = None
    location: Optional[str] = None
    risk_score: float = 0.0

@dataclass
class User:
    """User with security attributes"""
    user_id: str
    username: str
    password_hash: str
    roles: Set[SecurityRole]
    is_active: bool = True
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    locked_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    mfa_secret: Optional[str] = None
    api_keys: Set[str] = field(default_factory=set)

@dataclass 
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    severity: str  # low, medium, high, critical
    user_id: Optional[str]
    ip_address: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

# Cryptography Services
class CryptographyService:
    """Microsoft-inspired cryptography service"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = Fernet(config.encryption_key)
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    def hash_password(self, password: str) -> str:
        """Secure password hashing using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode('utf-8')).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
    
    def encrypt_with_rsa(self, data: str) -> str:
        """RSA encryption for key exchange"""
        encrypted = self.rsa_public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_with_rsa(self, encrypted_data: str) -> str:
        """RSA decryption"""
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted = self.rsa_private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode('utf-8')
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self) -> str:
        """Generate API key with prefix"""
        return f"ag06_{secrets.token_urlsafe(32)}"

# Authentication Service
class AuthenticationService:
    """Zero Trust authentication service"""
    
    def __init__(self, config: SecurityConfig, crypto_service: CryptographyService):
        self.config = config
        self.crypto = crypto_service
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.rate_limits: Dict[str, List[float]] = {}  # IP -> [timestamps]
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = self.crypto.generate_secure_token(16)
        admin_user = User(
            user_id=str(uuid.uuid4()),
            username="admin",
            password_hash=self.crypto.hash_password(admin_password),
            roles={SecurityRole.ADMIN}
        )
        self.users[admin_user.user_id] = admin_user
        logging.warning(f"Created default admin user - Password: {admin_password}")
    
    async def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user with Zero Trust principles"""
        
        # Rate limiting check
        if not self._check_rate_limit(ip_address):
            await self._log_security_event("RATE_LIMIT_EXCEEDED", "medium", None, ip_address, 
                                          {"username": username})
            raise SecurityException("Rate limit exceeded")
        
        # Find user
        user = self._find_user_by_username(username)
        if not user:
            await self._log_security_event("INVALID_USERNAME", "low", None, ip_address, 
                                          {"username": username})
            return None
        
        # Check if user is locked
        if self._is_user_locked(user):
            await self._log_security_event("LOGIN_ATTEMPT_LOCKED_USER", "medium", user.user_id, ip_address, 
                                          {"username": username})
            raise SecurityException("Account is locked")
        
        # Verify password
        if not self.crypto.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            user.last_failed_login = datetime.now()
            
            # Lock user if too many failed attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
                await self._log_security_event("USER_LOCKED", "high", user.user_id, ip_address, 
                                              {"reason": "too_many_failed_attempts"})
            
            await self._log_security_event("INVALID_PASSWORD", "medium", user.user_id, ip_address, 
                                          {"username": username, "failed_attempts": user.failed_login_attempts})
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create security context
        session_id = self.crypto.generate_secure_token()
        security_context = SecurityContext(
            user_id=user.user_id,
            roles=user.roles,
            permissions=self._get_permissions_for_roles(user.roles),
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            session_id=session_id,
            risk_score=self._calculate_risk_score(user, ip_address, user_agent)
        )
        
        self.active_sessions[session_id] = security_context
        
        # Generate JWT token
        token = self._generate_jwt_token(security_context)
        
        await self._log_security_event("SUCCESSFUL_LOGIN", "low", user.user_id, ip_address, 
                                      {"username": username, "session_id": session_id})
        
        return token
    
    async def validate_token(self, token: str, ip_address: str) -> Optional[SecurityContext]:
        """Validate JWT token with continuous verification"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            if not session_id or session_id not in self.active_sessions:
                await self._log_security_event("INVALID_SESSION", "medium", None, ip_address, 
                                              {"token_payload": payload})
                return None
            
            context = self.active_sessions[session_id]
            
            # Continuous validation checks
            if self.config.continuous_validation:
                # Check IP address consistency
                if context.ip_address != ip_address:
                    await self._log_security_event("IP_ADDRESS_CHANGED", "high", context.user_id, ip_address, 
                                                  {"original_ip": context.ip_address, "new_ip": ip_address})
                    # In Zero Trust, we might allow this but increase risk score
                    context.risk_score += 0.3
                
                # Check session age
                session_age = datetime.now() - context.timestamp
                if session_age > timedelta(minutes=self.config.jwt_expiry_minutes):
                    await self._log_security_event("SESSION_EXPIRED", "low", context.user_id, ip_address, 
                                                  {"session_age_minutes": session_age.total_seconds() / 60})
                    del self.active_sessions[session_id]
                    return None
                
                # Update timestamp for activity
                context.timestamp = datetime.now()
            
            return context
            
        except jwt.ExpiredSignatureError:
            await self._log_security_event("TOKEN_EXPIRED", "low", None, ip_address, {})
            return None
        except jwt.InvalidTokenError as e:
            await self._log_security_event("INVALID_TOKEN", "medium", None, ip_address, 
                                          {"error": str(e)})
            return None
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP is within rate limits"""
        now = time.time()
        minute_ago = now - 60
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Remove old entries
        self.rate_limits[ip_address] = [
            timestamp for timestamp in self.rate_limits[ip_address]
            if timestamp > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limits[ip_address]) >= self.config.rate_limit_per_ip:
            return False
        
        # Record this request
        self.rate_limits[ip_address].append(now)
        return True
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _is_user_locked(self, user: User) -> bool:
        """Check if user is currently locked"""
        if user.locked_until and user.locked_until > datetime.now():
            return True
        return False
    
    def _get_permissions_for_roles(self, roles: Set[SecurityRole]) -> Set[Permission]:
        """Get permissions for given roles"""
        permissions = set()
        
        role_permissions = {
            SecurityRole.ADMIN: {
                Permission.AUDIO_PROCESS, Permission.AUDIO_RECORD, Permission.AUDIO_EXPORT,
                Permission.HARDWARE_CONFIGURE, Permission.HARDWARE_DIAGNOSTICS, Permission.HARDWARE_RESET,
                Permission.SYSTEM_MONITOR, Permission.SYSTEM_CONFIGURE, Permission.SYSTEM_SHUTDOWN,
                Permission.USER_CREATE, Permission.USER_DELETE, Permission.USER_MODIFY
            },
            SecurityRole.USER: {
                Permission.AUDIO_PROCESS, Permission.AUDIO_RECORD,
                Permission.HARDWARE_DIAGNOSTICS, Permission.SYSTEM_MONITOR
            },
            SecurityRole.READONLY: {
                Permission.SYSTEM_MONITOR
            },
            SecurityRole.SERVICE: {
                Permission.AUDIO_PROCESS, Permission.HARDWARE_DIAGNOSTICS, Permission.SYSTEM_MONITOR
            }
        }
        
        for role in roles:
            permissions.update(role_permissions.get(role, set()))
        
        return permissions
    
    def _calculate_risk_score(self, user: User, ip_address: str, user_agent: str) -> float:
        """Calculate risk score for security context"""
        risk_score = 0.0
        
        # Check IP address
        if ip_address not in self.config.allowed_ips:
            risk_score += 0.2
        
        # Check for suspicious user agent
        suspicious_agents = ['curl', 'wget', 'python-requests']
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            risk_score += 0.3
        
        # Check failed login history
        if user.failed_login_attempts > 0:
            risk_score += user.failed_login_attempts * 0.1
        
        # Check time since last login
        if user.last_login:
            time_since_login = datetime.now() - user.last_login
            if time_since_login > timedelta(days=30):
                risk_score += 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _generate_jwt_token(self, context: SecurityContext) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'roles': [role.value for role in context.roles],
            'permissions': [perm.value for perm in context.permissions],
            'risk_score': context.risk_score,
            'iat': int(time.time()),
            'exp': int(time.time()) + (self.config.jwt_expiry_minutes * 60)
        }
        return jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
    
    async def _log_security_event(self, event_type: str, severity: str, user_id: Optional[str], 
                                ip_address: str, details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details
        )
        
        # In production, this would go to a security information and event management (SIEM) system
        logging.warning(f"Security Event [{severity.upper()}]: {event_type} from {ip_address} - {details}")

# Authorization Service
class AuthorizationService:
    """Policy-based authorization service"""
    
    def __init__(self):
        self.policies: Dict[str, 'AccessPolicy'] = {}
        self._create_default_policies()
    
    def _create_default_policies(self):
        """Create default access policies"""
        # Audio processing policy
        self.policies['audio_operations'] = AccessPolicy(
            name="audio_operations",
            required_permissions={Permission.AUDIO_PROCESS},
            allowed_roles={SecurityRole.ADMIN, SecurityRole.USER},
            max_risk_score=0.7,
            require_secure_transport=True
        )
        
        # Hardware control policy
        self.policies['hardware_control'] = AccessPolicy(
            name="hardware_control",
            required_permissions={Permission.HARDWARE_CONFIGURE},
            allowed_roles={SecurityRole.ADMIN},
            max_risk_score=0.3,
            require_secure_transport=True,
            allowed_ips=["127.0.0.1", "::1"]  # Local only for hardware control
        )
        
        # System administration policy
        self.policies['system_admin'] = AccessPolicy(
            name="system_admin",
            required_permissions={Permission.SYSTEM_CONFIGURE},
            allowed_roles={SecurityRole.ADMIN},
            max_risk_score=0.2,
            require_secure_transport=True,
            require_mfa=True
        )
    
    async def authorize_request(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Authorize request based on context and policies"""
        policy_name = f"{resource}_{action}"
        policy = self.policies.get(policy_name)
        
        if not policy:
            # Default deny for unknown policies
            logging.warning(f"No policy found for {policy_name} - defaulting to deny")
            return False
        
        return await policy.evaluate(context)

@dataclass
class AccessPolicy:
    """Access control policy"""
    name: str
    required_permissions: Set[Permission]
    allowed_roles: Set[SecurityRole] = field(default_factory=set)
    max_risk_score: float = 1.0
    require_secure_transport: bool = False
    require_mfa: bool = False
    allowed_ips: Optional[List[str]] = None
    time_based_access: Optional[Dict[str, Any]] = None
    
    async def evaluate(self, context: SecurityContext) -> bool:
        """Evaluate policy against security context"""
        
        # Check required permissions
        if not self.required_permissions.issubset(context.permissions):
            return False
        
        # Check allowed roles
        if self.allowed_roles and not context.roles.intersection(self.allowed_roles):
            return False
        
        # Check risk score
        if context.risk_score > self.max_risk_score:
            return False
        
        # Check IP restrictions
        if self.allowed_ips:
            if context.ip_address not in self.allowed_ips:
                try:
                    # Check if IP is in allowed subnets
                    client_ip = ipaddress.ip_address(context.ip_address)
                    allowed = False
                    for allowed_ip in self.allowed_ips:
                        if '/' in allowed_ip:  # Subnet notation
                            if client_ip in ipaddress.ip_network(allowed_ip, strict=False):
                                allowed = True
                                break
                        else:  # Single IP
                            if client_ip == ipaddress.ip_address(allowed_ip):
                                allowed = True
                                break
                    
                    if not allowed:
                        return False
                except ValueError:
                    return False
        
        # Check time-based access (if configured)
        if self.time_based_access:
            # Implementation would check current time against allowed time windows
            pass
        
        return True

# Input Validation Service
class InputValidationService:
    """Secure input validation service"""
    
    def __init__(self):
        self.max_string_length = 1000
        self.allowed_audio_formats = {'wav', 'mp3', 'aac', 'flac'}
        self.dangerous_patterns = [
            r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
            r'\.\./\.\.',  # Path traversal
            r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL injection
            r'[;&|`$]',  # Command injection
        ]
    
    def validate_string(self, value: str, field_name: str, max_length: Optional[int] = None) -> str:
        """Validate string input"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        max_len = max_length or self.max_string_length
        if len(value) > max_len:
            raise ValidationError(f"{field_name} exceeds maximum length of {max_len}")
        
        # Check for dangerous patterns
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError(f"{field_name} contains potentially dangerous content")
        
        return value.strip()
    
    def validate_audio_format(self, format_name: str) -> str:
        """Validate audio format"""
        format_name = format_name.lower().strip()
        if format_name not in self.allowed_audio_formats:
            raise ValidationError(f"Unsupported audio format: {format_name}")
        return format_name
    
    def validate_numeric_range(self, value: Union[int, float], field_name: str, 
                             min_val: Optional[Union[int, float]] = None,
                             max_val: Optional[Union[int, float]] = None) -> Union[int, float]:
        """Validate numeric input within range"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{field_name} must be at least {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{field_name} must be at most {max_val}")
        
        return value
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        import os
        # Remove path components and dangerous characters
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        
        if not filename or filename.startswith('.'):
            raise ValidationError("Invalid filename")
        
        return filename

# Security Exceptions
class SecurityException(Exception):
    """Security-related exception"""
    pass

class ValidationError(Exception):
    """Input validation error"""
    pass

# Secure Request Handler Decorator
class SecureRequestHandler:
    """Decorator for secure request handling"""
    
    def __init__(self, auth_service: AuthenticationService, 
                 authz_service: AuthorizationService,
                 validation_service: InputValidationService):
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.validation_service = validation_service
    
    def require_auth(self, resource: str, action: str):
        """Decorator to require authentication and authorization"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract security context (this would come from HTTP headers in practice)
                token = kwargs.get('token')
                ip_address = kwargs.get('ip_address', '127.0.0.1')
                
                if not token:
                    raise SecurityException("Authentication required")
                
                # Validate token and get context
                context = await self.auth_service.validate_token(token, ip_address)
                if not context:
                    raise SecurityException("Invalid or expired token")
                
                # Authorize request
                authorized = await self.authz_service.authorize_request(context, resource, action)
                if not authorized:
                    raise SecurityException("Access denied")
                
                # Add security context to kwargs
                kwargs['security_context'] = context
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Main Security Framework
class SecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Initialize services
        self.crypto_service = CryptographyService(self.config)
        self.auth_service = AuthenticationService(self.config, self.crypto_service)
        self.authz_service = AuthorizationService()
        self.validation_service = InputValidationService()
        
        # Create secure request handler
        self.secure_handler = SecureRequestHandler(
            self.auth_service, 
            self.authz_service, 
            self.validation_service
        )
        
        self.security_events: List[SecurityEvent] = []
    
    async def initialize(self):
        """Initialize security framework"""
        logging.info("Security framework initialized with Zero Trust principles")
        logging.info(f"Configuration: MFA={self.config.require_mfa}, HTTPS={self.config.require_https}")
        
        # Start security monitoring
        asyncio.create_task(self._security_monitoring_loop())
    
    async def create_user(self, username: str, password: str, roles: Set[SecurityRole]) -> str:
        """Create new user with security validation"""
        # Validate password strength
        if len(password) < self.config.password_min_length:
            raise ValidationError(f"Password must be at least {self.config.password_min_length} characters")
        
        # Create user
        user_id = str(uuid.uuid4())
        user = User(
            user_id=user_id,
            username=username,
            password_hash=self.crypto_service.hash_password(password),
            roles=roles
        )
        
        self.auth_service.users[user_id] = user
        
        logging.info(f"Created user: {username} with roles: {[r.value for r in roles]}")
        return user_id
    
    async def create_api_key(self, user_id: str) -> str:
        """Create API key for user"""
        user = self.auth_service.users.get(user_id)
        if not user:
            raise SecurityException("User not found")
        
        api_key = self.crypto_service.generate_api_key()
        user.api_keys.add(api_key)
        
        return api_key
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get security framework status"""
        return {
            "active_sessions": len(self.auth_service.active_sessions),
            "total_users": len(self.auth_service.users),
            "security_events_24h": len([
                event for event in self.security_events
                if (datetime.now() - event.timestamp).days == 0
            ]),
            "high_risk_sessions": len([
                context for context in self.auth_service.active_sessions.values()
                if context.risk_score > 0.7
            ]),
            "config": {
                "mfa_required": self.config.require_mfa,
                "https_required": self.config.require_https,
                "continuous_validation": self.config.continuous_validation
            }
        }
    
    async def _security_monitoring_loop(self):
        """Continuous security monitoring"""
        while True:
            try:
                # Monitor for suspicious activity
                current_time = datetime.now()
                
                # Check for inactive sessions
                inactive_sessions = [
                    session_id for session_id, context in self.auth_service.active_sessions.items()
                    if (current_time - context.timestamp) > timedelta(hours=1)
                ]
                
                # Clean up inactive sessions
                for session_id in inactive_sessions:
                    del self.auth_service.active_sessions[session_id]
                    logging.info(f"Cleaned up inactive session: {session_id}")
                
                # Check for high-risk sessions
                high_risk_sessions = [
                    context for context in self.auth_service.active_sessions.values()
                    if context.risk_score > 0.8
                ]
                
                for context in high_risk_sessions:
                    logging.warning(f"High-risk session detected: {context.session_id} (risk: {context.risk_score})")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Security monitoring error: {e}")
                await asyncio.sleep(5)

# Example usage
async def main():
    """Example usage of security framework"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize security framework
    security = SecurityFramework()
    await security.initialize()
    
    try:
        # Create a user
        user_id = await security.create_user(
            username="testuser",
            password="SecurePassword123!",
            roles={SecurityRole.USER}
        )
        print(f"Created user: {user_id}")
        
        # Authenticate user
        token = await security.auth_service.authenticate_user(
            username="testuser",
            password="SecurePassword123!",
            ip_address="127.0.0.1",
            user_agent="AG06-App/1.0"
        )
        print(f"Authentication successful: {token[:20]}...")
        
        # Validate token
        context = await security.auth_service.validate_token(token, "127.0.0.1")
        if context:
            print(f"Token valid - Risk score: {context.risk_score}")
        
        # Test authorization
        authorized = await security.authz_service.authorize_request(
            context, "audio", "operations"
        )
        print(f"Audio operations authorized: {authorized}")
        
        # Get security status
        status = await security.get_security_status()
        print(f"Security status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())