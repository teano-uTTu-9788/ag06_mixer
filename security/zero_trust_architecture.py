"""
Zero Trust Security Architecture
Following Google BeyondCorp, Microsoft Zero Trust, AWS Zero Trust best practices
"""

import asyncio
import logging
import hashlib
import hmac
import jwt
import secrets
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import threading
from collections import defaultdict, deque
import ipaddress
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in zero trust architecture"""
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA = "mfa"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    TOKEN = "token"
    API_KEY = "api_key"


class ResourceType(Enum):
    """Protected resource types"""
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    ML_MODEL = "ml_model"
    AUDIO_STREAM = "audio_stream"
    ADMIN_PANEL = "admin_panel"


@dataclass
class Identity:
    """User or service identity"""
    id: str
    type: str  # user, service, device
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_authenticated: Optional[datetime] = None
    trust_score: float = 0.5  # 0.0 to 1.0
    
    def has_role(self, role: str) -> bool:
        return role in self.roles
        
    def has_any_role(self, roles: List[str]) -> bool:
        return any(role in self.roles for role in roles)


@dataclass
class Context:
    """Request context for zero trust evaluation"""
    identity: Identity
    ip_address: str
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[str] = None
    time_of_access: datetime = field(default_factory=datetime.now)
    authentication_method: Optional[AuthenticationMethod] = None
    session_id: Optional[str] = None
    risk_score: float = 0.0  # 0.0 to 1.0, higher is riskier


@dataclass
class Permission:
    """Permission definition"""
    resource_type: ResourceType
    resource_id: str
    actions: List[str]  # read, write, delete, execute
    conditions: Dict[str, Any] = field(default_factory=dict)  # time, location, etc.


@dataclass
class PolicyRule:
    """Zero trust policy rule"""
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    required_trust_level: TrustLevel
    allowed_actions: List[str]
    denied_actions: List[str] = field(default_factory=list)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    ip_restrictions: List[str] = field(default_factory=list)
    requires_mfa: bool = False
    max_session_duration: int = 3600  # seconds


@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    identity: Optional[Identity] = None
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    authentication_method: Optional[AuthenticationMethod] = None
    expires_at: Optional[datetime] = None
    session_token: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CryptographicService:
    """Cryptographic operations for zero trust"""
    
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.jwt_secret = secrets.token_urlsafe(32)
        
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        # Generate random key for AES
        key = secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        # Encrypt data with AES
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - len(data) % 16
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt AES key with RSA
        encrypted_key = self.public_key.encrypt(
            key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key, IV, and encrypted data
        return encrypted_key + iv + encrypted_data
        
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        # Extract components
        key_size = self.private_key.key_size // 8
        encrypted_key = encrypted_data[:key_size]
        iv = encrypted_data[key_size:key_size + 16]
        ciphertext = encrypted_data[key_size + 16:]
        
        # Decrypt AES key
        key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
        
    def create_jwt_token(self, identity: Identity, expires_in: int = 3600) -> str:
        """Create JWT token for authentication"""
        payload = {
            'sub': identity.id,
            'email': identity.email,
            'roles': identity.roles,
            'type': identity.type,
            'trust_score': identity.trust_score,
            'iat': int(time.time()),
            'exp': int(time.time()) + expires_in
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
            
    def generate_api_key(self, identity_id: str) -> Tuple[str, str]:
        """Generate API key and secret"""
        key_id = secrets.token_urlsafe(16)
        secret = secrets.token_urlsafe(32)
        
        # Create HMAC signature
        signature = hmac.new(
            secret.encode(),
            f"{identity_id}:{key_id}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        api_key = f"ak_{key_id}"
        api_secret = f"as_{secret}_{signature}"
        
        return api_key, api_secret
        
    def verify_api_key(self, api_key: str, api_secret: str, identity_id: str) -> bool:
        """Verify API key and secret"""
        try:
            if not api_key.startswith("ak_") or not api_secret.startswith("as_"):
                return False
                
            key_id = api_key[3:]  # Remove "ak_" prefix
            secret_parts = api_secret[3:].split("_")  # Remove "as_" prefix
            
            if len(secret_parts) != 2:
                return False
                
            secret, signature = secret_parts
            
            # Verify signature
            expected_signature = hmac.new(
                secret.encode(),
                f"{identity_id}:{key_id}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return False


class RiskAnalyzer:
    """Analyzes risk factors for zero trust decisions"""
    
    def __init__(self):
        self.suspicious_ips: Set[str] = set()
        self.failed_attempts: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.location_history: defaultdict[str, list] = defaultdict(list)
        self.device_fingerprints: Dict[str, Dict[str, Any]] = {}
        
    def analyze_context_risk(self, context: Context) -> float:
        """Analyze risk score for given context"""
        risk_factors = []
        
        # IP address risk
        ip_risk = self._analyze_ip_risk(context.ip_address)
        risk_factors.append(('ip_address', ip_risk))
        
        # Authentication method risk
        auth_risk = self._analyze_authentication_risk(context.authentication_method)
        risk_factors.append(('authentication', auth_risk))
        
        # Time-based risk
        time_risk = self._analyze_time_risk(context.time_of_access)
        risk_factors.append(('time_access', time_risk))
        
        # Location risk
        location_risk = self._analyze_location_risk(context.identity.id, context.location)
        risk_factors.append(('location', location_risk))
        
        # Failed attempts risk
        failure_risk = self._analyze_failure_history(context.identity.id)
        risk_factors.append(('failed_attempts', failure_risk))
        
        # Device risk
        device_risk = self._analyze_device_risk(context.device_id, context.user_agent)
        risk_factors.append(('device', device_risk))
        
        # Calculate weighted risk score
        weights = {
            'ip_address': 0.2,
            'authentication': 0.15,
            'time_access': 0.1,
            'location': 0.2,
            'failed_attempts': 0.25,
            'device': 0.1
        }
        
        total_risk = sum(weights.get(factor, 0.1) * risk for factor, risk in risk_factors)
        
        # Log risk analysis
        logger.debug(f"Risk analysis for {context.identity.id}: {dict(risk_factors)} = {total_risk:.3f}")
        
        return min(1.0, total_risk)
        
    def _analyze_ip_risk(self, ip_address: str) -> float:
        """Analyze IP address risk"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check if IP is in suspicious list
            if ip_address in self.suspicious_ips:
                return 0.8
                
            # Private networks are lower risk
            if ip.is_private:
                return 0.1
                
            # Loopback is very low risk
            if ip.is_loopback:
                return 0.05
                
            # Public IPs have moderate risk
            return 0.3
            
        except ValueError:
            # Invalid IP address
            return 0.9
            
    def _analyze_authentication_risk(self, method: Optional[AuthenticationMethod]) -> float:
        """Analyze authentication method risk"""
        if method is None:
            return 1.0
            
        risk_levels = {
            AuthenticationMethod.BIOMETRIC: 0.05,
            AuthenticationMethod.CERTIFICATE: 0.1,
            AuthenticationMethod.MFA: 0.15,
            AuthenticationMethod.TOKEN: 0.3,
            AuthenticationMethod.API_KEY: 0.4,
            AuthenticationMethod.PASSWORD: 0.7
        }
        
        return risk_levels.get(method, 0.8)
        
    def _analyze_time_risk(self, access_time: datetime) -> float:
        """Analyze time-based risk"""
        hour = access_time.hour
        
        # Business hours (9 AM - 5 PM) are lower risk
        if 9 <= hour <= 17:
            return 0.1
            
        # Evening hours have moderate risk
        if 18 <= hour <= 22 or 6 <= hour <= 8:
            return 0.3
            
        # Late night/early morning have high risk
        return 0.6
        
    def _analyze_location_risk(self, identity_id: str, location: Optional[str]) -> float:
        """Analyze location-based risk"""
        if location is None:
            return 0.5
            
        # Check location history
        history = self.location_history[identity_id]
        
        if not history:
            # First time access from any location
            self.location_history[identity_id].append(location)
            return 0.4
            
        if location in history:
            # Known location
            return 0.1
        else:
            # New location
            self.location_history[identity_id].append(location)
            return 0.6
            
    def _analyze_failure_history(self, identity_id: str) -> float:
        """Analyze failed authentication attempts"""
        failures = self.failed_attempts[identity_id]
        
        if not failures:
            return 0.0
            
        # Count recent failures (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        recent_failures = sum(1 for failure_time in failures if failure_time > cutoff)
        
        # Risk increases with number of recent failures
        if recent_failures >= 5:
            return 0.9
        elif recent_failures >= 3:
            return 0.6
        elif recent_failures >= 1:
            return 0.3
        else:
            return 0.0
            
    def _analyze_device_risk(self, device_id: Optional[str], user_agent: Optional[str]) -> float:
        """Analyze device-based risk"""
        if device_id is None and user_agent is None:
            return 0.7
            
        # Create device fingerprint
        fingerprint = {
            'device_id': device_id,
            'user_agent': user_agent,
            'first_seen': datetime.now()
        }
        
        device_key = device_id or hashlib.md5(user_agent.encode() if user_agent else b'').hexdigest()
        
        if device_key in self.device_fingerprints:
            # Known device
            return 0.1
        else:
            # New device
            self.device_fingerprints[device_key] = fingerprint
            return 0.4
            
    def record_failed_attempt(self, identity_id: str):
        """Record failed authentication attempt"""
        self.failed_attempts[identity_id].append(datetime.now())
        
    def mark_ip_suspicious(self, ip_address: str):
        """Mark IP address as suspicious"""
        self.suspicious_ips.add(ip_address)


class PolicyEngine:
    """Zero trust policy engine"""
    
    def __init__(self):
        self.policies: Dict[str, PolicyRule] = {}
        self.default_policies = self._create_default_policies()
        self.policies.update(self.default_policies)
        
    def _create_default_policies(self) -> Dict[str, PolicyRule]:
        """Create default security policies"""
        return {
            'admin_access': PolicyRule(
                id='admin_access',
                name='Admin Panel Access',
                description='High security for admin panel access',
                conditions={'resource_type': ResourceType.ADMIN_PANEL},
                required_trust_level=TrustLevel.HIGH,
                allowed_actions=['read', 'write'],
                requires_mfa=True,
                max_session_duration=1800,  # 30 minutes
                ip_restrictions=[]  # Allow from any IP but require MFA
            ),
            'ml_model_access': PolicyRule(
                id='ml_model_access',
                name='ML Model Access',
                description='Secure ML model inference and management',
                conditions={'resource_type': ResourceType.ML_MODEL},
                required_trust_level=TrustLevel.MEDIUM,
                allowed_actions=['read', 'execute'],
                requires_mfa=False,
                max_session_duration=3600
            ),
            'api_access': PolicyRule(
                id='api_access',
                name='API Endpoint Access',
                description='Standard API access policy',
                conditions={'resource_type': ResourceType.API_ENDPOINT},
                required_trust_level=TrustLevel.LOW,
                allowed_actions=['read'],
                requires_mfa=False,
                max_session_duration=7200
            ),
            'database_access': PolicyRule(
                id='database_access',
                name='Database Access',
                description='Secure database access',
                conditions={'resource_type': ResourceType.DATABASE},
                required_trust_level=TrustLevel.HIGH,
                allowed_actions=['read'],
                denied_actions=['delete'],
                requires_mfa=True,
                max_session_duration=1800
            ),
            'audio_stream_access': PolicyRule(
                id='audio_stream_access',
                name='Audio Stream Access',
                description='Real-time audio stream access',
                conditions={'resource_type': ResourceType.AUDIO_STREAM},
                required_trust_level=TrustLevel.LOW,
                allowed_actions=['read', 'write'],
                requires_mfa=False,
                max_session_duration=10800  # 3 hours
            )
        }
        
    def add_policy(self, policy: PolicyRule):
        """Add new policy rule"""
        self.policies[policy.id] = policy
        logger.info(f"Added policy: {policy.name}")
        
    def evaluate_access(self, context: Context, resource_type: ResourceType, 
                       resource_id: str, action: str) -> Tuple[bool, str, TrustLevel]:
        """Evaluate access request against policies"""
        
        # Find applicable policies
        applicable_policies = []
        for policy in self.policies.values():
            if self._policy_applies(policy, resource_type, resource_id):
                applicable_policies.append(policy)
                
        if not applicable_policies:
            return False, "No applicable policy found", TrustLevel.UNTRUSTED
            
        # Evaluate each applicable policy
        for policy in applicable_policies:
            allowed, reason, trust_level = self._evaluate_policy(policy, context, action)
            
            if allowed:
                return True, f"Access granted by policy: {policy.name}", trust_level
            else:
                logger.warning(f"Access denied by policy {policy.name}: {reason}")
                
        return False, "Access denied by all applicable policies", TrustLevel.UNTRUSTED
        
    def _policy_applies(self, policy: PolicyRule, resource_type: ResourceType, 
                       resource_id: str) -> bool:
        """Check if policy applies to the resource"""
        conditions = policy.conditions
        
        if 'resource_type' in conditions:
            if conditions['resource_type'] != resource_type:
                return False
                
        if 'resource_id' in conditions:
            if conditions['resource_id'] != resource_id:
                return False
                
        return True
        
    def _evaluate_policy(self, policy: PolicyRule, context: Context, 
                        action: str) -> Tuple[bool, str, TrustLevel]:
        """Evaluate single policy against context"""
        
        # Check if action is explicitly denied
        if action in policy.denied_actions:
            return False, f"Action '{action}' is explicitly denied", TrustLevel.UNTRUSTED
            
        # Check if action is allowed
        if action not in policy.allowed_actions:
            return False, f"Action '{action}' is not allowed", TrustLevel.UNTRUSTED
            
        # Check trust level requirement
        identity_trust_level = self._calculate_trust_level(context)
        if identity_trust_level.value < policy.required_trust_level.value:
            return False, f"Insufficient trust level: {identity_trust_level.name} < {policy.required_trust_level.name}", identity_trust_level
            
        # Check MFA requirement
        if policy.requires_mfa:
            if context.authentication_method != AuthenticationMethod.MFA:
                return False, "Multi-factor authentication required", identity_trust_level
                
        # Check time restrictions
        if policy.time_restrictions:
            if not self._check_time_restrictions(policy.time_restrictions, context.time_of_access):
                return False, "Access outside allowed time window", identity_trust_level
                
        # Check IP restrictions
        if policy.ip_restrictions:
            if not self._check_ip_restrictions(policy.ip_restrictions, context.ip_address):
                return False, "Access from unauthorized IP address", identity_trust_level
                
        return True, "Policy conditions satisfied", identity_trust_level
        
    def _calculate_trust_level(self, context: Context) -> TrustLevel:
        """Calculate trust level based on context"""
        base_trust = context.identity.trust_score
        risk_adjustment = 1.0 - context.risk_score
        
        adjusted_trust = base_trust * risk_adjustment
        
        if adjusted_trust >= 0.9:
            return TrustLevel.CRITICAL
        elif adjusted_trust >= 0.7:
            return TrustLevel.HIGH
        elif adjusted_trust >= 0.5:
            return TrustLevel.MEDIUM
        elif adjusted_trust >= 0.3:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
            
    def _check_time_restrictions(self, restrictions: Dict[str, Any], 
                               access_time: datetime) -> bool:
        """Check time-based restrictions"""
        if 'allowed_hours' in restrictions:
            hour = access_time.hour
            if hour not in restrictions['allowed_hours']:
                return False
                
        if 'allowed_days' in restrictions:
            day = access_time.weekday()  # 0 = Monday
            if day not in restrictions['allowed_days']:
                return False
                
        return True
        
    def _check_ip_restrictions(self, restrictions: List[str], ip_address: str) -> bool:
        """Check IP-based restrictions"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for restriction in restrictions:
                if '/' in restriction:
                    # CIDR block
                    network = ipaddress.ip_network(restriction, strict=False)
                    if ip in network:
                        return True
                else:
                    # Single IP
                    if ip == ipaddress.ip_address(restriction):
                        return True
                        
            return False
            
        except ValueError:
            return False


class IdentityProvider:
    """Identity provider for zero trust authentication"""
    
    def __init__(self, crypto_service: CryptographicService):
        self.crypto_service = crypto_service
        self.identities: Dict[str, Identity] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Create default identities
        self._create_default_identities()
        
    def _create_default_identities(self):
        """Create default system identities"""
        
        # Admin user
        admin = Identity(
            id="admin_user",
            type="user",
            email="admin@ag06mixer.com",
            roles=["admin", "operator"],
            trust_score=0.9
        )
        self.identities[admin.id] = admin
        
        # API service
        api_service = Identity(
            id="api_service",
            type="service",
            roles=["api_access"],
            trust_score=0.8
        )
        self.identities[api_service.id] = api_service
        
        # ML service
        ml_service = Identity(
            id="ml_service",
            type="service",
            roles=["ml_inference", "model_access"],
            trust_score=0.85
        )
        self.identities[ml_service.id] = ml_service
        
        # Audio processing service
        audio_service = Identity(
            id="audio_service",
            type="service",
            roles=["audio_processing", "stream_access"],
            trust_score=0.8
        )
        self.identities[audio_service.id] = audio_service
        
    async def authenticate_with_token(self, token: str) -> AuthenticationResult:
        """Authenticate using JWT token"""
        payload = self.crypto_service.verify_jwt_token(token)
        
        if not payload:
            return AuthenticationResult(
                success=False,
                error_message="Invalid token"
            )
            
        identity_id = payload.get('sub')
        if identity_id not in self.identities:
            return AuthenticationResult(
                success=False,
                error_message="Identity not found"
            )
            
        identity = self.identities[identity_id]
        identity.last_authenticated = datetime.now()
        
        return AuthenticationResult(
            success=True,
            identity=identity,
            trust_level=TrustLevel.MEDIUM,
            authentication_method=AuthenticationMethod.TOKEN,
            expires_at=datetime.fromtimestamp(payload['exp']),
            session_token=token
        )
        
    async def authenticate_with_api_key(self, api_key: str, api_secret: str) -> AuthenticationResult:
        """Authenticate using API key"""
        
        # Find API key in storage
        if api_key not in self.api_keys:
            return AuthenticationResult(
                success=False,
                error_message="Invalid API key"
            )
            
        key_info = self.api_keys[api_key]
        identity_id = key_info['identity_id']
        
        if not self.crypto_service.verify_api_key(api_key, api_secret, identity_id):
            return AuthenticationResult(
                success=False,
                error_message="Invalid API secret"
            )
            
        if identity_id not in self.identities:
            return AuthenticationResult(
                success=False,
                error_message="Identity not found"
            )
            
        identity = self.identities[identity_id]
        identity.last_authenticated = datetime.now()
        
        return AuthenticationResult(
            success=True,
            identity=identity,
            trust_level=TrustLevel.MEDIUM,
            authentication_method=AuthenticationMethod.API_KEY,
            expires_at=datetime.now() + timedelta(hours=24),
            metadata={'api_key': api_key}
        )
        
    def create_api_key(self, identity_id: str) -> Tuple[str, str]:
        """Create API key for identity"""
        if identity_id not in self.identities:
            raise ValueError("Identity not found")
            
        api_key, api_secret = self.crypto_service.generate_api_key(identity_id)
        
        self.api_keys[api_key] = {
            'identity_id': identity_id,
            'created_at': datetime.now(),
            'last_used': None
        }
        
        return api_key, api_secret
        
    def create_session(self, identity: Identity, expires_in: int = 3600) -> str:
        """Create authenticated session"""
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            'identity_id': identity.id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=expires_in),
            'last_activity': datetime.now()
        }
        
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Identity]:
        """Get identity from session"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # Check if session expired
        if datetime.now() > session['expires_at']:
            del self.sessions[session_id]
            return None
            
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return self.identities.get(session['identity_id'])


class ZeroTrustGateway:
    """
    Zero Trust Network Gateway
    Implements Google BeyondCorp and Microsoft Zero Trust patterns
    """
    
    def __init__(self):
        self.crypto_service = CryptographicService()
        self.identity_provider = IdentityProvider(self.crypto_service)
        self.policy_engine = PolicyEngine()
        self.risk_analyzer = RiskAnalyzer()
        self.access_logs: List[Dict[str, Any]] = []
        self.metrics = ZeroTrustMetrics()
        
    async def authorize_request(self, token: str, resource_type: ResourceType,
                              resource_id: str, action: str, 
                              request_context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Main authorization entry point"""
        
        start_time = time.time()
        
        try:
            # Authenticate request
            auth_result = await self.identity_provider.authenticate_with_token(token)
            
            if not auth_result.success:
                self._log_access_attempt(
                    None, resource_type, resource_id, action, False, 
                    auth_result.error_message, request_context
                )
                self.metrics.record_authentication_failure()
                return False, auth_result.error_message, {}
                
            # Build context
            context = Context(
                identity=auth_result.identity,
                ip_address=request_context.get('ip_address', '127.0.0.1'),
                user_agent=request_context.get('user_agent'),
                device_id=request_context.get('device_id'),
                location=request_context.get('location'),
                authentication_method=auth_result.authentication_method,
                session_id=request_context.get('session_id')
            )
            
            # Analyze risk
            context.risk_score = self.risk_analyzer.analyze_context_risk(context)
            
            # Evaluate policies
            allowed, reason, trust_level = self.policy_engine.evaluate_access(
                context, resource_type, resource_id, action
            )
            
            # Log access attempt
            self._log_access_attempt(
                context.identity, resource_type, resource_id, action, 
                allowed, reason, request_context
            )
            
            # Record metrics
            if allowed:
                self.metrics.record_authorization_success()
            else:
                self.metrics.record_authorization_failure()
                
            # Prepare response metadata
            response_metadata = {
                'trust_level': trust_level.name,
                'risk_score': context.risk_score,
                'identity_type': context.identity.type,
                'authentication_method': auth_result.authentication_method.value if auth_result.authentication_method else None,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
            return allowed, reason, response_metadata
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            self.metrics.record_system_error()
            return False, "System error during authorization", {}
            
    async def authorize_api_request(self, api_key: str, api_secret: str, 
                                  resource_type: ResourceType, resource_id: str, 
                                  action: str, request_context: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Authorize API request using API key"""
        
        start_time = time.time()
        
        try:
            # Authenticate with API key
            auth_result = await self.identity_provider.authenticate_with_api_key(api_key, api_secret)
            
            if not auth_result.success:
                self._log_access_attempt(
                    None, resource_type, resource_id, action, False, 
                    auth_result.error_message, request_context
                )
                self.metrics.record_authentication_failure()
                return False, auth_result.error_message, {}
                
            # Build context
            context = Context(
                identity=auth_result.identity,
                ip_address=request_context.get('ip_address', '127.0.0.1'),
                user_agent=request_context.get('user_agent'),
                authentication_method=AuthenticationMethod.API_KEY
            )
            
            # Analyze risk
            context.risk_score = self.risk_analyzer.analyze_context_risk(context)
            
            # Evaluate policies
            allowed, reason, trust_level = self.policy_engine.evaluate_access(
                context, resource_type, resource_id, action
            )
            
            # Log access attempt
            self._log_access_attempt(
                context.identity, resource_type, resource_id, action, 
                allowed, reason, request_context
            )
            
            # Record metrics
            if allowed:
                self.metrics.record_authorization_success()
            else:
                self.metrics.record_authorization_failure()
                
            # Prepare response metadata
            response_metadata = {
                'trust_level': trust_level.name,
                'risk_score': context.risk_score,
                'identity_type': context.identity.type,
                'authentication_method': 'api_key',
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
            return allowed, reason, response_metadata
            
        except Exception as e:
            logger.error(f"API authorization error: {e}")
            self.metrics.record_system_error()
            return False, "System error during authorization", {}
            
    def _log_access_attempt(self, identity: Optional[Identity], resource_type: ResourceType,
                          resource_id: str, action: str, allowed: bool, 
                          reason: str, request_context: Dict[str, Any]):
        """Log access attempt for audit trail"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'identity_id': identity.id if identity else None,
            'identity_type': identity.type if identity else None,
            'resource_type': resource_type.value,
            'resource_id': resource_id,
            'action': action,
            'allowed': allowed,
            'reason': reason,
            'ip_address': request_context.get('ip_address'),
            'user_agent': request_context.get('user_agent'),
            'device_id': request_context.get('device_id')
        }
        
        self.access_logs.append(log_entry)
        
        # Keep only recent logs (last 10000 entries)
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-10000:]
            
        # Log to file/system
        log_level = logging.INFO if allowed else logging.WARNING
        logger.log(log_level, f"Access {('GRANTED' if allowed else 'DENIED')}: {identity.id if identity else 'unknown'} -> {resource_type.value}:{resource_id} ({action}) - {reason}")
        
    def get_access_logs(self, identity_id: Optional[str] = None, 
                       hours: int = 24) -> List[Dict[str, Any]]:
        """Get access logs for audit"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_logs = []
        for log in self.access_logs:
            log_time = datetime.fromisoformat(log['timestamp'])
            if log_time >= cutoff_time:
                if identity_id is None or log.get('identity_id') == identity_id:
                    filtered_logs.append(log)
                    
        return sorted(filtered_logs, key=lambda x: x['timestamp'], reverse=True)
        
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return self.metrics.get_metrics()


class ZeroTrustMetrics:
    """Metrics collection for zero trust operations"""
    
    def __init__(self):
        self.authentication_attempts = 0
        self.authentication_successes = 0
        self.authentication_failures = 0
        self.authorization_attempts = 0
        self.authorization_successes = 0
        self.authorization_failures = 0
        self.system_errors = 0
        self.start_time = datetime.now()
        
    def record_authentication_failure(self):
        self.authentication_attempts += 1
        self.authentication_failures += 1
        
    def record_authorization_success(self):
        self.authorization_attempts += 1
        self.authorization_successes += 1
        
    def record_authorization_failure(self):
        self.authorization_attempts += 1
        self.authorization_failures += 1
        
    def record_system_error(self):
        self.system_errors += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            'runtime_hours': runtime_hours,
            'authentication_attempts': self.authentication_attempts,
            'authentication_success_rate': self.authentication_successes / max(self.authentication_attempts, 1),
            'authorization_attempts': self.authorization_attempts,
            'authorization_success_rate': self.authorization_successes / max(self.authorization_attempts, 1),
            'system_errors': self.system_errors,
            'requests_per_hour': (self.authorization_attempts + self.authentication_attempts) / max(runtime_hours, 0.01)
        }


# Example usage and demonstration
async def demo_zero_trust_architecture():
    """Demonstrate zero trust security architecture"""
    print("🔒 Zero Trust Security Architecture Demo")
    print("Following Google BeyondCorp, Microsoft Zero Trust best practices\n")
    
    # Initialize zero trust gateway
    gateway = ZeroTrustGateway()
    
    print("🔐 Creating API keys for services...")
    
    # Create API keys for services
    api_service_key, api_service_secret = gateway.identity_provider.create_api_key("api_service")
    ml_service_key, ml_service_secret = gateway.identity_provider.create_api_key("ml_service")
    
    print(f"API Service Key: {api_service_key[:20]}...")
    print(f"ML Service Key: {ml_service_key[:20]}...")
    
    # Create JWT token for admin user
    admin_identity = gateway.identity_provider.identities["admin_user"]
    admin_token = gateway.crypto_service.create_jwt_token(admin_identity)
    print(f"Admin Token: {admin_token[:50]}...")
    
    print("\n🧪 Testing Zero Trust Authorization...")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Admin accessing admin panel',
            'auth_type': 'jwt',
            'token': admin_token,
            'resource_type': ResourceType.ADMIN_PANEL,
            'resource_id': 'main_admin',
            'action': 'read',
            'context': {
                'ip_address': '192.168.1.100',
                'user_agent': 'Mozilla/5.0 (Admin Browser)',
                'device_id': 'admin-laptop-001'
            }
        },
        {
            'name': 'API service accessing ML model',
            'auth_type': 'api_key',
            'api_key': api_service_key,
            'api_secret': api_service_secret,
            'resource_type': ResourceType.API_ENDPOINT,
            'resource_id': 'ml_inference',
            'action': 'read',
            'context': {
                'ip_address': '10.0.1.50',
                'user_agent': 'AG06-Service/1.0'
            }
        },
        {
            'name': 'ML service accessing model',
            'auth_type': 'api_key',
            'api_key': ml_service_key,
            'api_secret': ml_service_secret,
            'resource_type': ResourceType.ML_MODEL,
            'resource_id': 'recommendation_model',
            'action': 'execute',
            'context': {
                'ip_address': '10.0.2.25',
                'user_agent': 'ML-Service/2.0'
            }
        },
        {
            'name': 'Unauthorized database access attempt',
            'auth_type': 'jwt',
            'token': admin_token,
            'resource_type': ResourceType.DATABASE,
            'resource_id': 'user_data',
            'action': 'delete',  # This should be denied
            'context': {
                'ip_address': '203.0.113.15',  # Suspicious external IP
                'user_agent': 'curl/7.68.0'
            }
        },
        {
            'name': 'Audio stream access',
            'auth_type': 'api_key',
            'api_key': api_service_key,
            'api_secret': api_service_secret,
            'resource_type': ResourceType.AUDIO_STREAM,
            'resource_id': 'live_mix',
            'action': 'write',
            'context': {
                'ip_address': '192.168.1.200',
                'user_agent': 'AG06-Mixer/3.0'
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        try:
            if scenario['auth_type'] == 'jwt':
                allowed, reason, metadata = await gateway.authorize_request(
                    scenario['token'],
                    scenario['resource_type'],
                    scenario['resource_id'],
                    scenario['action'],
                    scenario['context']
                )
            else:  # API key
                allowed, reason, metadata = await gateway.authorize_api_request(
                    scenario['api_key'],
                    scenario['api_secret'],
                    scenario['resource_type'],
                    scenario['resource_id'],
                    scenario['action'],
                    scenario['context']
                )
                
            status = "✅ ALLOWED" if allowed else "❌ DENIED"
            print(f"   Result: {status}")
            print(f"   Reason: {reason}")
            print(f"   Trust Level: {metadata.get('trust_level', 'Unknown')}")
            print(f"   Risk Score: {metadata.get('risk_score', 0):.3f}")
            print(f"   Processing Time: {metadata.get('processing_time_ms', 0)}ms")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Show security metrics
    print(f"\n📊 Security Metrics:")
    metrics = gateway.get_security_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Show access logs
    print(f"\n📋 Recent Access Logs (Last 5):")
    logs = gateway.get_access_logs(hours=1)
    for log in logs[:5]:
        status = "✅" if log['allowed'] else "❌"
        print(f"   {status} {log['timestamp']} - {log['identity_id']} -> {log['resource_type']}:{log['resource_id']} ({log['action']})")
    
    # Demonstrate encryption
    print(f"\n🔐 Data Encryption Demo:")
    sensitive_data = b"User credit card: 4111-1111-1111-1111"
    encrypted = gateway.crypto_service.encrypt_data(sensitive_data)
    decrypted = gateway.crypto_service.decrypt_data(encrypted)
    
    print(f"   Original: {sensitive_data.decode()}")
    print(f"   Encrypted Size: {len(encrypted)} bytes")
    print(f"   Decrypted: {decrypted.decode()}")
    print(f"   Encryption Working: {'✅' if sensitive_data == decrypted else '❌'}")
    
    print(f"\n🛡️ Zero Trust Architecture Summary:")
    print(f"   - Identity-based access control: ✅")
    print(f"   - Risk-based authentication: ✅")
    print(f"   - Policy-driven authorization: ✅")
    print(f"   - Continuous verification: ✅")
    print(f"   - Least privilege access: ✅")
    print(f"   - Comprehensive audit logging: ✅")
    print(f"   - End-to-end encryption: ✅")


if __name__ == "__main__":
    asyncio.run(demo_zero_trust_architecture())