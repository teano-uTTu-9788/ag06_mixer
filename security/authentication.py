"""
Authentication and Security Layer for Production
MANU Compliance: Security Requirements
"""
import hashlib
import hmac
import secrets
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from functools import wraps


class AuthenticationManager:
    """
    Manages authentication for AG-06 mixer system
    Implements JWT-based authentication with rate limiting
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize authentication manager"""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        self.refresh_expiry = timedelta(days=7)
        self.rate_limits = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def generate_token(self, user_id: str, role: str = "user") -> Dict[str, str]:
        """
        Generate JWT access and refresh tokens
        
        Args:
            user_id: User identifier
            role: User role for RBAC
            
        Returns:
            Dict with access_token and refresh_token
        """
        # Check rate limiting
        if self._is_rate_limited(user_id):
            raise SecurityError("Too many authentication attempts. Please try again later.")
        
        now = datetime.utcnow()
        
        # Access token payload
        access_payload = {
            "user_id": user_id,
            "role": role,
            "type": "access",
            "iat": now,
            "exp": now + self.token_expiry
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_expiry
        }
        
        # Generate tokens
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": int(self.token_expiry.total_seconds())
        }
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            SecurityError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Additional validation
            if payload.get("type") != "access":
                raise SecurityError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {e}")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Generate new access token from refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token
        """
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "refresh":
                raise SecurityError("Invalid refresh token")
            
            # Generate new access token
            return self.generate_token(payload["user_id"], payload.get("role", "user"))
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid refresh token: {e}")
    
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        now = time.time()
        
        if user_id in self.rate_limits:
            attempts, last_attempt = self.rate_limits[user_id]
            
            # Reset if lockout period has passed
            if now - last_attempt > self.lockout_duration:
                self.rate_limits[user_id] = (1, now)
                return False
            
            # Check if exceeded max attempts
            if attempts >= self.max_attempts:
                return True
            
            # Increment attempts
            self.rate_limits[user_id] = (attempts + 1, now)
        else:
            self.rate_limits[user_id] = (1, now)
        
        return False


class RBACManager:
    """
    Role-Based Access Control Manager
    Manages permissions and authorization
    """
    
    def __init__(self):
        """Initialize RBAC manager"""
        self.roles = {
            "admin": {
                "permissions": ["*"],  # All permissions
                "description": "Full system access"
            },
            "engineer": {
                "permissions": [
                    "audio.read", "audio.write",
                    "preset.read", "preset.write",
                    "midi.read", "midi.write",
                    "config.read"
                ],
                "description": "Audio engineering access"
            },
            "user": {
                "permissions": [
                    "audio.read",
                    "preset.read",
                    "midi.read"
                ],
                "description": "Basic user access"
            },
            "guest": {
                "permissions": [
                    "audio.read"
                ],
                "description": "Read-only access"
            }
        }
    
    def check_permission(self, role: str, permission: str) -> bool:
        """
        Check if role has permission
        
        Args:
            role: User role
            permission: Required permission
            
        Returns:
            True if permission granted
        """
        if role not in self.roles:
            return False
        
        role_perms = self.roles[role]["permissions"]
        
        # Check for wildcard permission
        if "*" in role_perms:
            return True
        
        # Check specific permission
        return permission in role_perms
    
    def get_role_permissions(self, role: str) -> list:
        """Get all permissions for a role"""
        if role not in self.roles:
            return []
        
        return self.roles[role]["permissions"]


class SecurityMiddleware:
    """
    Security middleware for request processing
    Implements authentication and authorization checks
    """
    
    def __init__(self, auth_manager: AuthenticationManager, rbac_manager: RBACManager):
        """Initialize security middleware"""
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager
    
    def require_auth(self, required_permission: Optional[str] = None):
        """
        Decorator for requiring authentication
        
        Args:
            required_permission: Permission required for access
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token from request (simplified)
                token = kwargs.get("token")
                
                if not token:
                    raise SecurityError("Authentication required")
                
                try:
                    # Verify token
                    payload = self.auth_manager.verify_token(token)
                    
                    # Check permission if required
                    if required_permission:
                        role = payload.get("role", "guest")
                        if not self.rbac_manager.check_permission(role, required_permission):
                            raise SecurityError(f"Permission denied: {required_permission}")
                    
                    # Add user info to kwargs
                    kwargs["user_id"] = payload["user_id"]
                    kwargs["user_role"] = payload.get("role", "guest")
                    
                    # Execute function
                    return await func(*args, **kwargs)
                    
                except SecurityError:
                    raise
                except Exception as e:
                    raise SecurityError(f"Authentication failed: {e}")
            
            return wrapper
        return decorator


class InputValidator:
    """
    Input validation and sanitization
    Prevents injection attacks and validates data
    """
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # Truncate to max length
        value = value[:max_length]
        
        # Remove control characters
        value = "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")
        
        return value
    
    @staticmethod
    def validate_audio_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audio processing parameters"""
        validated = {}
        
        # Sample rate validation
        if "sample_rate" in params:
            sr = params["sample_rate"]
            if sr not in [44100, 48000, 96000, 192000]:
                raise ValueError(f"Invalid sample rate: {sr}")
            validated["sample_rate"] = sr
        
        # Buffer size validation
        if "buffer_size" in params:
            bs = params["buffer_size"]
            if not (64 <= bs <= 4096) or (bs & (bs - 1)) != 0:
                raise ValueError(f"Invalid buffer size: {bs}")
            validated["buffer_size"] = bs
        
        # Gain validation
        for gain_type in ["input_gain", "output_gain"]:
            if gain_type in params:
                gain = params[gain_type]
                if not -60 <= gain <= 12:
                    raise ValueError(f"Invalid {gain_type}: {gain}")
                validated[gain_type] = gain
        
        return validated


class SecurityError(Exception):
    """Security-related exception"""
    pass


# Export security components
__all__ = [
    'AuthenticationManager',
    'RBACManager',
    'SecurityMiddleware',
    'InputValidator',
    'SecurityError'
]