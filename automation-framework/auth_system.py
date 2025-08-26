#!/usr/bin/env python3
"""
Aioke Authentication System
Secure API authentication with JWT tokens and rate limiting
"""

import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from typing import Dict, Optional, List
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AuthenticationSystem:
    """
    Comprehensive authentication system for Aioke API
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = 'HS256'
        self.token_expiry_hours = 24
        
        # In-memory stores (use Redis in production)
        self.api_keys: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.blacklisted_tokens: set = set()
        
        # Rate limiting configuration
        self.rate_limit_requests = 100  # requests per window
        self.rate_limit_window = 3600   # 1 hour in seconds
        
        # Load existing API keys
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from file"""
        api_keys_file = Path('api_keys.json')
        if api_keys_file.exists():
            try:
                with open(api_keys_file, 'r') as f:
                    self.api_keys = json.load(f)
                logger.info(f"Loaded {len(self.api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
    
    def _save_api_keys(self):
        """Save API keys to file"""
        try:
            with open('api_keys.json', 'w') as f:
                json.dump(self.api_keys, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate a new API key for a user"""
        api_key = f"aioke_{secrets.token_urlsafe(32)}"
        
        key_data = {
            'user_id': user_id,
            'api_key_hash': hashlib.sha256(api_key.encode()).hexdigest(),
            'permissions': permissions or ['read', 'write'],
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0,
            'active': True
        }
        
        self.api_keys[api_key] = key_data
        self._save_api_keys()
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key and return user data"""
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        
        if not key_data.get('active', False):
            return None
        
        # Update usage statistics
        key_data['last_used'] = datetime.now().isoformat()
        key_data['usage_count'] = key_data.get('usage_count', 0) + 1
        self._save_api_keys()
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            self._save_api_keys()
            logger.info(f"API key revoked for user {self.api_keys[api_key]['user_id']}")
            return True
        return False
    
    def generate_jwt_token(self, user_data: Dict) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': user_data['user_id'],
            'permissions': user_data['permissions'],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iss': 'aioke'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload"""
        if token in self.blacklisted_tokens:
            return None
        
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def blacklist_token(self, token: str):
        """Add token to blacklist (for logout)"""
        self.blacklisted_tokens.add(token)
        
        # Clean up old blacklisted tokens (older than token expiry)
        # In production, use Redis with TTL
        if len(self.blacklisted_tokens) > 1000:
            # Keep only recent tokens
            self.blacklisted_tokens = set(list(self.blacklisted_tokens)[-500:])
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier (API key, IP, etc.) is within rate limits"""
        current_time = time.time()
        window_start = current_time - self.rate_limit_window
        
        # Get requests for this identifier in the current window
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        requests = self.rate_limits[identifier]
        
        # Remove requests outside current window
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # Check if under limit
        if len(requests) >= self.rate_limit_requests:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def get_rate_limit_status(self, identifier: str) -> Dict:
        """Get current rate limit status for identifier"""
        current_time = time.time()
        window_start = current_time - self.rate_limit_window
        
        if identifier not in self.rate_limits:
            requests_made = 0
        else:
            requests = [req for req in self.rate_limits[identifier] if req > window_start]
            requests_made = len(requests)
        
        return {
            'requests_made': requests_made,
            'requests_limit': self.rate_limit_requests,
            'window_seconds': self.rate_limit_window,
            'requests_remaining': max(0, self.rate_limit_requests - requests_made),
            'reset_time': window_start + self.rate_limit_window
        }

# Global auth system instance
auth_system = AuthenticationSystem()

def require_auth(permissions: List[str] = None):
    """
    Decorator to require authentication for API endpoints
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract authentication information
            auth_header = request.headers.get('Authorization', '')
            api_key = request.headers.get('X-API-Key', '')
            
            user_data = None
            auth_method = None
            identifier = None
            
            # Try JWT token first
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                payload = auth_system.validate_jwt_token(token)
                if payload:
                    user_data = payload
                    auth_method = 'jwt'
                    identifier = f"jwt_{payload['user_id']}"
            
            # Try API key
            elif api_key:
                key_data = auth_system.validate_api_key(api_key)
                if key_data:
                    user_data = key_data
                    auth_method = 'api_key'
                    identifier = api_key
            
            # No valid authentication
            if not user_data:
                return jsonify({
                    'error': 'Authentication required',
                    'message': 'Provide valid JWT token or API key'
                }), 401
            
            # Check rate limits
            if not auth_system.check_rate_limit(identifier):
                rate_status = auth_system.get_rate_limit_status(identifier)
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'rate_limit': rate_status
                }), 429
            
            # Check permissions
            if permissions:
                user_permissions = user_data.get('permissions', [])
                if not any(perm in user_permissions for perm in permissions):
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required': permissions,
                        'granted': user_permissions
                    }), 403
            
            # Store user data in Flask's g object
            g.current_user = user_data
            g.auth_method = auth_method
            g.rate_limit_status = auth_system.get_rate_limit_status(identifier)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_api_key(permissions: List[str] = None):
    """
    Decorator to require API key authentication only
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key', '')
            
            if not api_key:
                return jsonify({
                    'error': 'API key required',
                    'message': 'Provide X-API-Key header'
                }), 401
            
            key_data = auth_system.validate_api_key(api_key)
            if not key_data:
                return jsonify({
                    'error': 'Invalid API key'
                }), 401
            
            # Check rate limits
            if not auth_system.check_rate_limit(api_key):
                rate_status = auth_system.get_rate_limit_status(api_key)
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'rate_limit': rate_status
                }), 429
            
            # Check permissions
            if permissions:
                user_permissions = key_data.get('permissions', [])
                if not any(perm in user_permissions for perm in permissions):
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required': permissions,
                        'granted': user_permissions
                    }), 403
            
            g.current_user = key_data
            g.auth_method = 'api_key'
            g.rate_limit_status = auth_system.get_rate_limit_status(api_key)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def optional_auth(f):
    """
    Decorator for optional authentication (provides user data if available)
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Try to authenticate but don't require it
        auth_header = request.headers.get('Authorization', '')
        api_key = request.headers.get('X-API-Key', '')
        
        user_data = None
        auth_method = None
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            payload = auth_system.validate_jwt_token(token)
            if payload:
                user_data = payload
                auth_method = 'jwt'
        elif api_key:
            key_data = auth_system.validate_api_key(api_key)
            if key_data:
                user_data = key_data
                auth_method = 'api_key'
        
        g.current_user = user_data
        g.auth_method = auth_method
        g.authenticated = user_data is not None
        
        return f(*args, **kwargs)
    return decorated_function

# CLI functions for key management
def create_api_key(user_id: str, permissions: List[str] = None):
    """Create an API key via CLI"""
    api_key = auth_system.generate_api_key(user_id, permissions)
    print(f"Generated API key for {user_id}: {api_key}")
    print("Store this key securely - it won't be displayed again!")
    return api_key

def list_api_keys():
    """List all API keys (without showing the keys)"""
    print("Active API Keys:")
    print("-" * 50)
    for api_key, data in auth_system.api_keys.items():
        if data.get('active', False):
            print(f"User: {data['user_id']}")
            print(f"Permissions: {data['permissions']}")
            print(f"Created: {data['created_at']}")
            print(f"Last used: {data.get('last_used', 'Never')}")
            print(f"Usage count: {data.get('usage_count', 0)}")
            print(f"Key: {api_key[:12]}...")
            print("-" * 50)

def revoke_api_key_cli(api_key: str):
    """Revoke an API key via CLI"""
    if auth_system.revoke_api_key(api_key):
        print(f"API key revoked successfully")
    else:
        print("API key not found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python auth_system.py create <user_id> [permissions]")
        print("  python auth_system.py list")
        print("  python auth_system.py revoke <api_key>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 3:
            print("Usage: python auth_system.py create <user_id> [permissions]")
            sys.exit(1)
        
        user_id = sys.argv[2]
        permissions = sys.argv[3].split(',') if len(sys.argv) > 3 else ['read', 'write']
        create_api_key(user_id, permissions)
    
    elif command == "list":
        list_api_keys()
    
    elif command == "revoke":
        if len(sys.argv) < 3:
            print("Usage: python auth_system.py revoke <api_key>")
            sys.exit(1)
        
        api_key = sys.argv[2]
        revoke_api_key_cli(api_key)
    
    else:
        print(f"Unknown command: {command}")