#!/usr/bin/env python3
"""
Security Hardening 2025
Following Google/Amazon/Microsoft security best practices
"""

import os
import hashlib
import hmac
import secrets
import time
import jwt
import re
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import ipaddress
from functools import wraps
import asyncio

# Google: Zero Trust Security Model
@dataclass
class SecurityContext:
    """Security context for requests following Zero Trust principles"""
    user_id: str
    client_ip: str
    user_agent: str
    timestamp: datetime
    trust_score: float
    risk_level: str
    authenticated: bool = False
    authorized: bool = False
    device_fingerprint: Optional[str] = None
    location_risk: Optional[str] = None

class ZeroTrustValidator:
    """Google Zero Trust security validation"""
    
    def __init__(self):
        self.trust_thresholds = {
            "high_risk": 0.3,
            "medium_risk": 0.6,
            "low_risk": 0.8,
            "trusted": 0.95
        }
        
        # Known bad IP ranges (in production, use threat intel feeds)
        self.blocked_networks = [
            ipaddress.IPv4Network("10.0.0.0/8"),    # Private networks
            ipaddress.IPv4Network("172.16.0.0/12"),
            ipaddress.IPv4Network("192.168.0.0/16"),
        ]
        
        self.suspicious_user_agents = [
            r".*bot.*",
            r".*crawler.*",
            r".*spider.*",
            r".*scanner.*"
        ]
    
    def calculate_trust_score(self, context: SecurityContext) -> float:
        """Calculate trust score based on multiple factors"""
        score = 1.0
        
        # IP reputation check
        try:
            ip = ipaddress.IPv4Address(context.client_ip)
            for network in self.blocked_networks:
                if ip in network:
                    score -= 0.3
                    context.location_risk = "high"
                    break
        except ipaddress.AddressValueError:
            score -= 0.2  # Invalid IP format
        
        # User agent analysis
        for pattern in self.suspicious_user_agents:
            if re.match(pattern, context.user_agent.lower()):
                score -= 0.4
                break
        
        # Time-based analysis (unusual hours)
        hour = context.timestamp.hour
        if hour < 6 or hour > 22:  # Outside business hours
            score -= 0.1
        
        # Rate limiting history (simplified)
        # In production, check against rate limiting database
        score = max(0.0, min(1.0, score))
        
        return score
    
    def validate_request(self, context: SecurityContext) -> Dict[str, any]:
        """Validate request using Zero Trust principles"""
        trust_score = self.calculate_trust_score(context)
        context.trust_score = trust_score
        
        # Determine risk level
        if trust_score >= self.trust_thresholds["trusted"]:
            context.risk_level = "low"
            allow_request = True
        elif trust_score >= self.trust_thresholds["low_risk"]:
            context.risk_level = "medium"
            allow_request = True
        elif trust_score >= self.trust_thresholds["medium_risk"]:
            context.risk_level = "high"
            allow_request = False  # Require additional verification
        else:
            context.risk_level = "critical"
            allow_request = False
        
        return {
            "allow_request": allow_request,
            "trust_score": trust_score,
            "risk_level": context.risk_level,
            "required_actions": self._get_required_actions(context)
        }
    
    def _get_required_actions(self, context: SecurityContext) -> List[str]:
        """Get required security actions based on risk level"""
        actions = []
        
        if context.risk_level == "critical":
            actions.extend(["block_request", "log_security_event", "notify_security_team"])
        elif context.risk_level == "high":
            actions.extend(["require_mfa", "additional_verification", "enhanced_logging"])
        elif context.risk_level == "medium":
            actions.extend(["enhanced_monitoring", "log_event"])
        
        return actions

# Microsoft: Advanced Threat Protection patterns
class CodeSanitizer:
    """Microsoft-style code sanitization and threat detection"""
    
    def __init__(self):
        # OWASP Top 10 patterns
        self.dangerous_patterns = {
            "code_injection": [
                r"exec\s*\(",
                r"eval\s*\(",
                r"__import__\s*\(",
                r"compile\s*\(",
                r"subprocess\.",
                r"os\.system",
                r"os\.popen",
                r"os\.spawn"
            ],
            "file_access": [
                r"open\s*\(",
                r"file\s*\(",
                r"with\s+open",
                r"io\.open",
                r"pathlib\.",
                r"glob\.",
                r"os\.walk",
                r"os\.listdir"
            ],
            "network_access": [
                r"urllib\.",
                r"requests\.",
                r"http\.",
                r"socket\.",
                r"ftplib\.",
                r"smtplib\.",
                r"telnetlib\.",
                r"xmlrpc\."
            ],
            "privilege_escalation": [
                r"os\.setuid",
                r"os\.setgid",
                r"os\.chmod",
                r"subprocess\.call.*shell=True",
                r"__builtins__",
                r"globals\(\)",
                r"locals\(\)",
                r"vars\(\)"
            ]
        }
        
        self.obfuscation_patterns = [
            r"\\x[0-9a-fA-F]{2}",  # Hex encoding
            r"\\u[0-9a-fA-F]{4}",  # Unicode encoding
            r"\\[0-7]{3}",         # Octal encoding
            r"chr\(\d+\)",         # Character codes
            r"base64\.",           # Base64 encoding
            r"codecs\.",           # Codec manipulation
        ]
    
    def analyze_code_security(self, code: str, language: str = "python") -> Dict[str, any]:
        """Comprehensive code security analysis"""
        threats = []
        risk_score = 0.0
        
        code_lower = code.lower()
        
        # Check for dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                if matches:
                    severity = self._get_pattern_severity(category)
                    threats.append({
                        "type": category,
                        "pattern": pattern,
                        "matches": len(matches),
                        "severity": severity,
                        "first_match": matches[0] if matches else None
                    })
                    risk_score += severity * len(matches)
        
        # Check for obfuscation attempts
        obfuscation_score = 0
        for pattern in self.obfuscation_patterns:
            matches = re.findall(pattern, code)
            if matches:
                obfuscation_score += len(matches)
                threats.append({
                    "type": "obfuscation",
                    "pattern": pattern,
                    "matches": len(matches),
                    "severity": 0.3,
                    "description": "Potential code obfuscation detected"
                })
        
        # Language-specific checks
        if language == "python":
            risk_score += self._check_python_specific(code)
        elif language in ["javascript", "typescript"]:
            risk_score += self._check_javascript_specific(code)
        
        # Calculate overall risk level
        normalized_risk = min(1.0, risk_score / 10.0)  # Normalize to 0-1
        
        if normalized_risk > 0.8:
            risk_level = "critical"
            recommendation = "block_execution"
        elif normalized_risk > 0.6:
            risk_level = "high"
            recommendation = "require_approval"
        elif normalized_risk > 0.3:
            risk_level = "medium"
            recommendation = "enhanced_monitoring"
        else:
            risk_level = "low"
            recommendation = "allow_with_logging"
        
        return {
            "risk_score": normalized_risk,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "threats_detected": len(threats),
            "threats": threats,
            "obfuscation_score": obfuscation_score,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_pattern_severity(self, category: str) -> float:
        """Get severity score for threat category"""
        severity_map = {
            "code_injection": 1.0,
            "privilege_escalation": 0.9,
            "network_access": 0.7,
            "file_access": 0.6
        }
        return severity_map.get(category, 0.5)
    
    def _check_python_specific(self, code: str) -> float:
        """Python-specific security checks"""
        risk = 0.0
        
        # Check for dangerous imports
        dangerous_imports = ["subprocess", "os", "sys", "ctypes", "importlib"]
        for imp in dangerous_imports:
            if f"import {imp}" in code or f"from {imp}" in code:
                risk += 0.3
        
        # Check for eval/exec usage
        if "eval(" in code or "exec(" in code:
            risk += 0.5
        
        return risk
    
    def _check_javascript_specific(self, code: str) -> float:
        """JavaScript-specific security checks"""
        risk = 0.0
        
        # Check for dangerous functions
        dangerous_js = ["eval(", "Function(", "setTimeout(", "setInterval("]
        for func in dangerous_js:
            if func in code:
                risk += 0.4
        
        # Check for DOM manipulation that could be XSS
        dom_patterns = ["innerHTML", "outerHTML", "document.write", "eval"]
        for pattern in dom_patterns:
            if pattern in code:
                risk += 0.3
        
        return risk

# Amazon: Encryption and Key Management
class EnterpriseKeyManager:
    """AWS KMS-inspired key management"""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.key_rotation_interval = timedelta(days=30)
        self.keys: Dict[str, Dict] = {}
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        return Fernet.generate_key()
    
    def create_data_key(self, context: str = "default") -> Dict[str, any]:
        """Create encrypted data encryption key"""
        # Generate data encryption key
        dek = Fernet.generate_key()
        
        # Encrypt DEK with master key
        f = Fernet(self.master_key)
        encrypted_dek = f.encrypt(dek)
        
        key_id = secrets.token_hex(16)
        
        key_metadata = {
            "key_id": key_id,
            "encrypted_dek": encrypted_dek,
            "plaintext_dek": dek,  # In production, don't store plaintext
            "created_at": datetime.utcnow(),
            "context": context,
            "usage_count": 0
        }
        
        self.keys[key_id] = key_metadata
        
        return {
            "key_id": key_id,
            "encrypted_dek": encrypted_dek.decode(),
            "plaintext_dek": dek.decode()
        }
    
    def encrypt_data(self, data: str, key_id: str) -> Dict[str, any]:
        """Encrypt data using specified key"""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key_meta = self.keys[key_id]
        f = Fernet(key_meta["plaintext_dek"])
        encrypted_data = f.encrypt(data.encode())
        
        key_meta["usage_count"] += 1
        
        return {
            "encrypted_data": encrypted_data.decode(),
            "key_id": key_id,
            "algorithm": "AES-256-GCM",
            "encrypted_at": datetime.utcnow().isoformat()
        }
    
    def decrypt_data(self, encrypted_data: str, key_id: str) -> str:
        """Decrypt data using specified key"""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key_meta = self.keys[key_id]
        f = Fernet(key_meta["plaintext_dek"])
        decrypted_data = f.decrypt(encrypted_data.encode())
        
        return decrypted_data.decode()

# JWT Token Management with security best practices
class SecureJWTManager:
    """Enterprise JWT token management"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=1)
        self.refresh_expiry = timedelta(days=7)
    
    def create_token(self, user_id: str, permissions: List[str]) -> Dict[str, str]:
        """Create JWT access and refresh tokens"""
        now = datetime.utcnow()
        
        # Access token
        access_payload = {
            "user_id": user_id,
            "permissions": permissions,
            "type": "access",
            "iat": now,
            "exp": now + self.token_expiry,
            "jti": secrets.token_hex(16)  # JWT ID for blacklisting
        }
        
        # Refresh token
        refresh_payload = {
            "user_id": user_id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_expiry,
            "jti": secrets.token_hex(16)
        }
        
        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": int(self.token_expiry.total_seconds()),
            "token_type": "Bearer"
        }
    
    def verify_token(self, token: str) -> Dict[str, any]:
        """Verify JWT token with comprehensive validation"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_signature": True
                }
            )
            
            # Additional security checks
            if payload.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            
            # Check if token is not too old (even if not expired)
            issued_at = datetime.fromtimestamp(payload["iat"])
            if datetime.utcnow() - issued_at > timedelta(hours=24):
                raise jwt.InvalidTokenError("Token too old")
            
            return {
                "valid": True,
                "payload": payload,
                "user_id": payload["user_id"],
                "permissions": payload.get("permissions", [])
            }
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": str(e)}

# Security middleware integration
def security_middleware(zero_trust: ZeroTrustValidator, code_sanitizer: CodeSanitizer):
    """Security middleware factory"""
    
    def middleware_decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            # Create security context
            context = SecurityContext(
                user_id=getattr(request, 'user_id', 'anonymous'),
                client_ip=request.client.host if hasattr(request, 'client') else 'unknown',
                user_agent=request.headers.get('user-agent', 'unknown'),
                timestamp=datetime.utcnow(),
                trust_score=0.0,
                risk_level='unknown'
            )
            
            # Zero Trust validation
            validation_result = zero_trust.validate_request(context)
            
            if not validation_result["allow_request"]:
                logging.warning(
                    f"Security: Blocked request from {context.client_ip}, "
                    f"trust_score={validation_result['trust_score']}, "
                    f"risk_level={validation_result['risk_level']}"
                )
                raise Exception(f"Request blocked due to {validation_result['risk_level']} risk")
            
            # Additional code sanitization for code execution requests
            if hasattr(request, 'json') and 'code' in str(request.json):
                # This would need to be adapted based on your actual request structure
                pass
            
            # Add security context to request
            request.security_context = context
            request.security_validation = validation_result
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return middleware_decorator

# Security audit logging
class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("security_audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(self, event_type: str, details: Dict[str, any], 
                          severity: str = "info"):
        """Log security events with structured data"""
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "details": details
        }
        
        log_message = f"SECURITY_EVENT: {event_type} | {json.dumps(log_entry)}"
        
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "high":
            self.logger.error(log_message)
        elif severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

# Example usage and testing
async def main():
    """Test security components"""
    # Initialize security components
    zero_trust = ZeroTrustValidator()
    code_sanitizer = CodeSanitizer()
    key_manager = EnterpriseKeyManager()
    jwt_manager = SecureJWTManager("your-secret-key-here")
    audit_logger = SecurityAuditLogger()
    
    # Test Zero Trust validation
    context = SecurityContext(
        user_id="test_user",
        client_ip="192.168.1.100",
        user_agent="Mozilla/5.0 (suspicious bot)",
        timestamp=datetime.utcnow(),
        trust_score=0.0,
        risk_level="unknown"
    )
    
    validation = zero_trust.validate_request(context)
    print(f"Zero Trust Validation: {validation}")
    
    # Test code sanitization
    suspicious_code = """
import os
import subprocess
exec("print('malicious code')")
subprocess.call("rm -rf /", shell=True)
"""
    
    analysis = code_sanitizer.analyze_code_security(suspicious_code)
    print(f"Code Security Analysis: {analysis}")
    
    # Test encryption
    data_key = key_manager.create_data_key("test-context")
    encrypted = key_manager.encrypt_data("sensitive data", data_key["key_id"])
    decrypted = key_manager.decrypt_data(encrypted["encrypted_data"], data_key["key_id"])
    print(f"Encryption test: {decrypted}")
    
    # Test JWT
    token_data = jwt_manager.create_token("user123", ["read", "write"])
    verification = jwt_manager.verify_token(token_data["access_token"])
    print(f"JWT Verification: {verification}")
    
    # Log security event
    audit_logger.log_security_event(
        "code_execution_attempt",
        {"user": "test_user", "risk_level": "high", "action": "blocked"},
        "medium"
    )

if __name__ == "__main__":
    asyncio.run(main())