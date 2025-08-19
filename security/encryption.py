"""
TLS/SSL Encryption Layer for Production
MANU Compliance: Security Requirements
"""
import ssl
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
from typing import Optional, Tuple
import base64


class EncryptionManager:
    """
    Manages encryption for AG-06 mixer system
    Provides TLS/SSL configuration and data encryption
    """
    
    def __init__(self):
        """Initialize encryption manager"""
        self.ssl_context = None
        self.data_cipher = None
        self.key_derivation_salt = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption components"""
        # Generate salt for key derivation
        self.key_derivation_salt = secrets.token_bytes(32)
        
        # Create SSL context for TLS
        self.ssl_context = self.create_ssl_context()
        
        # Initialize data encryption cipher
        master_key = self.generate_master_key()
        self.data_cipher = Fernet(master_key)
    
    def create_ssl_context(self, 
                          cert_file: Optional[str] = None,
                          key_file: Optional[str] = None) -> ssl.SSLContext:
        """
        Create SSL context for TLS encryption
        
        Args:
            cert_file: Path to certificate file
            key_file: Path to private key file
            
        Returns:
            Configured SSL context
        """
        # Create SSL context with secure defaults
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Set minimum TLS version to 1.2
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Disable weak ciphers
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        # Load certificate and key if provided
        if cert_file and key_file:
            context.load_cert_chain(cert_file, key_file)
        else:
            # For development/testing, create self-signed cert
            # In production, use proper certificates
            self._create_self_signed_cert(context)
        
        # Enable hostname checking
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        return context
    
    def _create_self_signed_cert(self, context: ssl.SSLContext):
        """Create self-signed certificate for development"""
        # In production, this should load real certificates
        # This is a placeholder for development
        pass
    
    def generate_master_key(self, password: Optional[str] = None) -> bytes:
        """
        Generate master encryption key
        
        Args:
            password: Optional password for key derivation
            
        Returns:
            Base64-encoded encryption key
        """
        if password:
            # Derive key from password
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.key_derivation_salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(password.encode())
        else:
            # Generate random key
            key = secrets.token_bytes(32)
        
        return base64.urlsafe_b64encode(key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.data_cipher:
            raise RuntimeError("Encryption not initialized")
        
        return self.data_cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.data_cipher:
            raise RuntimeError("Encryption not initialized")
        
        return self.data_cipher.decrypt(encrypted_data)
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """
        Hash password using PBKDF2
        
        Args:
            password: Password to hash
            
        Returns:
            Tuple of (salt, hash)
        """
        salt = secrets.token_hex(32)
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return salt, pwdhash.hex()
    
    def verify_password(self, password: str, salt: str, pwdhash: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Password to verify
            salt: Salt used for hashing
            pwdhash: Password hash
            
        Returns:
            True if password matches
        """
        test_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return test_hash.hex() == pwdhash
    
    def get_tls_config(self) -> dict:
        """
        Get TLS configuration for production
        
        Returns:
            TLS configuration dictionary
        """
        return {
            'enabled': True,
            'min_version': 'TLSv1.2',
            'ciphers': 'ECDHE+AESGCM:ECDHE+CHACHA20',
            'verify_mode': 'CERT_REQUIRED',
            'check_hostname': True,
            'session_tickets': False,  # Disable for perfect forward secrecy
            'compression': False  # Disable to prevent CRIME attack
        }


class SecretsVault:
    """
    Secrets management vault
    Provides secure storage and retrieval of secrets
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        """
        Initialize secrets vault
        
        Args:
            encryption_manager: Encryption manager for secret encryption
        """
        self.encryption_manager = encryption_manager
        self._secrets = {}  # In production, use external vault service
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from secure storage"""
        # In production, integrate with:
        # - HashiCorp Vault
        # - AWS Secrets Manager
        # - Azure Key Vault
        # - Google Secret Manager
        
        # For now, using environment variables as placeholder
        import os
        
        # Load from environment
        self._secrets = {
            'database_url': os.getenv('DATABASE_URL', ''),
            'api_key': os.getenv('API_KEY', ''),
            'jwt_secret': os.getenv('JWT_SECRET', secrets.token_urlsafe(32)),
            'encryption_key': os.getenv('ENCRYPTION_KEY', '')
        }
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Retrieve secret by key
        
        Args:
            key: Secret key
            
        Returns:
            Secret value or None
        """
        encrypted_secret = self._secrets.get(key)
        
        if encrypted_secret:
            # In production, secrets would be encrypted at rest
            return encrypted_secret
        
        return None
    
    def set_secret(self, key: str, value: str) -> bool:
        """
        Store secret
        
        Args:
            key: Secret key
            value: Secret value
            
        Returns:
            True if stored successfully
        """
        try:
            # In production, encrypt before storing
            encrypted_value = self.encryption_manager.encrypt_data(value.encode())
            self._secrets[key] = encrypted_value.decode('utf-8')
            return True
        except Exception as e:
            print(f"Failed to store secret: {e}")
            return False
    
    def rotate_secret(self, key: str) -> str:
        """
        Rotate a secret
        
        Args:
            key: Secret key to rotate
            
        Returns:
            New secret value
        """
        # Generate new secret
        new_secret = secrets.token_urlsafe(32)
        
        # Store new secret
        self.set_secret(key, new_secret)
        
        # In production, would also:
        # - Notify dependent services
        # - Maintain old secret for grace period
        # - Log rotation event
        
        return new_secret
    
    def delete_secret(self, key: str) -> bool:
        """
        Delete a secret
        
        Args:
            key: Secret key
            
        Returns:
            True if deleted successfully
        """
        if key in self._secrets:
            del self._secrets[key]
            return True
        return False


# Export encryption components
__all__ = ['EncryptionManager', 'SecretsVault']