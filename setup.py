#!/usr/bin/env python3
"""
AG06 SDK Setup Configuration
Enterprise Python package setup for AG06 workflow system integration
"""

from setuptools import setup, find_packages
import os

# Read README file
current_dir = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = """
AG06 SDK - Enterprise Workflow System Integration

A comprehensive Python SDK for integrating with the AG06 production workflow system.
Provides ML analytics, real-time monitoring, workflow orchestration, and enterprise integrations.
    """.strip()

# Read version from module
version = "1.0.0"
try:
    exec(open('ag06_sdk.py').read())
    version = locals().get('__version__', '1.0.0')
except:
    pass

setup(
    name="ag06-sdk",
    version=version,
    author="AG06 Team",
    author_email="support@ag06.com",
    description="Enterprise SDK for AG06 workflow system integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ag06/ag06-sdk",
    project_urls={
        "Documentation": "https://docs.ag06.com/sdk",
        "Source": "https://github.com/ag06/ag06-sdk",
        "Bug Reports": "https://github.com/ag06/ag06-sdk/issues",
    },
    packages=find_packages(),
    py_modules=["ag06_sdk", "integration_examples"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "asyncio-compat>=0.1.0",
    ],
    extras_require={
        "full": [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "flask>=2.0.0",
            "flask-restful>=0.3.9",
            "flask-socketio>=5.0.0",
            "requests>=2.25.0",
        ],
        "ml": [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scikit-learn>=1.0.0",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-restful>=0.3.9",
            "flask-socketio>=5.0.0",
            "requests>=2.25.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-asyncio>=0.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ag06-sdk=ag06_sdk:sdk_demo",
            "ag06-examples=integration_examples:run_integration_examples",
            "ag06-health=ag06_sdk:quick_health_check",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "ag06",
        "workflow",
        "automation",
        "ml",
        "analytics",
        "monitoring",
        "enterprise",
        "integration",
        "api",
        "sdk",
        "microservices",
        "orchestration"
    ],
)