#!/usr/bin/env python3
"""
AGMixer - Health Check Script
Simple import-based health check for deployment
"""

import sys
import os
from pathlib import Path
import traceback

def health_check():
    """Run AGMixer health check"""
    results = {
        "structure": False,
        "interfaces": False, 
        "implementations": False,
        "main": False,
        "web_app": False,
        "manu_workflow": False
    }
    
    project_root = Path(__file__).parent
    print(f"🧭 AGMixer Health Check - {project_root}")
    
    # 1. Check project structure
    try:
        required_files = ["main.py", "web_app.py", "ag06_manu_workflow.py"]
        required_dirs = ["interfaces", "implementations", "core"]
        
        for file in required_files:
            if not (project_root / file).exists():
                print(f"❌ Missing file: {file}")
                return False
        
        for directory in required_dirs:
            if not (project_root / directory).exists():
                print(f"❌ Missing directory: {directory}")
                return False
        
        results["structure"] = True
        print("✅ Project structure: OK")
    except Exception as e:
        print(f"❌ Project structure check failed: {e}")
    
    # 2. Check interfaces
    try:
        sys.path.insert(0, str(project_root))
        from interfaces.audio_engine import IAudioEngine
        results["interfaces"] = True
        print("✅ Interfaces module: OK")
    except Exception as e:
        print(f"❌ Interfaces check failed: {e}")
        traceback.print_exc()
    
    # 3. Check implementations
    try:
        from implementations.audio_engine import AudioEngineImpl
        results["implementations"] = True
        print("✅ Implementations module: OK")
    except Exception as e:
        print(f"❌ Implementations check failed: {e}")
    
    # 4. Check main module (without instantiation)
    try:
        import main
        if hasattr(main, 'AG06MixerApplication'):
            results["main"] = True
            print("✅ Main module: OK")
        else:
            print("❌ Main module missing AG06MixerApplication")
    except Exception as e:
        print(f"❌ Main module check failed: {e}")
    
    # 5. Check web app
    try:
        import web_app
        if hasattr(web_app, 'AG06WebApp'):
            results["web_app"] = True
            print("✅ Web app module: OK")
        else:
            print("❌ Web app module missing AG06WebApp")
    except Exception as e:
        print(f"❌ Web app check failed: {e}")
    
    # 6. Check MANU workflow
    try:
        import ag06_manu_workflow
        if hasattr(ag06_manu_workflow, 'AG06WorkflowFactory'):
            results["manu_workflow"] = True
            print("✅ MANU workflow module: OK")
        else:
            print("❌ MANU workflow missing AG06WorkflowFactory")
    except Exception as e:
        print(f"❌ MANU workflow check failed: {e}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"\n📊 Health Check Results: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ AGMixer health check: PASSED")
        return True
    else:
        print("❌ AGMixer health check: FAILED")
        return False

if __name__ == "__main__":
    sys.exit(0 if health_check() else 1)