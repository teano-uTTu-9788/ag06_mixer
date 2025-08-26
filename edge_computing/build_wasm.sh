#!/bin/bash
# Build script for AI Mixer WebAssembly module
# Requires Emscripten SDK (emsdk) to be installed and activated

set -e

echo "🔧 Building AI Mixer WebAssembly Module"
echo "========================================"

# Check if Emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "❌ Error: Emscripten (emcc) not found"
    echo "Please install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

# Navigate to WASM directory
cd "$(dirname "$0")/wasm"

echo "🎯 Compiling C++ to WebAssembly..."

# Compile with Emscripten
emcc ai_mixer_wasm.cpp \
    -o ai_mixer_wasm.js \
    -s WASM=1 \
    -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="'AIMixerWASMModule'" \
    -s USE_ES6_IMPORT_META=0 \
    -s ENVIRONMENT=web,worker \
    -s FILESYSTEM=0 \
    -O3 \
    --bind \
    -std=c++17

if [ $? -eq 0 ]; then
    echo "✅ WebAssembly compilation successful"
    echo "Generated files:"
    ls -la ai_mixer_wasm.*
else
    echo "❌ WebAssembly compilation failed"
    exit 1
fi

# Verify output files
if [ -f "ai_mixer_wasm.wasm" ] && [ -f "ai_mixer_wasm.js" ]; then
    echo "✅ All required files generated"
    
    # Show file sizes
    echo "📊 File sizes:"
    du -h ai_mixer_wasm.wasm ai_mixer_wasm.js
    
    # Test WASM module loading
    echo "🧪 Testing WASM module loading..."
    node -e "
        const Module = require('./ai_mixer_wasm.js');
        Module().then(module => {
            console.log('✅ WASM module loads successfully');
            console.log('Exported functions:', Object.keys(module).filter(k => !k.startsWith('_')));
        }).catch(err => {
            console.error('❌ WASM module loading failed:', err);
            process.exit(1);
        });
    "
else
    echo "❌ Required output files missing"
    exit 1
fi

echo "🎉 Build completed successfully!"
echo "Next steps:"
echo "  1. Deploy to Cloudflare Workers: wrangler publish"
echo "  2. Test in browser environment"
echo "  3. Deploy to CDN for global distribution"