#!/bin/bash

# Demo Deployment - GitHub Pages + Mock Backend
# Since Azure subscription is disabled, create a demo version

set -e

echo "🚀 DEMO DEPLOYMENT - AG06 Mixer"
echo "================================"
echo ""
echo "Azure subscription is currently disabled."
echo "Creating a demo deployment instead:"
echo "  • Frontend: GitHub Pages"
echo "  • Backend: Mock SSE simulation"
echo ""

# Create a demo version of the HTML that works without backend
cat > webapp/demo.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AG06 Cloud Mixer - Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        
        .spectrum-bar {
            transition: height 0.1s ease-out;
            background: linear-gradient(to top, #3b82f6, #8b5cf6);
        }
        
        .glass {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 min-h-screen text-white">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-8">
            <h1 class="text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
                AG06 Cloud Mixer - Demo
            </h1>
            <p class="text-gray-400">Real-time Audio Processing Simulation</p>
            <div class="mt-4">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm glass">
                    <span id="status-indicator" class="w-2 h-2 bg-green-500 rounded-full mr-2 pulse"></span>
                    <span id="status-text">Demo Mode Active</span>
                </span>
            </div>
        </header>

        <!-- Demo Banner -->
        <div class="glass rounded-xl p-4 mb-6 text-center">
            <p class="text-yellow-400 font-semibold">🚀 DEMO VERSION</p>
            <p class="text-sm text-gray-300">Azure deployment ready - subscription temporarily disabled</p>
        </div>

        <!-- Main Controls -->
        <div class="grid md:grid-cols-2 gap-6 mb-8">
            <!-- Control Panel -->
            <div class="glass rounded-xl p-6">
                <h2 class="text-2xl font-semibold mb-4">Control Panel</h2>
                
                <div class="space-y-4">
                    <button id="demo-btn" 
                            class="w-full py-3 px-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 transition-all">
                        Start Demo Simulation
                    </button>
                    
                    <div>
                        <label class="block text-sm mb-2">AI Mix Level</label>
                        <input type="range" id="ai-mix" min="0" max="100" value="70" 
                               class="w-full accent-purple-500">
                        <span id="ai-mix-value" class="text-sm text-gray-400">70%</span>
                    </div>
                    
                    <div>
                        <label class="block text-sm mb-2">Bass Boost</label>
                        <input type="range" id="bass-boost" min="0" max="200" value="130" 
                               class="w-full accent-purple-500">
                        <span id="bass-boost-value" class="text-sm text-gray-400">130%</span>
                    </div>
                </div>
            </div>

            <!-- Metrics Display -->
            <div class="glass rounded-xl p-6">
                <h2 class="text-2xl font-semibold mb-4">Live Metrics (Simulated)</h2>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm text-gray-400">Input Level</p>
                        <p class="text-2xl font-bold">
                            <span id="input-rms">-35.2</span> dB
                        </p>
                        <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
                            <div id="input-level-bar" class="bg-green-500 h-2 rounded-full" style="width: 60%"></div>
                        </div>
                    </div>
                    
                    <div>
                        <p class="text-sm text-gray-400">Output Level</p>
                        <p class="text-2xl font-bold">
                            <span id="output-rms">-32.1</span> dB
                        </p>
                        <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
                            <div id="output-level-bar" class="bg-blue-500 h-2 rounded-full" style="width: 65%"></div>
                        </div>
                    </div>
                    
                    <div>
                        <p class="text-sm text-gray-400">Genre</p>
                        <p class="text-xl font-semibold" id="genre">Electronic</p>
                    </div>
                    
                    <div>
                        <p class="text-sm text-gray-400">Latency</p>
                        <p class="text-xl font-semibold">
                            <span id="latency">8.3</span> ms
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Spectrum Visualizer -->
        <div class="glass rounded-xl p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">Spectrum Analyzer</h2>
            <div class="h-48 flex items-end justify-between" id="spectrum-container">
                <!-- Spectrum bars will be inserted here -->
            </div>
        </div>

        <!-- Deployment Status -->
        <div class="glass rounded-xl p-6">
            <h2 class="text-2xl font-semibold mb-4">Deployment Status</h2>
            <div class="space-y-3">
                <div class="flex items-center">
                    <span class="w-3 h-3 bg-green-500 rounded-full mr-3"></span>
                    <span>✅ Backend containerized and ready</span>
                </div>
                <div class="flex items-center">
                    <span class="w-3 h-3 bg-green-500 rounded-full mr-3"></span>
                    <span>✅ Azure Container Apps configuration complete</span>
                </div>
                <div class="flex items-center">
                    <span class="w-3 h-3 bg-green-500 rounded-full mr-3"></span>
                    <span>✅ GitHub Actions CI/CD with OIDC ready</span>
                </div>
                <div class="flex items-center">
                    <span class="w-3 h-3 bg-yellow-500 rounded-full mr-3"></span>
                    <span>⏳ Azure subscription pending re-activation</span>
                </div>
                <div class="flex items-center">
                    <span class="w-3 h-3 bg-gray-500 rounded-full mr-3"></span>
                    <span>⏸️ Production deployment on hold</span>
                </div>
            </div>
            
            <div class="mt-6 p-4 bg-blue-900/30 rounded-lg">
                <p class="text-blue-300 font-semibold">💡 Ready for Production</p>
                <p class="text-sm text-gray-300 mt-1">
                    All deployment scripts are ready. When Azure subscription is re-enabled, 
                    run <code class="bg-gray-800 px-2 py-1 rounded">./deploy-now.sh</code> to deploy instantly.
                </p>
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let animationId;
        
        // Initialize spectrum bars
        function initSpectrum() {
            const container = document.getElementById('spectrum-container');
            container.innerHTML = '';
            for (let i = 0; i < 16; i++) {
                const bar = document.createElement('div');
                bar.className = 'spectrum-bar rounded-t';
                bar.style.width = '5%';
                bar.style.height = '20%';
                bar.id = `spectrum-bar-${i}`;
                container.appendChild(bar);
            }
        }
        
        // Simulate spectrum animation
        function animateSpectrum() {
            if (!isRunning) return;
            
            for (let i = 0; i < 16; i++) {
                const bar = document.getElementById(`spectrum-bar-${i}`);
                if (bar) {
                    const height = Math.random() * 80 + 20;
                    bar.style.height = `${height}%`;
                }
            }
            
            // Update metrics
            const inputRms = -40 + Math.random() * 10;
            const outputRms = inputRms + 2 + Math.random() * 3;
            const latency = 5 + Math.random() * 10;
            
            document.getElementById('input-rms').textContent = inputRms.toFixed(1);
            document.getElementById('output-rms').textContent = outputRms.toFixed(1);
            document.getElementById('latency').textContent = latency.toFixed(1);
            
            const genres = ['Electronic', 'Rock', 'Pop', 'Vocal', 'Ambient'];
            document.getElementById('genre').textContent = genres[Math.floor(Math.random() * genres.length)];
            
            animationId = setTimeout(animateSpectrum, 100);
        }
        
        // Toggle demo
        document.getElementById('demo-btn').addEventListener('click', () => {
            const btn = document.getElementById('demo-btn');
            if (isRunning) {
                isRunning = false;
                clearTimeout(animationId);
                btn.textContent = 'Start Demo Simulation';
                btn.classList.remove('bg-gradient-to-r', 'from-red-500', 'to-pink-600');
                btn.classList.add('bg-gradient-to-r', 'from-blue-500', 'to-purple-600');
            } else {
                isRunning = true;
                btn.textContent = 'Stop Demo Simulation';
                btn.classList.remove('bg-gradient-to-r', 'from-blue-500', 'to-purple-600');
                btn.classList.add('bg-gradient-to-r', 'from-red-500', 'to-pink-600');
                animateSpectrum();
            }
        });
        
        // Slider event listeners
        ['ai-mix', 'bass-boost'].forEach(id => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(`${id}-value`);
            
            slider.addEventListener('input', () => {
                valueDisplay.textContent = `${slider.value}%`;
            });
        });
        
        // Initialize on load
        window.addEventListener('load', () => {
            initSpectrum();
            // Auto-start demo after 2 seconds
            setTimeout(() => {
                document.getElementById('demo-btn').click();
            }, 2000);
        });
    </script>
</body>
</html>
EOF

echo "✅ Demo HTML created at webapp/demo.html"

# Check if we're in a git repo
if [ -d ".git" ]; then
    echo "📦 Git repository detected"
    
    # Add and commit demo
    git add webapp/demo.html
    git commit -m "Add demo version for AG06 Mixer" || echo "No changes to commit"
    
    # Check if we have a GitHub remote
    if git remote | grep -q origin; then
        echo "🚀 Pushing to GitHub..."
        git push origin main || echo "Push failed - manual push may be needed"
        
        REPO_URL=$(git remote get-url origin)
        if [[ $REPO_URL == *github.com* ]]; then
            # Extract repo info
            REPO_PATH=$(echo $REPO_URL | sed 's/.*github.com[/:]//g' | sed 's/.git$//')
            
            echo ""
            echo "🌐 GitHub Pages URLs:"
            echo "  Demo: https://${REPO_PATH%/*}.github.io/${REPO_PATH#*/}/webapp/demo.html"
            echo "  Raw: https://raw.githubusercontent.com/$REPO_PATH/main/webapp/demo.html"
        fi
    else
        echo "⚠️  No GitHub remote found"
        echo "To deploy to GitHub Pages:"
        echo "  1. Create GitHub repo"
        echo "  2. git remote add origin https://github.com/USER/REPO.git"
        echo "  3. git push -u origin main"
        echo "  4. Enable GitHub Pages in repo settings"
    fi
else
    echo "⚠️  Not a git repository"
    echo "To deploy:"
    echo "  1. git init"
    echo "  2. git add ."
    echo "  3. git commit -m 'Initial commit'"
    echo "  4. Create GitHub repo and push"
fi

echo ""
echo "📊 DEPLOYMENT SUMMARY"
echo "===================="
echo ""
echo "🎯 READY COMPONENTS:"
echo "  ✅ Production backend (fixed_ai_mixer.py)"
echo "  ✅ Docker containerization" 
echo "  ✅ Azure deployment scripts"
echo "  ✅ GitHub Actions CI/CD"
echo "  ✅ Vercel configuration"
echo "  ✅ Demo version created"
echo ""
echo "📂 LOCAL FILES:"
echo "  • webapp/demo.html - Interactive demo"
echo "  • fixed_ai_mixer.py - Production backend"
echo "  • Dockerfile - Container configuration"
echo "  • deploy-now.sh - Azure deployment"
echo "  • .github/workflows/deploy-aca.yml - CI/CD"
echo ""
echo "⏳ PENDING:"
echo "  • Azure subscription re-activation"
echo "  • Production deployment"
echo ""
echo "🎉 Demo is ready to view at:"
echo "   Open: webapp/demo.html"
echo ""
echo "When Azure is ready: ./deploy-now.sh"