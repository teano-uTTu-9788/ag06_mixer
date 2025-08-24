/**
 * WebRTC Audio Streaming Client
 * Following Google's WebRTC best practices
 */

class AudioStreamingClient {
    constructor(serverUrl = 'http://localhost:8080') {
        this.serverUrl = serverUrl;
        this.socket = null;
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.isPublisher = false;
        this.isSubscriber = false;
        this.roomId = null;
        this.peerId = null;
        
        // WebRTC configuration
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ],
            iceCandidatePoolSize: 10
        };
        
        // Audio constraints for high quality
        this.audioConstraints = {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 48000,
                channelCount: 2,
                sampleSize: 16,
                latency: 0.01  // 10ms target latency
            },
            video: false
        };
        
        this.initializeEventHandlers();
    }
    
    async connect() {
        console.log('Connecting to signaling server...');
        
        // Initialize Socket.IO connection
        this.socket = io(this.serverUrl, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
        });
        
        // Setup socket event handlers
        this.setupSocketHandlers();
        
        // Initialize audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 48000,
            latencyHint: 'interactive'
        });
        
        return new Promise((resolve, reject) => {
            this.socket.on('connected', (data) => {
                console.log('Connected to server:', data);
                this.peerId = data.peerId;
                resolve(data);
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                reject(error);
            });
        });
    }
    
    setupSocketHandlers() {
        // Peer joined room
        this.socket.on('peer_joined', async (data) => {
            console.log('Peer joined:', data);
            if (this.isPublisher) {
                // Create offer for new peer
                await this.createOffer();
            }
        });
        
        // Peer left room
        this.socket.on('peer_left', (data) => {
            console.log('Peer left:', data);
            this.handlePeerDisconnect(data.sessionId);
        });
        
        // Received offer
        this.socket.on('offer', async (data) => {
            console.log('Received offer from:', data.from);
            await this.handleOffer(data);
        });
        
        // Received answer
        this.socket.on('answer', async (data) => {
            console.log('Received answer from:', data.from);
            await this.handleAnswer(data);
        });
        
        // Received ICE candidate
        this.socket.on('ice_candidate', async (data) => {
            console.log('Received ICE candidate from:', data.from);
            await this.handleIceCandidate(data);
        });
        
        // Room members list
        this.socket.on('room_members', (data) => {
            console.log('Room members:', data);
            this.updateMembersList(data.members);
        });
        
        // Error handling
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showError(error.message);
        });
    }
    
    async startPublishing() {
        console.log('Starting audio publishing...');
        
        try {
            // Get user media
            this.localStream = await navigator.mediaDevices.getUserMedia(this.audioConstraints);
            console.log('Got local audio stream');
            
            // Setup audio analysis
            this.setupAudioAnalysis(this.localStream);
            
            // Create peer connection
            await this.createPeerConnection();
            
            // Add tracks to peer connection
            this.localStream.getTracks().forEach(track => {
                this.peerConnection.addTrack(track, this.localStream);
            });
            
            // Create and send offer
            await this.createOffer();
            
            this.isPublisher = true;
            this.updateStatus('Publishing audio...');
            
        } catch (error) {
            console.error('Failed to start publishing:', error);
            this.showError('Failed to access microphone: ' + error.message);
        }
    }
    
    async startSubscribing(targetPeerId) {
        console.log('Starting audio subscription to:', targetPeerId);
        
        try {
            // Create peer connection
            await this.createPeerConnection();
            
            // Wait for remote stream
            this.peerConnection.ontrack = (event) => {
                console.log('Received remote track');
                this.remoteStream = event.streams[0];
                this.playRemoteAudio(this.remoteStream);
                this.setupAudioAnalysis(this.remoteStream);
            };
            
            this.isSubscriber = true;
            this.updateStatus('Subscribing to audio...');
            
        } catch (error) {
            console.error('Failed to start subscribing:', error);
            this.showError('Failed to subscribe: ' + error.message);
        }
    }
    
    async createPeerConnection() {
        this.peerConnection = new RTCPeerConnection(this.rtcConfig);
        
        // ICE candidate handler
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('Sending ICE candidate');
                this.socket.emit('ice_candidate', {
                    candidate: event.candidate,
                    to: null  // Server will handle routing
                });
            }
        };
        
        // Connection state monitoring
        this.peerConnection.onconnectionstatechange = () => {
            console.log('Connection state:', this.peerConnection.connectionState);
            this.updateConnectionStatus(this.peerConnection.connectionState);
        };
        
        // ICE connection state monitoring
        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', this.peerConnection.iceConnectionState);
            if (this.peerConnection.iceConnectionState === 'failed') {
                this.handleConnectionFailure();
            }
        };
    }
    
    async createOffer() {
        const offer = await this.peerConnection.createOffer({
            offerToReceiveAudio: true
        });
        
        await this.peerConnection.setLocalDescription(offer);
        
        console.log('Sending offer');
        this.socket.emit('offer', {
            offer: offer
        });
    }
    
    async handleOffer(data) {
        if (!this.peerConnection) {
            await this.createPeerConnection();
        }
        
        await this.peerConnection.setRemoteDescription(
            new RTCSessionDescription(data.offer)
        );
        
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);
        
        console.log('Sending answer');
        this.socket.emit('answer', {
            answer: answer,
            to: data.from
        });
    }
    
    async handleAnswer(data) {
        await this.peerConnection.setRemoteDescription(
            new RTCSessionDescription(data.answer)
        );
    }
    
    async handleIceCandidate(data) {
        try {
            await this.peerConnection.addIceCandidate(
                new RTCIceCandidate(data.candidate)
            );
        } catch (error) {
            console.error('Error adding ICE candidate:', error);
        }
    }
    
    async joinRoom(roomId) {
        console.log('Joining room:', roomId);
        this.roomId = roomId;
        
        return new Promise((resolve, reject) => {
            this.socket.emit('join_room', roomId, (success) => {
                if (success) {
                    console.log('Joined room successfully');
                    this.updateStatus(`Joined room: ${roomId}`);
                    resolve();
                } else {
                    reject(new Error('Failed to join room'));
                }
            });
        });
    }
    
    async leaveRoom() {
        if (this.roomId) {
            console.log('Leaving room:', this.roomId);
            this.socket.emit('leave_room', this.roomId);
            this.roomId = null;
        }
    }
    
    setupAudioAnalysis(stream) {
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        
        const bufferLength = this.analyser.frequencyBinCount;
        this.dataArray = new Uint8Array(bufferLength);
        
        source.connect(this.analyser);
        
        // Start visualization
        this.visualizeAudio();
    }
    
    visualizeAudio() {
        const canvas = document.getElementById('audioVisualizer');
        if (!canvas) return;
        
        const canvasCtx = canvas.getContext('2d');
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;
        
        const draw = () => {
            requestAnimationFrame(draw);
            
            this.analyser.getByteFrequencyData(this.dataArray);
            
            canvasCtx.fillStyle = 'rgb(0, 0, 0)';
            canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
            
            const barWidth = (WIDTH / this.dataArray.length) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < this.dataArray.length; i++) {
                barHeight = this.dataArray[i] / 2;
                
                canvasCtx.fillStyle = `rgb(50, ${barHeight + 100}, 50)`;
                canvasCtx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }
    
    playRemoteAudio(stream) {
        const audioElement = document.getElementById('remoteAudio');
        if (audioElement) {
            audioElement.srcObject = stream;
            audioElement.play().catch(e => {
                console.error('Error playing audio:', e);
                // User interaction required
                this.showMessage('Click to enable audio playback');
            });
        }
    }
    
    async updateAudioSettings(settings) {
        console.log('Updating audio settings:', settings);
        
        // Send metadata to server
        this.socket.emit('audio_stream_metadata', {
            config: {
                sampleRate: settings.sampleRate || 48000,
                channels: settings.channels || 2,
                bitDepth: settings.bitDepth || 16,
                codec: settings.codec || 'opus'
            },
            genre: settings.genre
        });
    }
    
    getAudioStats() {
        if (!this.peerConnection) return null;
        
        return this.peerConnection.getStats().then(stats => {
            const audioStats = {};
            
            stats.forEach(report => {
                if (report.type === 'inbound-rtp' && report.mediaType === 'audio') {
                    audioStats.inbound = {
                        bytesReceived: report.bytesReceived,
                        packetsReceived: report.packetsReceived,
                        packetsLost: report.packetsLost,
                        jitter: report.jitter,
                        audioLevel: report.audioLevel
                    };
                } else if (report.type === 'outbound-rtp' && report.mediaType === 'audio') {
                    audioStats.outbound = {
                        bytesSent: report.bytesSent,
                        packetsSent: report.packetsSent,
                        audioLevel: report.audioLevel
                    };
                }
            });
            
            return audioStats;
        });
    }
    
    // UI Helper methods
    initializeEventHandlers() {
        // Override in UI implementation
    }
    
    updateStatus(message) {
        console.log('Status:', message);
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
    
    updateConnectionStatus(state) {
        const indicator = document.getElementById('connectionIndicator');
        if (indicator) {
            indicator.className = `connection-${state}`;
            indicator.textContent = state;
        }
    }
    
    updateMembersList(members) {
        const list = document.getElementById('membersList');
        if (list) {
            list.innerHTML = '';
            members.forEach(member => {
                const item = document.createElement('li');
                item.textContent = `${member.peerId} ${member.isPublisher ? '🎤' : '🎧'}`;
                list.appendChild(item);
            });
        }
    }
    
    showError(message) {
        console.error('Error:', message);
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
    }
    
    showMessage(message) {
        const messageElement = document.getElementById('infoMessage');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.style.display = 'block';
            setTimeout(() => {
                messageElement.style.display = 'none';
            }, 3000);
        }
    }
    
    handlePeerDisconnect(sessionId) {
        // Clean up peer connection if needed
        if (this.peerConnection && this.peerConnection.connectionState !== 'closed') {
            // Handle specific peer disconnection
        }
    }
    
    handleConnectionFailure() {
        console.error('Connection failed, attempting to reconnect...');
        this.showError('Connection failed. Reconnecting...');
        
        // Cleanup and retry
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        // Retry logic
        setTimeout(() => {
            if (this.isPublisher) {
                this.startPublishing();
            } else if (this.isSubscriber) {
                // Retry subscription
            }
        }, 2000);
    }
    
    async disconnect() {
        console.log('Disconnecting...');
        
        // Stop local stream
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }
        
        // Close peer connection
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        
        // Leave room
        await this.leaveRoom();
        
        // Disconnect socket
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.updateStatus('Disconnected');
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioStreamingClient;
}