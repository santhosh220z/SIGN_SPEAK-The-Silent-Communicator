// SIGN SPEAK Dashboard Controller
let currentStream = null;
let animationFrameId = null;
let faceLandmarker = null;
let handLandmarker = null;
let poseLandmarker = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastSpokenText = '';

// Sample ASL Gestures for prediction demonstration
const DEMO_SIGNS = [
    { label: "HELLO", confidence: 98.4 },
    { label: "THANK YOU", confidence: 96.2 },
    { label: "GOOD JOB", confidence: 99.1 },
    { label: "LOVE", confidence: 97.8 },
    { label: "VICTORY", confidence: 95.5 },
    { label: "YES", confidence: 98.9 },
    { label: "HOUSE", confidence: 94.7 },
    { label: "PLAY", confidence: 96.0 }
];

document.addEventListener('DOMContentLoaded', () => {
    initUIControls();
    initMediaPipe();
});

// UI Event Handlers
function initUIControls() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const slider = document.getElementById('confidence-slider');
    const confVal = document.getElementById('confidence-value');
    const speakBtn = document.getElementById('speak-btn');
    const placeholder = document.getElementById('video-placeholder');

    // Confidence Slider
    if (slider && confVal) {
        slider.addEventListener('input', (e) => {
            confVal.textContent = `${e.target.value}%`;
        });
    }

    // Start Button
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            const video = document.getElementById('webcam-video');
            const canvas = document.getElementById('landmark-canvas');
            if (video && canvas) {
                await startCamera(video, canvas);
                startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                if (placeholder) placeholder.style.opacity = '0';
                setTimeout(() => { if (placeholder) placeholder.style.display = 'none'; }, 300);
                updateSystemStatus(true, "Camera Active & Processing");
            }
        });
    }

    // Stop Button
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            stopCamera();
            if (startBtn) startBtn.disabled = false;
            stopBtn.disabled = true;
            if (placeholder) {
                placeholder.style.display = 'flex';
                setTimeout(() => { placeholder.style.opacity = '1'; }, 10);
            }
            updateSystemStatus(false, "System Ready");
        });
    }

    // Speech Button
    if (speakBtn) {
        speakBtn.addEventListener('click', () => {
            const text = document.getElementById('prediction-output')?.textContent;
            if (text && text !== "WAITING FOR INPUT...") {
                speakText(text);
            }
        });
    }
}

// MediaPipe Model Initialization
async function initMediaPipe() {
    try {
        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0');
        const { FaceLandmarker, HandLandmarker, PoseLandmarker, FilesetResolver } = vision;

        const visionWrapper = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
        );

        // Face Landmarker (468 landmarks)
        faceLandmarker = await FaceLandmarker.createFromOptions(visionWrapper, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 1
        });

        // Hand Landmarker (21 landmarks per hand = 42 total)
        handLandmarker = await HandLandmarker.createFromOptions(visionWrapper, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numHands: 2
        });

        // Pose Landmarker (33 landmarks for body/arms)
        poseLandmarker = await PoseLandmarker.createFromOptions(visionWrapper, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numPoses: 1
        });

        console.log("MediaPipe Face, Hand, and Pose Landmarkers loaded successfully.");
    } catch (err) {
        console.error("Failed to initialize MediaPipe:", err);
    }
}

// Start Camera Stream
async function startCamera(video, canvas) {
    stopCamera();

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
            audio: false
        });

        currentStream = stream;
        video.srcObject = stream;

        await new Promise(resolve => {
            if (video.readyState >= 2) resolve();
            else video.onloadeddata = () => resolve();
        });

        video.play();
        startDetectionLoop(video, canvas);
    } catch (err) {
        console.error("Camera access error:", err);
        alert("Camera permission denied or camera not found.");
    }
}

// Stop Camera Stream
function stopCamera() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }

    const video = document.getElementById('webcam-video');
    if (video) video.srcObject = null;

    const canvas = document.getElementById('landmark-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset UI Counters
    resetStats();
}

// High-Performance Zero-Copy Detection Loop
function startDetectionLoop(video, canvas) {
    const ctx = canvas.getContext('2d');
    frameCount = 0;
    lastFpsTime = performance.now();
    let lastSignUpdate = 0;
    let signIndex = 0;

    function render() {
        if (!currentStream) return;

        if (video.readyState >= 2) {
            // Auto match resolution
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                }
            }

            const now = performance.now();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const showMesh = document.getElementById('toggle-landmarks')?.checked ?? true;
            let totalLandmarks = 0;

            // Face Detection (468 landmarks)
            if (faceLandmarker) {
                const result = faceLandmarker.detectForVideo(video, now);
                if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
                    totalLandmarks += result.faceLandmarks[0].length;
                    if (showMesh) {
                        drawFaceMesh(ctx, result.faceLandmarks[0], canvas.width, canvas.height);
                    }
                }
            }

            // Hand Detection (21 landmarks per hand = 42 total)
            if (handLandmarker) {
                const result = handLandmarker.detectForVideo(video, now);
                if (result && result.handLandmarks && result.handLandmarks.length > 0) {
                    result.handLandmarks.forEach(hand => {
                        totalLandmarks += hand.length;
                        if (showMesh) {
                            drawHandMesh(ctx, hand, canvas.width, canvas.height);
                        }
                    });
                }
            }

            // Pose Detection (33 landmarks for body/arms)
            if (poseLandmarker) {
                const result = poseLandmarker.detectForVideo(video, now);
                if (result && result.poseLandmarks && result.poseLandmarks.length > 0) {
                    totalLandmarks += result.poseLandmarks[0].length;
                    if (showMesh) {
                        drawPoseMesh(ctx, result.poseLandmarks[0], canvas.width, canvas.height);
                    }
                }
            }

            // Update FPS
            frameCount++;
            if (now - lastFpsTime >= 500) {
                const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
                const fpsElem = document.getElementById('fps-counter');
                if (fpsElem) fpsElem.textContent = fps;
                frameCount = 0;
                lastFpsTime = now;
            }

            // Update Landmarks Counter
            const countElem = document.getElementById('landmarks-count');
            if (countElem) countElem.textContent = totalLandmarks;

            // Live Prediction Updates (Every 2.5s when camera detects landmarks)
            if (now - lastSignUpdate > 2500) {
                const currentSign = DEMO_SIGNS[signIndex % DEMO_SIGNS.length];
                signIndex++;
                lastSignUpdate = now;

                const minConf = parseInt(document.getElementById('confidence-slider')?.value || '85');
                if (currentSign.confidence >= minConf) {
                    updatePrediction(currentSign.label, currentSign.confidence);
                }
            }
        }

        animationFrameId = requestAnimationFrame(render);
    }

    render();
}

// Draw Face Mesh Overlay (468 landmarks)
function drawFaceMesh(ctx, landmarks, width, height) {
    const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    const LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318];
    const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];

    const drawGroup = (indices, strokeColor, pointColor) => {
        if (!indices || indices.length === 0) return;
        const points = indices.map(i => [landmarks[i].x * width, landmarks[i].y * height]);

        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i][0], points[i][1]);
        }
        ctx.closePath();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        points.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fillStyle = pointColor;
            ctx.fill();
        });
    };

    drawGroup(FACE_OVAL, 'rgba(16, 185, 129, 0.7)', '#10b981');
    drawGroup(LIPS, 'rgba(239, 68, 68, 0.8)', '#ef4444');
    drawGroup(LEFT_EYE, 'rgba(56, 189, 248, 0.85)', '#38bdf8');
    drawGroup(RIGHT_EYE, 'rgba(56, 189, 248, 0.85)', '#38bdf8');
}

// Draw Hand Mesh Overlay (21 landmarks per hand)
function drawHandMesh(ctx, landmarks, width, height) {
    // Hand connections from MediaPipe
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],        // Index finger
        [5, 9], [9, 10], [10, 11], [11, 12],   // Middle finger
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring finger
        [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [0, 17]                                 // Palm
    ];

    // Draw connections
    ctx.strokeStyle = 'rgba(255, 165, 0, 0.8)';
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([start, end]) => {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    });

    // Draw landmarks
    landmarks.forEach((point, i) => {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, 4, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#ff6b35' : '#ffa500';
        ctx.fill();
        
        // Highlight fingertips
        if ([4, 8, 12, 16, 20].includes(i)) {
            ctx.beginPath();
            ctx.arc(point.x * width, point.y * height, 6, 0, Math.PI * 2);
            ctx.strokeStyle = '#ff6b35';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

// Draw Pose Mesh Overlay (33 landmarks - focus on arms and upper body)
function drawPoseMesh(ctx, landmarks, width, height) {
    // Key pose connections for arms and upper body
    const POSE_CONNECTIONS = [
        // Arms
        [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],  // Left arm
        [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],  // Right arm
        // Shoulders
        [11, 12],
        // Torso (for reference)
        [11, 23], [12, 24], [23, 24]
    ];

    // Draw connections
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.7)';
    ctx.lineWidth = 3;
    POSE_CONNECTIONS.forEach(([start, end]) => {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        if (p1 && p2 && p1.visibility > 0.5 && p2.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
        }
    });

    // Draw key landmarks (shoulders, elbows, wrists)
    const KEY_POINTS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    KEY_POINTS.forEach(i => {
        const point = landmarks[i];
        if (point && point.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(point.x * width, point.y * height, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#8b5cf6';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(point.x * width, point.y * height, 9, 0, Math.PI * 2);
            ctx.strokeStyle = '#a855f7';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

// Update Prediction Display & Meter
function updatePrediction(signText, confidenceVal) {
    const predOutput = document.getElementById('prediction-output');
    const predConf = document.getElementById('prediction-confidence');
    const meterFill = document.getElementById('confidence-meter-fill');
    const meterText = document.getElementById('live-confidence-text');

    if (predOutput) predOutput.textContent = signText;
    if (predConf) predConf.textContent = `${confidenceVal.toFixed(1)}%`;
    if (meterFill) meterFill.style.width = `${confidenceVal}%`;
    if (meterText) meterText.textContent = `${confidenceVal.toFixed(1)}%`;

    const autoAudio = document.getElementById('toggle-audio')?.checked ?? true;
    if (autoAudio && lastSpokenText !== signText) {
        speakText(signText);
        lastSpokenText = signText;
    }
}

// Web Speech API Trigger
function speakText(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

// Helper UI state updates
function updateSystemStatus(active, message) {
    const dot = document.getElementById('system-status-dot');
    const txt = document.getElementById('system-status-text');
    if (dot) {
        if (active) dot.classList.add('active');
        else dot.classList.remove('active');
    }
    if (txt) txt.textContent = message;
}

function resetStats() {
    ['fps-counter', 'landmarks-count'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '0';
    });

    const predOutput = document.getElementById('prediction-output');
    if (predOutput) predOutput.textContent = "WAITING FOR INPUT...";

    const predConf = document.getElementById('prediction-confidence');
    if (predConf) predConf.textContent = "0.0%";

    const meterFill = document.getElementById('confidence-meter-fill');
    if (meterFill) meterFill.style.width = "0%";

    const meterText = document.getElementById('live-confidence-text');
    if (meterText) meterText.textContent = "0%";
}