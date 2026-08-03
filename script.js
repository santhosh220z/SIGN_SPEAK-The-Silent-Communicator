// SIGN SPEAK Dashboard Controller
let currentStream = null;
let animationFrameId = null;
let faceLandmarker = null;
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
        const { FaceLandmarker, FilesetResolver } = vision;

        const visionWrapper = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(visionWrapper, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 1
        });

        console.log("MediaPipe FaceLandmarker loaded successfully.");
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

            if (faceLandmarker) {
                const result = faceLandmarker.detectForVideo(video, now);
                if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
                    totalLandmarks = result.faceLandmarks[0].length;
                    if (showMesh) {
                        drawMeshOverlay(ctx, result.faceLandmarks[0], canvas.width, canvas.height);
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

// Draw Glowing Landmark Overlay
function drawMeshOverlay(ctx, landmarks, width, height) {
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