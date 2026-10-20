from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tempfile
import os
from io import BytesIO

app = FastAPI(title="Video Analyzer")

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Analyzer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            h1 { color: #333; }
            .box { border: 2px dashed #ccc; padding: 30px; border-radius: 12px; text-align: center; margin: 20px 0; }
            button { background: #0070f3; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 16px; cursor: pointer; }
            button:hover { background: #0051cc; }
            #result { margin-top: 30px; padding: 20px; background: #f5f5f5; border-radius: 12px; }
            .metric { display: inline-block; margin: 10px 20px; font-size: 18px; }
            .error { color: #e00; }
        </style>
    </head>
    <body>
        <h1>🎬 Video Analyzer</h1>
        <p>Upload a video and a reference image to check for changes using SSIM.</p>
        
        <form id="form">
            <div class="box">
                <p><strong>Video</strong></p>
                <input type="file" name="video" accept="video/*" required>
            </div>
            <div class="box">
                <p><strong>Reference Image</strong></p>
                <input type="file" name="reference" accept="image/*" required>
            </div>
            <p>
                Interval (seconds): <input type="number" name="interval" value="0.5" step="0.1" min="0.2" max="2" style="width:80px">
                &nbsp;&nbsp;
                Threshold: <input type="number" name="threshold" value="0.75" step="0.05" min="0.5" max="0.95" style="width:80px">
            </p>
            <button type="submit">Analyze</button>
        </form>
        
        <div id="result" style="display:none;"></div>
        
        <script>
        document.getElementById('form').onsubmit = async (e) => {
            e.preventDefault();
            const result = document.getElementById('result');
            result.style.display = 'block';
            result.innerHTML = '<p>Analyzing... Please wait (may take 10-40 seconds)</p>';
            
            const formData = new FormData(e.target);
            try {
                const res = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (data.error) {
                    result.innerHTML = `<p class="error">${data.error}</p>`;
                    return;
                }
                
                let flagged = data.flagged.length ? data.flagged.map(t => `<li>${t}</li>`).join('') : '<li>None</li>';
                
                result.innerHTML = `
                    <h3>Results</h3>
                    <div class="metric"><strong>Average SSIM:</strong> ${data.average_ssim}</div>
                    <div class="metric"><strong>Min:</strong> ${data.min_ssim}</div>
                    <div class="metric"><strong>Max:</strong> ${data.max_ssim}</div>
                    <div class="metric"><strong>Low similarity:</strong> ${data.low_percent}%</div>
                    <p><strong>Verdict:</strong> ${data.verdict}</p>
                    <h4>Flagged timestamps</h4>
                    <ul>${flagged}</ul>
                `;
            } catch (err) {
                result.innerHTML = `<p class="error">Error: ${err.message}</p>`;
            }
        };
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    reference: UploadFile = File(...),
    interval: float = Form(0.5),
    threshold: float = Form(0.75)
):
    try:
        # Read files
        video_bytes = await video.read()
        ref_bytes = await reference.read()
        
        # Temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_bytes)
            video_path = tmp_vid.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ref:
            tmp_ref.write(ref_bytes)
            ref_path = tmp_ref.name
        
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return JSONResponse({"error": "Hindi mabasa ang reference image"})
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Hindi mabuksan ang video"})
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * interval))
        
        results = []
        low_similarity = []
        frame_idx = 0
        max_frames = 300  # safety limit
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                h, w = ref_img.shape[:2]
                frame_resized = cv2.resize(frame, (w, h))
                score = compute_ssim(frame_resized, ref_img)
                timestamp = frame_idx / fps
                results.append(score)
                
                if score < threshold:
                    low_similarity.append(f"{int(timestamp//60):02d}:{int(timestamp%60):02d} (SSIM: {score:.3f})")
            
            frame_idx += 1
        
        cap.release()
        
        # Cleanup
        for p in [video_path, ref_path]:
            try: os.unlink(p)
            except: pass
        
        if not results:
            return JSONResponse({"error": "Walang frames na na-analyze"})
        
        scores = np.array(results)
        avg = float(np.mean(scores))
        
        if avg >= 0.90:
            verdict = "VERY SIMILAR — maliit ang pagbabago"
        elif avg >= 0.75:
            verdict = "MODERATE — may noticeable differences"
        elif avg >= 0.50:
            verdict = "LOW SIMILARITY — malaki ang binago"
        else:
            verdict = "VERY DIFFERENT"
        
        return {
            "average_ssim": round(avg, 4),
            "min_ssim": round(float(np.min(scores)), 4),
            "max_ssim": round(float(np.max(scores)), 4),
            "low_percent": round(100 * len(low_similarity) / len(results), 1),
            "flagged": low_similarity[:20],
            "verdict": verdict,
            "samples": len(results)
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)})
