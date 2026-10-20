from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tempfile
import os

app = FastAPI(title="Video Analyzer")

def compute_metrics(frame, ref):
    """Compute multiple similarity metrics for better sensitivity"""
    # Resize if needed
    if frame.shape != ref.shape:
        frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]))
    
    # 1. SSIM (structural)
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ssim_score = float(ssim(gray1, gray2))
    
    # 2. Histogram correlation (color)
    hist1 = cv2.calcHist([frame], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist2 = cv2.calcHist([ref], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    
    # 3. Mean Absolute Difference (pixel)
    mad = float(np.mean(cv2.absdiff(frame, ref)))
    mad_score = max(0, 1 - (mad / 50))  # normalize roughly
    
    # Combined score (weighted)
    combined = (ssim_score * 0.5) + (hist_corr * 0.3) + (mad_score * 0.2)
    
    return {
        "ssim": round(ssim_score, 4),
        "hist": round(hist_corr, 4),
        "mad": round(mad, 2),
        "combined": round(combined, 4)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Analyzer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }
            h1 { color: #333; }
            .box { border: 2px dashed #ccc; padding: 30px; border-radius: 12px; text-align: center; margin: 15px 0; }
            button { background: #0070f3; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 16px; cursor: pointer; }
            button:hover { background: #0051cc; }
            #result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 12px; }
            .metric { display: inline-block; margin: 8px 16px; font-size: 16px; }
            .error { color: #dc3545; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #e9ecef; }
        </style>
    </head>
    <body>
        <h1>🎬 Video Analyzer</h1>
        <p>Upload video + reference image. Uses <strong>SSIM + Color Histogram + Pixel Difference</strong> for better detection.</p>
        
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
                Interval: <input type="number" name="interval" value="0.5" step="0.1" min="0.2" max="2" style="width:70px"> sec
                &nbsp;&nbsp; Threshold: <input type="number" name="threshold" value="0.70" step="0.05" min="0.4" max="0.95" style="width:70px">
            </p>
            <button type="submit">Analyze</button>
        </form>
        
        <div id="result" style="display:none;"></div>
        
        <script>
        document.getElementById('form').onsubmit = async (e) => {
            e.preventDefault();
            const result = document.getElementById('result');
            result.style.display = 'block';
            result.innerHTML = '<p>Analyzing... Please wait</p>';
            
            const formData = new FormData(e.target);
            try {
                const res = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (data.error) {
                    result.innerHTML = `<p class="error">${data.error}</p>`;
                    return;
                }
                
                let rows = data.details.map(d => 
                    `<tr><td>${d.time}</td><td>${d.ssim}</td><td>${d.hist}</td><td>${d.mad}</td><td><strong>${d.combined}</strong></td></tr>`
                ).join('');
                
                result.innerHTML = `
                    <h3>Results</h3>
                    <div class="metric"><strong>Avg Combined:</strong> ${data.avg_combined}</div>
                    <div class="metric"><strong>Min:</strong> ${data.min_combined}</div>
                    <div class="metric"><strong>Max:</strong> ${data.max_combined}</div>
                    <div class="metric"><strong>Flagged:</strong> ${data.low_percent}%</div>
                    <p><strong>Verdict:</strong> ${data.verdict}</p>
                    <h4>Per-frame scores (showing variation)</h4>
                    <table>
                        <tr><th>Time</th><th>SSIM</th><th>Histogram</th><th>MAD</th><th>Combined</th></tr>
                        ${rows}
                    </table>
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
    threshold: float = Form(0.70)
):
    try:
        video_bytes = await video.read()
        ref_bytes = await reference.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_bytes)
            video_path = tmp_vid.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ref:
            tmp_ref.write(ref_bytes)
            ref_path = tmp_ref.name
        
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return JSONResponse({"error": "Cannot read reference image"})
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot open video"})
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * interval))
        
        details = []
        combined_scores = []
        frame_idx = 0
        max_samples = 60  # limit for serverless
        
        while len(details) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                metrics = compute_metrics(frame, ref_img)
                timestamp = frame_idx / fps
                time_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                
                details.append({
                    "time": time_str,
                    "ssim": metrics["ssim"],
                    "hist": metrics["hist"],
                    "mad": metrics["mad"],
                    "combined": metrics["combined"]
                })
                combined_scores.append(metrics["combined"])
            
            frame_idx += 1
        
        cap.release()
        
        for p in [video_path, ref_path]:
            try: os.unlink(p)
            except: pass
        
        if not combined_scores:
            return JSONResponse({"error": "No frames analyzed"})
        
        scores = np.array(combined_scores)
        avg = float(np.mean(scores))
        low_count = sum(1 for s in combined_scores if s < threshold)
        
        if avg >= 0.85:
            verdict = "VERY SIMILAR"
        elif avg >= 0.70:
            verdict = "MODERATE DIFFERENCES"
        elif avg >= 0.50:
            verdict = "NOTICEABLE CHANGES"
        else:
            verdict = "VERY DIFFERENT / POSSIBLE EDIT"
        
        return {
            "avg_combined": round(avg, 4),
            "min_combined": round(float(np.min(scores)), 4),
            "max_combined": round(float(np.max(scores)), 4),
            "low_percent": round(100 * low_count / len(combined_scores), 1),
            "verdict": verdict,
            "details": details,
            "samples": len(details)
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)})
