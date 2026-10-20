from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tempfile
import os

app = FastAPI(title="Video Analyzer")

def compute_metrics(frame, ref):
    """More sensitive multi-metric comparison"""
    # Ensure same size
    if frame.shape != ref.shape:
        frame = cv2.resize(frame, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    
    # SSIM (structural)
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ssim_score = float(ssim(gray1, gray2, data_range=255))
    
    # Histogram Bhattacharyya (lower = more different)
    hist1 = cv2.calcHist([frame], [0,1,2], None, [16,16,16], [0,256]*3)
    hist2 = cv2.calcHist([ref], [0,1,2], None, [16,16,16], [0,256]*3)
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    hist_bhat = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    hist_score = max(0.0, 1.0 - hist_bhat * 1.5)  # more sensitive
    
    # Mean Absolute Difference (higher MAD = more different)
    mad = float(np.mean(cv2.absdiff(frame.astype(np.float32), ref.astype(np.float32))))
    mad_score = max(0.0, 1.0 - (mad / 40.0))  # stricter
    
    # Edge difference (detects structural changes better)
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    edge_diff = float(np.mean(cv2.absdiff(edges1, edges2)))
    edge_score = max(0.0, 1.0 - (edge_diff / 30.0))
    
    # Combined (stricter weights)
    combined = (ssim_score * 0.35) + (hist_score * 0.25) + (mad_score * 0.25) + (edge_score * 0.15)
    
    return {
        "ssim": round(ssim_score, 3),
        "hist": round(hist_score, 3),
        "mad": round(mad, 1),
        "edge": round(edge_score, 3),
        "combined": round(combined, 3)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #94a3b8;
            margin-bottom: 32px;
            font-size: 1.05rem;
        }
        .card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
        }
        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 640px) {
            .upload-grid { grid-template-columns: 1fr; }
        }
        .upload-box {
            border: 2px dashed #475569;
            border-radius: 12px;
            padding: 32px 20px;
            text-align: center;
            transition: all 0.2s;
            background: #0f172a;
        }
        .upload-box:hover {
            border-color: #60a5fa;
            background: #1e293b;
        }
        .upload-box p {
            margin-bottom: 12px;
            font-weight: 600;
            color: #cbd5e1;
        }
        input[type="file"] {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        .controls {
            display: flex;
            gap: 24px;
            align-items: center;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .controls label {
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .controls input {
            background: #0f172a;
            border: 1px solid #475569;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            width: 80px;
        }
        button {
            background: linear-gradient(90deg, #3b82f6, #6366f1);
            color: white;
            border: none;
            padding: 14px 36px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        }
        #result {
            display: none;
            margin-top: 8px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .metric-card {
            background: #0f172a;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #334155;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #60a5fa;
        }
        .metric-card .label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        .verdict {
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 16px 0;
            font-size: 1.1rem;
        }
        .verdict.high { background: #065f46; color: #6ee7b7; }
        .verdict.mid { background: #854d0e; color: #fde047; }
        .verdict.low { background: #7f1d1d; color: #fca5a5; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 0.9rem;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }
        th { color: #94a3b8; font-weight: 500; }
        .error { color: #f87171; background: #7f1d1d; padding: 12px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Analyzer</h1>
        <p class="subtitle">Detect changes or additions by comparing a video against a reference image.</p>
        
        <div class="card">
            <form id="form">
                <div class="upload-grid">
                    <div class="upload-box">
                        <p>Video</p>
                        <input type="file" name="video" accept="video/*" required>
                    </div>
                    <div class="upload-box">
                        <p>Reference Image</p>
                        <input type="file" name="reference" accept="image/*" required>
                    </div>
                </div>
                
                <div class="controls">
                    <label>Interval <input type="number" name="interval" value="0.5" step="0.1" min="0.2" max="2"> sec</label>
                    <label>Threshold <input type="number" name="threshold" value="0.65" step="0.05" min="0.3" max="0.9"></label>
                </div>
                
                <button type="submit">Analyze Video</button>
            </form>
        </div>
        
        <div class="card" id="result"></div>
    </div>

    <script>
    document.getElementById('form').onsubmit = async (e) => {
        e.preventDefault();
        const result = document.getElementById('result');
        result.style.display = 'block';
        result.innerHTML = '<p style="text-align:center; color:#94a3b8;">Analyzing... This may take 15-40 seconds</p>';
        
        const formData = new FormData(e.target);
        try {
            const res = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await res.json();
            
            if (data.error) {
                result.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }
            
            let verdictClass = 'mid';
            if (data.avg_combined >= 0.75) verdictClass = 'high';
            else if (data.avg_combined < 0.5) verdictClass = 'low';
            
            let rows = data.details.map(d => 
                `<tr><td>${d.time}</td><td>${d.ssim}</td><td>${d.hist}</td><td>${d.mad}</td><td><strong>${d.combined}</strong></td></tr>`
            ).join('');
            
            result.innerHTML = `
                <div class="metrics">
                    <div class="metric-card"><div class="value">${data.avg_combined}</div><div class="label">Avg Combined</div></div>
                    <div class="metric-card"><div class="value">${data.min_combined}</div><div class="label">Minimum</div></div>
                    <div class="metric-card"><div class="value">${data.max_combined}</div><div class="label">Maximum</div></div>
                    <div class="metric-card"><div class="value">${data.low_percent}%</div><div class="label">Flagged</div></div>
                </div>
                <div class="verdict ${verdictClass}">${data.verdict}</div>
                <h3 style="margin: 20px 0 8px; color:#cbd5e1;">Frame Scores</h3>
                <table>
                    <thead><tr><th>Time</th><th>SSIM</th><th>Hist</th><th>MAD</th><th>Combined</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
        } catch (err) {
            result.innerHTML = `<div class="error">Error: ${err.message}</div>`;
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
    threshold: float = Form(0.65)
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
        max_samples = 40
        
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
        
        if avg >= 0.80:
            verdict = "VERY SIMILAR — minimal changes detected"
        elif avg >= 0.60:
            verdict = "MODERATE DIFFERENCES"
        elif avg >= 0.40:
            verdict = "NOTICEABLE CHANGES OR DIFFERENT CONTENT"
        else:
            verdict = "VERY DIFFERENT — likely edited or unrelated"
        
        return {
            "avg_combined": round(avg, 3),
            "min_combined": round(float(np.min(scores)), 3),
            "max_combined": round(float(np.max(scores)), 3),
            "low_percent": round(100 * low_count / len(combined_scores), 1),
            "verdict": verdict,
            "details": details,
            "samples": len(details)
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)})
