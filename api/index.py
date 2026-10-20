from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tempfile
import os

VERSION = "v1.9.0"
app = FastAPI(title="Video Analyzer")

def isolate_product(img):
    """Isolate light product with broader color range"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Broader light color range
    lower = np.array([0, 0, 120])
    upper = np.array([180, 100, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get top 2 largest contours and combine
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            for c in contours[1:]:
                x2, y2, w2, h2 = cv2.boundingRect(c)
                x = min(x, x2)
                y = min(y, y2)
                w = max(x + w, x2 + w2) - x
                h = max(y + h, y2 + h2) - y
            
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + pad*2)
            h = min(img.shape[0] - y, h + pad*2)
            cropped = img[y:y+h, x:x+w]
            if cropped.size > 0 and min(cropped.shape[:2]) > 30:
                return cropped
    
    # Fallback
    h, w = img.shape[:2]
    return img[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]

def compute_metrics(frame, ref):
    """Lenient product comparison"""
    frame_prod = isolate_product(frame)
    ref_prod = isolate_product(ref)
    
    size = (250, 250)
    frame_r = cv2.resize(frame_prod, size, interpolation=cv2.INTER_AREA)
    ref_r = cv2.resize(ref_prod, size, interpolation=cv2.INTER_AREA)
    
    gray1 = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ref_r, cv2.COLOR_BGR2GRAY)
    
    # SSIM with lower weight
    ssim_score = float(ssim(gray1, gray2, data_range=255))
    
    # Color histogram - more lenient
    hist1 = cv2.calcHist([frame_r], [0,1,2], None, [8,8,8], [0,256]*3)
    hist2 = cv2.calcHist([ref_r], [0,1,2], None, [8,8,8], [0,256]*3)
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    hist_score = max(0.0, hist_corr)
    
    # Boost score if colors are reasonably similar
    if hist_score > 0.4:
        hist_score = min(1.0, hist_score * 1.3)
    
    # Diff
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    diff_percent = float(np.sum(thresh > 0) / thresh.size * 100)
    
    # More weight on color, less on exact structure
    combined = (ssim_score * 0.35) + (hist_score * 0.65)
    
    # Soften very low scores
    if combined < 0.3 and hist_score > 0.3:
        combined = combined * 1.4
    
    return {
        "ssim": round(ssim_score, 3),
        "hist": round(hist_score, 3),
        "diff_percent": round(diff_percent, 1),
        "combined": round(min(1.0, combined), 3)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analyzer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 8px;
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .version {{
            background: #334155;
            color: #94a3b8;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .subtitle {{ color: #94a3b8; margin-bottom: 32px; font-size: 1.05rem; }}
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
        }}
        .upload-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 640px) {{ .upload-grid {{ grid-template-columns: 1fr; }} }}
        .upload-box {{
            border: 2px dashed #475569;
            border-radius: 12px;
            padding: 32px 20px;
            text-align: center;
            transition: all 0.2s;
            background: #0f172a;
        }}
        .upload-box:hover {{ border-color: #60a5fa; background: #1e293b; }}
        .upload-box p {{ margin-bottom: 12px; font-weight: 600; color: #cbd5e1; }}
        input[type="file"] {{ color: #94a3b8; font-size: 0.9rem; }}
        .controls {{
            display: flex;
            gap: 24px;
            align-items: center;
            flex-wrap: wrap;
            margin: 20px 0;
        }}
        .controls label {{ color: #cbd5e1; font-size: 0.95rem; }}
        .controls input {{
            background: #0f172a;
            border: 1px solid #475569;
            color: white;
            padding: 6px 10px;
            border-radius: 6px;
            width: 80px;
        }}
        button {{
            background: linear-gradient(90deg, #3b82f6, #6366f1);
            color: white;
            border: none;
            padding: 14px 36px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        }}
        #result {{ display: none; margin-top: 8px; }}
        .overall {{
            text-align: center;
            margin: 10px 0 24px;
        }}
        .overall-score {{
            font-size: 4.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .overall-label {{ color: #94a3b8; font-size: 0.95rem; margin-top: 4px; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #0f172a;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: #60a5fa; }}
        .metric-card .label {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }}
        .verdict {{
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 16px 0;
            font-size: 1.1rem;
        }}
        .verdict.high {{ background: #065f46; color: #6ee7b7; }}
        .verdict.mid {{ background: #854d0e; color: #fde047; }}
        .verdict.low {{ background: #7f1d1d; color: #fca5a5; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.85rem; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 500; }}
        .error {{ color: #f87171; background: #7f1d1d; padding: 12px; border-radius: 8px; }}
        .note {{
            background: #1e3a5f;
            border-left: 4px solid #60a5fa;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 16px;
            font-size: 0.9rem;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Video Analyzer</h1>
            <span class="version">{VERSION}</span>
        </div>
        <p class="subtitle">Lenient product matching. Focuses on product color and type across different scenes.</p>
        
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
                    <label>Threshold <input type="number" name="threshold" value="0.30" step="0.05" min="0.15" max="0.60"></label>
                </div>
                
                <button type="submit">Analyze Video</button>
            </form>
        </div>
        
        <div class="card" id="result"></div>
    </div>

    <script>
    document.getElementById('form').onsubmit = async (e) => {{
        e.preventDefault();
        const result = document.getElementById('result');
        result.style.display = 'block';
        result.innerHTML = '<p style="text-align:center; color:#94a3b8;">Analyzing product... 15-40 seconds</p>';
        
        const formData = new FormData(e.target);
        try {{
            const res = await fetch('/analyze', {{ method: 'POST', body: formData }});
            const data = await res.json();
            
            if (data.error) {{
                result.innerHTML = `<div class="error">${{data.error}}</div>`;
                return;
            }}
            
            let verdictClass = 'mid';
            if (data.overall_score >= 0.45) verdictClass = 'high';
            else if (data.overall_score < 0.28) verdictClass = 'low';
            
            let rows = data.details.map(d => 
                `<tr><td>${{d.time}}</td><td>${{d.ssim}}</td><td>${{d.hist}}</td><td>${{d.diff_percent}}%</td><td><strong>${{d.combined}}</strong></td></tr>`
            ).join('');
            
            result.innerHTML = `
                <div class="overall">
                    <div class="overall-score">${{data.overall_score}}</div>
                    <div class="overall-label">Overall Product Score</div>
                </div>
                <div class="verdict ${{verdictClass}}">${{data.verdict}}</div>
                <div class="metrics">
                    <div class="metric-card"><div class="value">${{data.min_combined}}</div><div class="label">Min Score</div></div>
                    <div class="metric-card"><div class="value">${{data.max_combined}}</div><div class="label">Max Score</div></div>
                    <div class="metric-card"><div class="value">${{data.avg_diff}}%</div><div class="label">Diff Area</div></div>
                    <div class="metric-card"><div class="value">${{data.low_percent}}%</div><div class="label">Flagged</div></div>
                </div>
                <div class="note">
                    <strong>Lenient Mode:</strong> Prioritizes product color similarity. More forgiving of different backgrounds and angles.
                </div>
                <h3 style="margin: 20px 0 8px; color:#cbd5e1;">Frame Details</h3>
                <table>
                    <thead><tr><th>Time</th><th>SSIM</th><th>Hist</th><th>Diff %</th><th>Score</th></tr></thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }} catch (err) {{
            result.innerHTML = `<div class="error">Error: ${{err.message}}</div>`;
        }}
    }};
    </script>
</body>
</html>
"""

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    reference: UploadFile = File(...),
    interval: float = Form(0.5),
    threshold: float = Form(0.30)
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
        diff_percents = []
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
                    "diff_percent": metrics["diff_percent"],
                    "combined": metrics["combined"]
                })
                combined_scores.append(metrics["combined"])
                diff_percents.append(metrics["diff_percent"])
            
            frame_idx += 1
        
        cap.release()
        
        for p in [video_path, ref_path]:
            try: os.unlink(p)
            except: pass
        
        if not combined_scores:
            return JSONResponse({"error": "No frames analyzed"})
        
        scores = np.array(combined_scores)
        overall = float(np.mean(scores))
        avg_diff = float(np.mean(diff_percents))
        low_count = sum(1 for s in combined_scores if s < threshold)
        
        if overall >= 0.50:
            verdict = "GOOD MATCH — product type is consistent"
        elif overall >= 0.35:
            verdict = "MODERATE MATCH — similar product"
        elif overall >= 0.22:
            verdict = "PARTIAL MATCH — some similarities"
        else:
            verdict = "POOR MATCH — different product"
        
        return {
            "overall_score": round(overall, 3),
            "min_combined": round(float(np.min(scores)), 3),
            "max_combined": round(float(np.max(scores)), 3),
            "avg_diff": round(avg_diff, 1),
            "low_percent": round(100 * low_count / len(combined_scores), 1),
            "verdict": verdict,
            "details": details,
            "samples": len(details),
            "version": VERSION
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)})
