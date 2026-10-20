from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import numpy as np
import imagehash
from PIL import Image
import tempfile
import os

VERSION = "v2.3.0"
app = FastAPI(title="Video Analyzer")

MAX_VIDEO_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_SIZE = 8 * 1024 * 1024   # 8 MB

def isolate_product(img):
    """Isolate light product"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 120])
    upper = np.array([180, 100, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
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
    
    h, w = img.shape[:2]
    return img[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]

def compute_orb_score(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=400)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        return 0.0
    
    matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m in matches if m.distance < 60]
    
    score = min(1.0, len(good) / 20.0)
    return float(score)

def compute_edge_score(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    edges1 = cv2.Canny(gray1, 40, 120)
    edges2 = cv2.Canny(gray2, 40, 120)
    
    intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
    union = np.logical_or(edges1 > 0, edges2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(min(1.0, iou * 2.5))

def compute_hash_score(img1, img2):
    def to_pil(cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    hash1 = imagehash.phash(to_pil(img1))
    hash2 = imagehash.phash(to_pil(img2))
    
    distance = hash1 - hash2
    score = 1.0 - (distance / 64.0)
    return max(0.0, float(score))

def compute_metrics(frame, ref):
    frame_prod = isolate_product(frame)
    ref_prod = isolate_product(ref)
    
    size = (240, 240)
    frame_r = cv2.resize(frame_prod, size, interpolation=cv2.INTER_AREA)
    ref_r = cv2.resize(ref_prod, size, interpolation=cv2.INTER_AREA)
    
    orb_score = compute_orb_score(frame_r, ref_r)
    edge_score = compute_edge_score(frame_r, ref_r)
    hash_score = compute_hash_score(frame_r, ref_r)
    
    hist1 = cv2.calcHist([frame_r], [0,1,2], None, [6,6,6], [0,256]*3)
    hist2 = cv2.calcHist([ref_r], [0,1,2], None, [6,6,6], [0,256]*3)
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    color_score = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))
    
    combined = (orb_score * 0.45) + (edge_score * 0.30) + (hash_score * 0.15) + (color_score * 0.10)
    
    return {
        "orb": round(orb_score * 100, 1),
        "edge": round(edge_score * 100, 1),
        "hash": round(hash_score * 100, 1),
        "color": round(color_score * 100, 1),
        "combined": round(combined * 100, 1)
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
            font-size: 4.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .overall-label {{ color: #94a3b8; font-size: 0.95rem; margin-top: 4px; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 12px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #0f172a;
            padding: 14px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .metric-card .value {{ font-size: 1.4rem; font-weight: 700; color: #60a5fa; }}
        .metric-card .label {{ font-size: 0.7rem; color: #94a3b8; margin-top: 4px; }}
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
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.78rem; }}
        th, td {{ padding: 8px 8px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 500; }}
        .error {{ color: #f87171; background: #7f1d1d; padding: 12px; border-radius: 8px; white-space: pre-wrap; }}
        .note {{
            background: #1e3a5f;
            border-left: 4px solid #60a5fa;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 16px;
            font-size: 0.9rem;
            color: #cbd5e1;
        }}
        .warning {{ color: #fbbf24; font-size: 0.85rem; margin-top: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Video Analyzer</h1>
            <span class="version">{VERSION}</span>
        </div>
        <p class="subtitle">Style-focused matching. Max video: 15MB, image: 8MB.</p>
        
        <div class="card">
            <form id="form">
                <div class="upload-grid">
                    <div class="upload-box">
                        <p>Video</p>
                        <input type="file" name="video" accept="video/*" required>
                        <div class="warning">Max 15MB</div>
                    </div>
                    <div class="upload-box">
                        <p>Reference Image</p>
                        <input type="file" name="reference" accept="image/*" required>
                        <div class="warning">Max 8MB</div>
                    </div>
                </div>
                
                <div class="controls">
                    <label>Interval <input type="number" name="interval" value="0.5" step="0.1" min="0.2" max="2"> sec</label>
                    <label>Threshold <input type="number" name="threshold" value="35" step="5" min="15" max="65"> %</label>
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
        result.innerHTML = '<p style="text-align:center; color:#94a3b8;">Analyzing style... 15-45 seconds</p>';
        
        const formData = new FormData(e.target);
        
        // Client-side size check
        const videoFile = e.target.video.files[0];
        const imgFile = e.target.reference.files[0];
        if (videoFile && videoFile.size > 15 * 1024 * 1024) {{
            result.innerHTML = '<div class="error">Video too large. Max 15MB.</div>';
            return;
        }}
        if (imgFile && imgFile.size > 8 * 1024 * 1024) {{
            result.innerHTML = '<div class="error">Image too large. Max 8MB.</div>';
            return;
        }}
        
        try {{
            const res = await fetch('/analyze', {{ method: 'POST', body: formData }});
            const text = await res.text();
            
            let data;
            try {{
                data = JSON.parse(text);
            }} catch {{
                result.innerHTML = `<div class="error">Server error (not JSON):\n${{text.substring(0, 500)}}</div>`;
                return;
            }}
            
            if (data.error) {{
                result.innerHTML = `<div class="error">${{data.error}}</div>`;
                return;
            }}
            
            let verdictClass = 'mid';
            if (data.overall_score >= 50) verdictClass = 'high';
            else if (data.overall_score < 30) verdictClass = 'low';
            
            let rows = data.details.map(d => 
                `<tr><td>${{d.time}}</td><td>${{d.orb}}%</td><td>${{d.edge}}%</td><td>${{d.hash}}%</td><td>${{d.color}}%</td><td><strong>${{d.combined}}%</strong></td></tr>`
            ).join('');
            
            result.innerHTML = `
                <div class="overall">
                    <div class="overall-score">${{data.overall_score}}%</div>
                    <div class="overall-label">Overall Style Score</div>
                </div>
                <div class="verdict ${{verdictClass}}">${{data.verdict}}</div>
                <div class="metrics">
                    <div class="metric-card"><div class="value">${{data.avg_orb}}%</div><div class="label">ORB Style</div></div>
                    <div class="metric-card"><div class="value">${{data.avg_edge}}%</div><div class="label">Edges</div></div>
                    <div class="metric-card"><div class="value">${{data.avg_hash}}%</div><div class="label">Hash</div></div>
                    <div class="metric-card"><div class="value">${{data.avg_color}}%</div><div class="label">Color</div></div>
                    <div class="metric-card"><div class="value">${{data.low_percent}}%</div><div class="label">Flagged</div></div>
                </div>
                <div class="note">
                    <strong>v2.3:</strong> Style-focused (ORB + Edges). Better error messages.
                </div>
                <h3 style="margin: 20px 0 8px; color:#cbd5e1;">Frame Details</h3>
                <table>
                    <thead><tr><th>Time</th><th>ORB</th><th>Edge</th><th>Hash</th><th>Color</th><th>Score</th></tr></thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }} catch (err) {{
            result.innerHTML = `<div class="error">Request failed: ${{err.message}}</div>`;
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
    threshold: float = Form(35)
):
    try:
        video_bytes = await video.read()
        ref_bytes = await reference.read()
        
        if len(video_bytes) > MAX_VIDEO_SIZE:
            return JSONResponse({"error": f"Video too large ({len(video_bytes)//1024//1024}MB). Max 15MB."})
        if len(ref_bytes) > MAX_IMAGE_SIZE:
            return JSONResponse({"error": f"Image too large. Max 8MB."})
        
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
        orb_scores = []
        edge_scores = []
        hash_scores = []
        color_scores = []
        frame_idx = 0
        max_samples = 25
        
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
                    "orb": metrics["orb"],
                    "edge": metrics["edge"],
                    "hash": metrics["hash"],
                    "color": metrics["color"],
                    "combined": metrics["combined"]
                })
                combined_scores.append(metrics["combined"])
                orb_scores.append(metrics["orb"])
                edge_scores.append(metrics["edge"])
                hash_scores.append(metrics["hash"])
                color_scores.append(metrics["color"])
            
            frame_idx += 1
        
        cap.release()
        
        for p in [video_path, ref_path]:
            try: os.unlink(p)
            except: pass
        
        if not combined_scores:
            return JSONResponse({"error": "No frames analyzed"})
        
        scores = np.array(combined_scores)
        overall = float(np.mean(scores))
        low_count = sum(1 for s in combined_scores if s < threshold)
        
        if overall >= 55:
            verdict = "GOOD STYLE MATCH — product shape aligns"
        elif overall >= 38:
            verdict = "MODERATE MATCH — similar style"
        elif overall >= 25:
            verdict = "PARTIAL MATCH — some style similarities"
        else:
            verdict = "POOR MATCH — different style"
        
        return {
            "overall_score": round(overall, 1),
            "avg_orb": round(float(np.mean(orb_scores)), 1),
            "avg_edge": round(float(np.mean(edge_scores)), 1),
            "avg_hash": round(float(np.mean(hash_scores)), 1),
            "avg_color": round(float(np.mean(color_scores)), 1),
            "min_combined": round(float(np.min(scores)), 1),
            "max_combined": round(float(np.max(scores)), 1),
            "low_percent": round(100 * low_count / len(combined_scores), 1),
            "verdict": verdict,
            "details": details,
            "samples": len(details),
            "version": VERSION
        }
        
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"})
