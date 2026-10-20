import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import tempfile
import os
from datetime import timedelta

st.set_page_config(page_title="Video Analyzer", page_icon="🎬", layout="wide")

st.title("🎬 Video Analyzer vs Image Reference")
st.markdown("I-upload ang video at reference image para malaman kung may binago o dinagdagan gamit ang **SSIM**.")

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)

def analyze(video_bytes, ref_bytes, interval=0.5, threshold=0.75):
    # Save temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(video_bytes)
        video_path = tmp_vid.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ref:
        tmp_ref.write(ref_bytes)
        ref_path = tmp_ref.name
    
    try:
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return None, "Hindi mabasa ang reference image"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Hindi mabuksan ang video"
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * interval))
        
        results = []
        low_similarity = []
        frame_idx = 0
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                h, w = ref_img.shape[:2]
                frame_resized = cv2.resize(frame, (w, h))
                score = compute_ssim(frame_resized, ref_img)
                timestamp = frame_idx / fps
                
                results.append({"timestamp": timestamp, "ssim": score})
                
                if score < threshold:
                    low_similarity.append({"timestamp": timestamp, "ssim": score})
                
                status.text(f"Processing... Frame {frame_idx} | SSIM: {score:.3f}")
            
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
        
        cap.release()
        progress_bar.empty()
        status.empty()
        
        if not results:
            return None, "Walang frames na na-analyze"
        
        scores = [r["ssim"] for r in results]
        report = {
            "samples": len(results),
            "average_ssim": round(float(np.mean(scores)), 4),
            "min_ssim": round(float(np.min(scores)), 4),
            "max_ssim": round(float(np.max(scores)), 4),
            "threshold": threshold,
            "low_count": len(low_similarity),
            "low_percent": round(100 * len(low_similarity) / len(results), 1),
            "flagged": [
                f"{int(item['timestamp']//60):02d}:{int(item['timestamp']%60):02d} (SSIM: {item['ssim']:.3f})"
                for item in low_similarity[:30]
            ]
        }
        return report, None
    
    finally:
        # Cleanup
        for p in [video_path, ref_path]:
            try:
                os.unlink(p)
            except:
                pass

# UI
col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])

with col2:
    ref_file = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png", "webp"])

st.divider()

c1, c2 = st.columns(2)
with c1:
    interval = st.slider("Sample interval (seconds)", 0.2, 2.0, 0.5, 0.1)
with c2:
    threshold = st.slider("SSIM Threshold", 0.5, 0.95, 0.75, 0.05)

if st.button("Analyze", type="primary", use_container_width=True):
    if video_file is None or ref_file is None:
        st.error("Kailangan ng video at reference image.")
    else:
        with st.spinner("Analyzing video..."):
            report, error = analyze(
                video_file.getvalue(),
                ref_file.getvalue(),
                interval=interval,
                threshold=threshold
            )
        
        if error:
            st.error(error)
        else:
            st.success("Analysis complete!")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average SSIM", f"{report['average_ssim']:.3f}")
            m2.metric("Min SSIM", f"{report['min_ssim']:.3f}")
            m3.metric("Max SSIM", f"{report['max_ssim']:.3f}")
            m4.metric("Low Similarity", f"{report['low_percent']}%")
            
            st.subheader("Flagged Timestamps")
            if report["flagged"]:
                for ts in report["flagged"]:
                    st.write(f"- {ts}")
            else:
                st.info("Walang frames na below threshold.")
            
            # Verdict
            avg = report["average_ssim"]
            if avg >= 0.90:
                verdict = "VERY SIMILAR — maliit ang pagbabago"
            elif avg >= 0.75:
                verdict = "MODERATE — may noticeable differences"
            elif avg >= 0.50:
                verdict = "LOW SIMILARITY — malaki ang binago"
            else:
                verdict = "VERY DIFFERENT"
            
            st.info(f"**Verdict:** {verdict}")

st.caption("Tip: Mas maliit na video = mas mabilis ang analysis. SSIM compares overall structure.")
