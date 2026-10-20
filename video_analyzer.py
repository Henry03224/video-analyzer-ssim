#!/usr/bin/env python3
"""
Video Analyzer vs Image Reference
Gumagamit ng OpenCV + SSIM para i-check kung may binago/dinagdagan sa video
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import subprocess
import json
import argparse
from datetime import timedelta

def get_video_metadata(video_path):
    """Kunin ang metadata gamit ang ffprobe (FFmpeg)"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        format_info = data.get("format", {})
        
        fps_str = stream.get("r_frame_rate", "30/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den else 30
        
        duration = float(format_info.get("duration", stream.get("duration", 0)))
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        
        return {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height
        }
    except Exception as e:
        print(f"Warning: Hindi makuha ang metadata via ffprobe: {e}")
        return {"fps": 30, "duration": 0, "width": 0, "height": 0}

def load_and_resize(image, target_size):
    """I-resize ang image para mag-match sa target"""
    if image is None:
        return None
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def compute_ssim(img1, img2):
    """I-compute ang SSIM score (0-1, mas mataas = mas similar)"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score)

def analyze_video(video_path, reference_path, interval=0.5, threshold=0.75, save_diff=False):
    video_path = Path(video_path)
    reference_path = Path(reference_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    
    ref_img = cv2.imread(str(reference_path))
    if ref_img is None:
        raise ValueError("Hindi mabasa ang reference image")
    
    meta = get_video_metadata(video_path)
    print(f"Video: {meta['width']}x{meta['height']} | {meta['fps']:.2f} fps | {meta['duration']:.2f}s")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Hindi mabuksan ang video")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or meta["fps"]
    frame_interval = max(1, int(fps * interval))
    
    results = []
    low_similarity = []
    frame_idx = 0
    sample_count = 0
    
    print(f"Sampling every {interval}s ({frame_interval} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            h, w = ref_img.shape[:2]
            frame_resized = load_and_resize(frame, (w, h))
            
            if frame_resized is not None:
                score = compute_ssim(frame_resized, ref_img)
                timestamp = frame_idx / fps
                
                results.append({
                    "frame": frame_idx,
                    "timestamp": timestamp,
                    "ssim": score
                })
                
                if score < threshold:
                    low_similarity.append({
                        "timestamp": timestamp,
                        "ssim": score,
                        "frame": frame_idx
                    })
                    
                    if save_diff:
                        out_path = video_path.parent / f"diff_{int(timestamp)}s.jpg"
                        cv2.imwrite(str(out_path), frame)
                
                sample_count += 1
                if sample_count % 20 == 0:
                    print(f"  Processed {sample_count} samples... current SSIM: {score:.3f}")
        
        frame_idx += 1
    
    cap.release()
    
    if not results:
        return {"error": "Walang frames na na-analyze"}
    
    scores = [r["ssim"] for r in results]
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    report = {
        "video": str(video_path.name),
        "reference": str(reference_path.name),
        "duration_sec": meta["duration"],
        "samples": len(results),
        "average_ssim": round(avg_score, 4),
        "min_ssim": round(min_score, 4),
        "max_ssim": round(max_score, 4),
        "threshold": threshold,
        "low_similarity_count": len(low_similarity),
        "low_similarity_percent": round(100 * len(low_similarity) / len(results), 1),
        "flagged_timestamps": [
            f"{td.seconds//60:02d}:{td.seconds%60:02d} (SSIM: {item['ssim']:.3f})"
            for item in low_similarity
            for td in [timedelta(seconds=int(item["timestamp"]))]
        ]
    }
    
    return report, results, low_similarity

def print_report(report):
    print("\n" + "="*60)
    print("VIDEO ANALYSIS REPORT")
    print("="*60)
    print(f"Video          : {report['video']}")
    print(f"Reference      : {report['reference']}")
    print(f"Duration       : {report['duration_sec']:.2f} seconds")
    print(f"Samples taken  : {report['samples']}")
    print("-"*60)
    print(f"Average SSIM   : {report['average_ssim']:.4f}")
    print(f"Min SSIM       : {report['min_ssim']:.4f}")
    print(f"Max SSIM       : {report['max_ssim']:.4f}")
    print(f"Threshold      : {report['threshold']}")
    print("-"*60)
    print(f"Low similarity : {report['low_similarity_count']} frames ({report['low_similarity_percent']}%)")
    
    if report["flagged_timestamps"]:
        print("\nFlagged timestamps (possible changes):")
        for ts in report["flagged_timestamps"][:20]:
            print(f"  - {ts}")
        if len(report["flagged_timestamps"]) > 20:
            print(f"  ... at {len(report['flagged_timestamps'])-20} pa")
    else:
        print("\nWalang frames na below threshold — mukhang consistent ang video sa reference.")
    
    print("="*60)
    
    avg = report["average_ssim"]
    if avg >= 0.90:
        verdict = "VERY SIMILAR — maliit ang pagbabago"
    elif avg >= 0.75:
        verdict = "MODERATE — may noticeable differences"
    elif avg >= 0.50:
        verdict = "LOW SIMILARITY — malaki ang binago o ibang content"
    else:
        verdict = "VERY DIFFERENT — malamang ibang scene o heavily edited"
    
    print(f"Verdict: {verdict}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Analyze video against image reference using SSIM")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("reference", help="Path to reference image")
    parser.add_argument("--interval", type=float, default=0.5, help="Sample interval in seconds (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=0.75, help="SSIM threshold for flagging (default: 0.75)")
    parser.add_argument("--save-diff", action="store_true", help="Save frames with low similarity")
    parser.add_argument("--output", "-o", help="Save report as JSON")
    
    args = parser.parse_args()
    
    try:
        report, _, _ = analyze_video(
            args.video,
            args.reference,
            interval=args.interval,
            threshold=args.threshold,
            save_diff=args.save_diff
        )
        
        print_report(report)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
