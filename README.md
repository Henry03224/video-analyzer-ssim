# Video Analyzer (SSIM)

Analyze a video against an image reference to detect if something was changed or added.

Uses **OpenCV + SSIM** (Structural Similarity Index).

## Features
- Compare video frames against a reference image
- Get average / min / max similarity score
- Flag timestamps with low similarity (possible edits or additions)
- Simple verdict

## Install
```bash
pip install -r requirements.txt
```

FFmpeg must be installed on your system (for metadata).

## Usage
```bash
python video_analyzer.py your_video.mp4 reference.jpg
```

### Options
```bash
python video_analyzer.py video.mp4 ref.jpg --interval 0.5 --threshold 0.75 --save-diff
```

- `--interval` : How often to sample (in seconds)
- `--threshold` : SSIM score below this will be flagged
- `--save-diff` : Save frames that are below threshold
- `-o report.json` : Save the report as JSON

## Score Meaning
| SSIM Score | Meaning |
|------------|---------|
| 0.90 – 1.00 | Very similar |
| 0.75 – 0.89 | Moderate differences |
| 0.50 – 0.74 | Noticeable changes |
| < 0.50 | Very different |

## License
MIT
