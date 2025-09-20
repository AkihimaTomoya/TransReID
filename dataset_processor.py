
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Processor for Multi-Camera Vehicle ReID
- Quét cấu trúc thư mục RealData/ do bạn mô tả
- Sinh dataset_config.json đúng format multi_camera_vehicle_reid.py yêu cầu
- Validate nhanh các đường dẫn/metadata
- Xuất summary CSV
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

try:
    import cv2
except Exception:
    cv2 = None

DEFAULT_FPS = 25

def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

def infer_camera_id(video_path: Path) -> str:
    """
    Ưu tiên tên thư mục chứa file (ví dụ: goc_1/ goc_2/ camera_001/ ...).
    Nếu không rõ, dùng stem của file.
    """
    parent_name = video_path.parent.name
    if parent_name and parent_name != "":  # ex: goc_1
        return parent_name
    return f"cam_{video_path.stem}"

def read_optional_metadata(video_path: Path) -> Dict[str, Any]:
    """
    Nếu cùng thư mục có metadata.json, đọc để lấy location/start_time/fps.
    Nếu không có, gán mặc định.
    """
    md_file = video_path.parent / "metadata.json"
    meta = {}
    if md_file.exists():
        try:
            meta = json.loads(md_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    # defaults
    meta.setdefault("location", {"x": 0, "y": 0, "description": video_path.parent.as_posix()})
    meta.setdefault("start_time", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    meta.setdefault("fps", DEFAULT_FPS)
    return meta

def scan_realdata(dataset_root: Path) -> Dict[str, Any]:
    """
    Hỗ trợ cấu trúc bạn đưa ra:
    RealData/
    ├── 1/
    │      └── goc_1.mov
    ├── 1/2/
    │      └── goc_2.mov
    ...
    └── dataID/
           ├── goc_1/1/view1/*.jpg ...
    Trình quét sẽ tìm TẤT CẢ các file video trong cây thư mục và coi mỗi file là một camera.
    """
    cameras = {}
    for p in dataset_root.rglob("*"):
        if p.is_file() and is_video_file(p):
            cam_id = infer_camera_id(p)
            meta = read_optional_metadata(p)

            cameras[cam_id] = {
                "video_path": p.as_posix(),
                "location": meta["location"],
                "start_time": meta["start_time"],
                "fps": meta["fps"]
            }
    config = {
        "dataset_name": dataset_root.name,
        "description": "Auto-generated config from dataset_processor.py",
        "cameras": cameras
    }
    return config

def validate_config(config: Dict[str, Any]) -> List[str]:
    errors = []
    cams = config.get("cameras", {})
    if not cams:
        errors.append("No cameras found in config.")
        return errors
    for cam_id, info in cams.items():
        vp = info.get("video_path", "")
        if not vp or not Path(vp).exists():
            errors.append(f"[{cam_id}] video not found: {vp}")
        if "start_time" not in info:
            errors.append(f"[{cam_id}] missing start_time")
        if "fps" not in info:
            errors.append(f"[{cam_id}] missing fps")
        if "location" not in info:
            errors.append(f"[{cam_id}] missing location")
    return errors

def write_summary_csv(config: Dict[str, Any], out_csv: Path):
    """
    Xuất CSV tóm tắt: camera_id, video_path, size_bytes, fps, start_time, width, height, frame_count, duration_s
    Nếu không có OpenCV, chỉ xuất các cột cơ bản.
    """
    import csv
    header = ["camera_id", "video_path", "size_bytes", "fps", "start_time", "width", "height", "frame_count", "duration_s"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for cam_id, info in config.get("cameras", {}).items():
            vp = Path(info["video_path"])
            row = {
                "camera_id": cam_id,
                "video_path": vp.as_posix(),
                "size_bytes": vp.stat().st_size if vp.exists() else 0,
                "fps": info.get("fps", DEFAULT_FPS),
                "start_time": info.get("start_time", ""),
                "width": "",
                "height": "",
                "frame_count": "",
                "duration_s": ""
            }
            if cv2 is not None and vp.exists():
                try:
                    cap = cv2.VideoCapture(vp.as_posix())
                    if cap.isOpened():
                        wdh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps_vid = cap.get(cv2.CAP_PROP_FPS) or row["fps"]
                        dur = frames / fps_vid if fps_vid else ""
                        row.update({
                            "width": wdh, "height": hgt,
                            "frame_count": frames, "duration_s": round(dur, 2) if dur else ""
                        })
                    cap.release()
                except Exception:
                    pass
            w.writerow(row)

def save_config(config: Dict[str, Any], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Dataset processor for RealData → dataset_config.json")
    ap.add_argument("--mode", choices=["scan", "validate", "summary"], required=True)
    ap.add_argument("--dataset_root", help="Path to RealData/ (or any root that contains videos)")
    ap.add_argument("--output_config", help="Where to write dataset_config.json")
    ap.add_argument("--config_file", help="Existing config file for validate/summary modes")
    ap.add_argument("--output_summary", help="Where to write summary CSV")
    args = ap.parse_args()

    if args.mode == "scan":
        if not args.dataset_root or not args.output_config:
            print("scan mode requires --dataset_root and --output_config")
            return 1
        root = Path(args.dataset_root)
        if not root.exists():
            print(f"Dataset root not found: {root}")
            return 1
        config = scan_realdata(root)
        save_config(config, Path(args.output_config))
        print(f"Config written to: {args.output_config}")
        return 0

    if args.mode == "validate":
        cfg_path = Path(args.config_file) if args.config_file else None
        if not cfg_path or not cfg_path.exists():
            print("validate mode requires --config_file (exists)")
            return 1
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        errors = validate_config(config)
        if errors:
            print("Validation errors:")
            for e in errors:
                print(" -", e)
            return 2
        print("Validation OK")
        return 0

    if args.mode == "summary":
        cfg_path = Path(args.config_file) if args.config_file else None
        if not cfg_path or not cfg_path.exists():
            print("summary mode requires --config_file (exists)")
            return 1
        out_csv = Path(args.output_summary) if args.output_summary else Path("processing_summary.csv")
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        write_summary_csv(config, out_csv)
        print(f"Summary written to: {out_csv}")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
