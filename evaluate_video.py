#!/usr/bin/env python3

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
from PIL import Image
from collections import defaultdict
import json
import pickle
import logging
from pathlib import Path
import gc

# Import YOLO cho vehicle detection (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

# Import ReID model
import sys
sys.path.append('.')
from config import cfg
from model import make_model
from utils.metrics import R1_mAP_eval


class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required for vehicle detection")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = [2, 3, 5, 7]

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id in self.vehicle_classes and confidence > self.conf_threshold:
                        detections.append([x1, y1, x2, y2, confidence, class_id])
        return detections


class SimpleTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _calculate_centroid(self, box):
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, cent1, cent2):
        return np.sqrt((cent1[0] - cent2[0]) ** 2 + (cent1[1] - cent2[1]) ** 2)

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return {}
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
        else:
            object_centroids = {object_id: self._calculate_centroid(obj) for object_id, obj in self.objects.items()}
            detection_centroids = [self._calculate_centroid(d) for d in detections]
            used_detection_indices = set()
            used_object_ids = set()
            for object_id, object_centroid in object_centroids.items():
                min_distance = float('inf')
                min_index = -1
                for i, detection_centroid in enumerate(detection_centroids):
                    if i in used_detection_indices:
                        continue
                    distance = self._calculate_distance(object_centroid, detection_centroid)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                if min_index != -1 and min_distance < self.max_distance:
                    self.objects[object_id] = detections[min_index]
                    if object_id in self.disappeared:
                        del self.disappeared[object_id]
                    used_detection_indices.add(min_index)
                    used_object_ids.add(object_id)
                else:
                    if object_id not in self.disappeared:
                        self.disappeared[object_id] = 0
                    else:
                        self.disappeared[object_id] += 1
            for object_id in list(self.disappeared.keys()):
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    self._register(detection)
        return self.objects.copy()

    def _register(self, detection):
        self.objects[self.next_id] = detection
        self.next_id += 1


class VehicleInstanceDataset(Dataset):
    def __init__(self, instances_data, transform=None):
        self.instances_data = instances_data
        self.transform = transform

    def __len__(self):
        return len(self.instances_data)

    def __getitem__(self, idx):
        instance = self.instances_data[idx]
        image = instance['image']
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        vehicle_id = instance['vehicle_id']
        camera_id = instance['camera_id']
        return image, vehicle_id, camera_id, camera_id, 0, f"{instance['video_path']}_{instance['track_id']}"


class MultiVehicleReIDPipeline:
    def __init__(self, config_file, model_path, output_dir="./reid_output", batch_size=8, num_workers=0, use_fp16=False):
        self.config_file = config_file
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'reid_log_optimized.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        if YOLO_AVAILABLE:
            try:
                self.detector = VehicleDetector()
            except Exception:
                self.detector = None
                self.logger.warning("YOLO init failed; must use pre-extracted instances")
        else:
            self.detector = None
            self.logger.warning("YOLO not available; will need pre-extracted vehicle instances")
        self.transform = transforms.Compose([
            transforms.Resize(cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fp16 = use_fp16

    def extract_vehicle_instances(self, video_path, camera_id, max_instances_per_track=10):
        if self.detector is None:
            self.logger.error("Detector not available")
            return []
        self.logger.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return []
        tracker = SimpleTracker()
        instances = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps // 2))
        frame_id = 0
        track_instances_count = defaultdict(int)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval == 0:
                detections = self.detector.detect_vehicles(frame)
                tracked_objects = tracker.update(detections)
                for track_id, detection in tracked_objects.items():
                    if track_instances_count[track_id] >= max_instances_per_track:
                        continue
                    x1, y1, x2, y2, confidence, class_id = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size > 0:
                        vehicle_crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                        vehicle_id = f"{Path(video_path).stem}_{track_id}"
                        instances.append({
                            'image': vehicle_crop_rgb,
                            'vehicle_id': vehicle_id,
                            'camera_id': camera_id,
                            'video_path': video_path,
                            'track_id': track_id,
                            'frame_id': frame_id,
                            'confidence': confidence,
                            'class_id': class_id
                        })
                        track_instances_count[track_id] += 1
            frame_id += 1
            if frame_id % 100 == 0:
                self.logger.info(f"Processed {frame_id}/{total_frames} frames")
        cap.release()
        self.logger.info(f"Extracted {len(instances)} vehicle instances from {len(set(track_instances_count.keys()))} tracks")
        return instances

    def process_video_dataset(self, data_root, save_instances=True):
        self.logger.info(f"Processing dataset: {data_root}")
        all_instances = []
        for camera_folder in ['1', '2', '3', '4']:
            camera_dir = Path(data_root) / camera_folder
            if not camera_dir.exists():
                continue
            self.logger.info(f"Processing camera {camera_folder}")
            video_files = list(camera_dir.glob('*.MOV')) + list(camera_dir.glob('*.mp4')) + list(camera_dir.glob('*.avi')) + list(camera_dir.glob('*.mkv'))
            for video_file in video_files:
                instances = self.extract_vehicle_instances(str(video_file), int(camera_folder))
                all_instances.extend(instances)
        if save_instances:
            instances_file = self.output_dir / 'vehicle_instances.pkl'
            with open(instances_file, 'wb') as f:
                pickle.dump(all_instances, f)
            self.logger.info(f"Saved {len(all_instances)} instances to {instances_file}")
        return all_instances

    def create_reid_dataset(self, instances, min_instances_per_vehicle=2):
        vehicle_instances = defaultdict(list)
        for instance in instances:
            vehicle_instances[instance['vehicle_id']].append(instance)
        valid_instances = []
        for vehicle_id, vehicle_inst_list in vehicle_instances.items():
            if len(vehicle_inst_list) >= min_instances_per_vehicle:
                valid_instances.extend(vehicle_inst_list)
            else:
                self.logger.info(f"Skipping vehicle {vehicle_id}: only {len(vehicle_inst_list)} instances")
        self.logger.info(f"Valid instances: {len(valid_instances)} from {len(set([x['vehicle_id'] for x in valid_instances]))} vehicles")
        return valid_instances

    def evaluate_reid(self, instances):
        dataset = VehicleInstanceDataset(instances, transform=self.transform)
        vehicle_groups = defaultdict(list)
        for idx, instance in enumerate(instances):
            vehicle_groups[instance['vehicle_id']].append(idx)
        query_indices = []
        gallery_indices = []
        for vehicle_id, indices in vehicle_groups.items():
            if len(indices) >= 2:
                query_indices.append(indices[0])
                gallery_indices.extend(indices[1:])
        if len(query_indices) == 0:
            self.logger.error("No valid query/gallery pairs found")
            return
        all_indices = query_indices + gallery_indices
        subset_dataset = torch.utils.data.Subset(dataset, all_indices)
        # safer defaults for memory-constrained machines
        batch_size = max(1, int(self.batch_size))
        num_workers = max(0, int(self.num_workers))
        dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
        self.logger.info(f"Query instances: {len(query_indices)}")
        self.logger.info(f"Gallery instances: {len(gallery_indices)}")
        # Load checkpoint (cpu)
        try:
            ckpt = torch.load(self.model_path, map_location='cpu')
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {self.model_path}: {e}")
            raise
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        def _strip_module(key):
            return key.replace('module.', '') if key.startswith('module.') else key
        # Heuristic to pick num_classes: prefer keys that match model naming if available else prefer smaller classifier dim
        num_classes_from_ckpt = None
        matched_ckpt_key = None
        # Build a lightweight proto model once using cfg default to inspect keys
        proto_num = getattr(cfg, "MODEL_NUM_CLASS", None) or 1000
        proto_model = make_model(cfg, num_class=int(proto_num), camera_num=4, view_num=1)
        proto_state = proto_model.state_dict()
        candidate_keys = []
        for k, v in proto_state.items():
            if k.endswith('.weight') and getattr(v, 'ndim', 0) == 2:
                parts = k.split('.')
                if any(p in ('head', 'classifier', 'fc', 'last', 'cls') for p in parts[-3:]):
                    candidate_keys.append(k)
        for proto_k in candidate_keys:
            for ckpt_k, v in state_dict.items():
                if _strip_module(ckpt_k) == proto_k and hasattr(v, 'shape') and len(v.shape) >= 2:
                    num_classes_from_ckpt = int(v.shape[0])
                    matched_ckpt_key = ckpt_k
                    break
            if num_classes_from_ckpt is not None:
                break
        if num_classes_from_ckpt is None:
            candidates = []
            for k, v in state_dict.items():
                k2 = _strip_module(k)
                if k2.endswith('.weight') and ('classifier' in k2 or k2.endswith('fc.weight') or 'head' in k2 or 'cls' in k2):
                    if hasattr(v, 'shape') and len(v.shape) >= 2:
                        candidates.append((k, int(v.shape[0])))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                matched_ckpt_key, num_classes_from_ckpt = candidates[0]
                self.logger.info(f"Classifier candidates from ckpt: {[( _strip_module(k),d) for k,d in candidates]}; choosing {_strip_module(matched_ckpt_key)} -> {num_classes_from_ckpt}")
        if num_classes_from_ckpt is None:
            self.logger.warning("Could not infer num_classes from checkpoint; defaulting to proto_num")
            num_classes_from_ckpt = int(proto_num)
        self.logger.info(f"Inferred num_classes_from_ckpt={num_classes_from_ckpt} (matched_ckpt_key={_strip_module(matched_ckpt_key) if matched_ckpt_key else None})")
        # Rebuild model only if necessary
        if int(num_classes_from_ckpt) != int(proto_num):
            try:
                del proto_model
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            model = make_model(cfg, num_class=int(num_classes_from_ckpt), camera_num=4, view_num=1)
        else:
            model = proto_model
        # Safe copy of tensors that match shapes
        own_state = model.state_dict()
        loaded_keys = []
        skipped = []
        for ckpt_k, v in state_dict.items():
            k2 = _strip_module(ckpt_k)
            if k2 in own_state:
                try:
                    if own_state[k2].shape == v.shape:
                        own_state[k2].copy_(v)
                        loaded_keys.append(k2)
                    else:
                        skipped.append((k2, f"shape_mismatch model{tuple(own_state[k2].shape)} ckpt{tuple(v.shape)}"))
                except Exception as e:
                    skipped.append((k2, f"copy_error {e}"))
            else:
                skipped.append((k2, "not_in_model"))
        # assign and free checkpoint
        model.load_state_dict(own_state)
        try:
            del state_dict, ckpt
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"Safely loaded {len(loaded_keys)} tensors; skipped {len(skipped)} tensors.")
        if len(skipped) > 0:
            # log up to 20 skipped entries for debugging
            self.logger.debug("Skipped examples: " + ", ".join([f"{k}:{r}" for k, r in skipped[:20]]))
        # move model to device and (optionally) enable fp16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device)
        except Exception as e:
            self.logger.warning(f"Failed to move model to {device}: {e}; retrying with cpu")
            device = "cpu"
            model.to(device)
        scaler = None
        use_amp = False
        if self.use_fp16 and device == "cuda":
            use_amp = True
            try:
                model.half()
            except Exception as e:
                self.logger.warning(f"model.half() failed: {e}; continuing with fp32/autocast only")
        model.eval()
        evaluator = R1_mAP_eval(len(query_indices), max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        # Feature extraction loop: compute on device, move to CPU immediately
        with torch.no_grad():
            for batch_idx, (img, pid, camid, camids, target_view, imgpath) in enumerate(dataloader):
                try:
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            feat = model(img, cam_label=camids, view_label=target_view)
                    else:
                        feat = model(img, cam_label=camids, view_label=target_view)
                    feat_cpu = feat.detach().cpu()
                    if isinstance(pid, torch.Tensor):
                        pid_cpu = pid.cpu()
                    else:
                        pid_cpu = pid
                    if isinstance(camid, torch.Tensor):
                        camid_cpu = camid.cpu()
                    else:
                        camid_cpu = camid
                    evaluator.update((feat_cpu, pid_cpu, camid_cpu))
                    del feat, feat_cpu
                    gc.collect()
                    if (batch_idx + 1) % 10 == 0:
                        self.logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                except RuntimeError as e:
                    self.logger.error(f"RuntimeError during batch processing: {e}")
                    if 'out of memory' in str(e).lower():
                        self.logger.error("CUDA OOM detected during inference. Suggest lowering --batch_size and --num_workers.")
                    raise
        # compute metrics
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        self.logger.info("=" * 60)
        self.logger.info("MULTI-VEHICLE REID EVALUATION RESULTS (OPTIMIZED)")
        self.logger.info("=" * 60)
        self.logger.info(f"Total vehicle instances: {len(instances)}")
        self.logger.info(f"Unique vehicles: {len(set([x['vehicle_id'] for x in instances]))}")
        self.logger.info(f"Query instances: {len(query_indices)}")
        self.logger.info(f"Gallery instances: {len(gallery_indices)}")
        self.logger.info("-" * 40)
        self.logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"Rank-{r}: {cmc[r - 1]:.1%}")
        self.logger.info("=" * 60)
        return cmc, mAP


def main():
    parser = argparse.ArgumentParser(description="Multi-Vehicle ReID Evaluation (Optimized)")
    parser.add_argument("--config_file", required=True, help="Path to config file")
    parser.add_argument("--model_path", required=True, help="Path to trained ReID model")
    parser.add_argument("--data_root", required=True, help="Path to video dataset")
    parser.add_argument("--output_dir", default="./reid_output", help="Output directory")
    parser.add_argument("--load_instances", help="Load pre-extracted instances from file")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip instance extraction")
    parser.add_argument("--batch_size", default=8, type=int, help="DataLoader batch size (default 8)")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader num_workers (default 0)")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision fp16 where possible (requires CUDA)")

    args = parser.parse_args()
    pipeline = MultiVehicleReIDPipeline(args.config_file, args.model_path, args.output_dir, batch_size=args.batch_size, num_workers=args.num_workers, use_fp16=args.use_fp16)

    if args.load_instances and os.path.exists(args.load_instances):
        pipeline.logger.info(f"Loading instances from {args.load_instances}")
        with open(args.load_instances, 'rb') as f:
            all_instances = pickle.load(f)
    elif not args.skip_extraction:
        all_instances = pipeline.process_video_dataset(args.data_root)
    else:
        pipeline.logger.error("No instances to process")
        return

    reid_instances = pipeline.create_reid_dataset(all_instances)
    if len(reid_instances) == 0:
        pipeline.logger.error("No valid ReID instances found")
        return
    pipeline.evaluate_reid(reid_instances)


if __name__ == "__main__":
    main()
