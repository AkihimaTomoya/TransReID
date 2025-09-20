#!/usr/bin/env python3
"""
Multi-Camera Vehicle Re-Identification System
Theo dõi xe qua nhiều camera tại các vị trí khác nhau
"""

import os
import cv2
import torch
import numpy as np

# --- JSON helpers: make numpy types serializable ---
def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    return str(o)

import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict, deque
import argparse
from pathlib import Path
import pandas as pd

from PIL import Image
import torchvision.transforms as T
from config import cfg
from model import make_model
from ultralytics import YOLO


class MultiCameraVehicleReID:
    def __init__(self, config_file, model_weight, camera_config_file=None):
        """
        Multi-Camera Vehicle Re-ID System

        Args:
            config_file: Path to ReID model config
            model_weight: Path to trained ReID model
            camera_config_file: JSON file chứa thông tin các camera
        """
        # Load ReID model config
        cfg.merge_from_file(config_file)
        cfg.freeze()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup transforms
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        # Load ReID model
        self.reid_model = make_model(cfg, num_class=576, camera_num=20, view_num=8)
        self.reid_model.load_param(model_weight)
        self.reid_model.to(self.device)
        self.reid_model.eval()

        # Load vehicle detection model
        self.detection_model = YOLO('yolov8n.pt')

        # Vehicle classes in COCO
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        # Camera configuration
        self.cameras = {}
        if camera_config_file and os.path.exists(camera_config_file):
            with open(camera_config_file, 'r') as f:
                self.cameras = json.load(f)

        # Vehicle database - lưu trữ toàn bộ thông tin xe
        self.vehicle_database = {}  # {vehicle_global_id: VehicleInfo}
        self.next_global_id = 1

        # Mapping from camera_id (str) to numeric label
        self.cam_id_to_label = {}

        # Tracking parameters
        self.similarity_threshold = 0.7
        self.temporal_threshold = 300  # seconds - max time gap between appearances
        self.spatial_threshold = 1000  # meters - max distance between cameras

        print(f"Multi-Camera Vehicle ReID System initialized")
        print(f"Device: {self.device}")
        print(f"Cameras configured: {len(self.cameras)}")

    def detect_vehicles_in_frame(self, frame, camera_id, timestamp, conf_threshold=0.5):
        """Detect vehicles in a single frame"""
        detections = []

        # Run YOLO detection
        results = self.detection_model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])

                    if class_id in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])

                        if confidence > conf_threshold:
                            # Filter minimum size
                            width, height = x2 - x1, y2 - y1
                            if width > 50 and height > 50:
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'class': self.vehicle_classes[class_id],
                                    'camera_id': camera_id,
                                    'timestamp': timestamp
                                }
                                detections.append(detection)

        return detections

    def extract_vehicle_features(self, vehicle_crop, camera_id=0, view_id=0):
        """Extract ReID features from vehicle crop"""
        if isinstance(vehicle_crop, np.ndarray):
            vehicle_crop = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))

        img_tensor = self.transform(vehicle_crop).unsqueeze(0).to(self.device)

        cam_key = str(camera_id)
        cam_label = self.cam_id_to_label.get(cam_key, 0)
        view_label = int(view_id) if not isinstance(view_id, str) else 0

        cam_tensor = torch.tensor([cam_label], dtype=torch.long, device=self.device)
        view_tensor = torch.tensor([view_label], dtype=torch.long, device=self.device)

        with torch.no_grad():
            features = self.reid_model(img_tensor, cam_label=cam_tensor, view_label=view_tensor)

        return features.cpu().numpy().flatten()

    def compute_feature_similarity(self, feat1, feat2):
        """Compute cosine similarity between features"""
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        return np.dot(feat1_norm, feat2_norm)

    def compute_temporal_compatibility(self, timestamp1, timestamp2):
        """Check if two timestamps are temporally compatible"""
        if isinstance(timestamp1, str):
            timestamp1 = datetime.fromisoformat(timestamp1)
        if isinstance(timestamp2, str):
            timestamp2 = datetime.fromisoformat(timestamp2)

        time_diff = abs((timestamp2 - timestamp1).total_seconds())
        return time_diff <= self.temporal_threshold

    def compute_spatial_compatibility(self, camera1_id, camera2_id):
        """Check if two cameras are spatially compatible"""
        if camera1_id not in self.cameras or camera2_id not in self.cameras:
            return True  # Assume compatible if no location info

        cam1_info = self.cameras[camera1_id]
        cam2_info = self.cameras[camera2_id]

        # Simple distance calculation (you can improve this)
        if 'location' in cam1_info and 'location' in cam2_info:
            loc1 = cam1_info['location']
            loc2 = cam2_info['location']

            # Euclidean distance (simplified)
            distance = np.sqrt((loc1['x'] - loc2['x']) ** 2 + (loc1['y'] - loc2['y']) ** 2)
            return distance <= self.spatial_threshold

        return True

    def find_matching_vehicle(self, new_detection, new_features, camera_id, timestamp):
        """
        Find matching vehicle in database based on features, temporal and spatial constraints
        """
        best_match_id = None
        best_similarity = 0

        for global_id, vehicle_info in self.vehicle_database.items():
            # Skip if same camera (assuming continuous tracking within camera)
            last_appearance = vehicle_info['appearances'][-1]
            if last_appearance['camera_id'] == camera_id:
                # Check if too close in time (might be same detection)
                last_timestamp = datetime.fromisoformat(last_appearance['timestamp'])
                current_timestamp = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
                if (current_timestamp - last_timestamp).total_seconds() < 5:
                    continue

            # Check temporal compatibility with any previous appearance
            temporally_compatible = False
            spatially_compatible = False

            for appearance in vehicle_info['appearances']:
                if self.compute_temporal_compatibility(appearance['timestamp'], timestamp):
                    temporally_compatible = True
                    if self.compute_spatial_compatibility(appearance['camera_id'], camera_id):
                        spatially_compatible = True
                        break

            if not (temporally_compatible and spatially_compatible):
                continue

            # Compute feature similarity with representative features
            similarity = self.compute_feature_similarity(new_features, vehicle_info['representative_features'])

            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = global_id

        return best_match_id, best_similarity

    def add_or_update_vehicle(self, detection, features, camera_id, timestamp, frame_idx, global_id=None):
        """Add new vehicle or update existing vehicle in database"""

        appearance = {
            'camera_id': camera_id,
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'frame_idx': frame_idx,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class': detection['class'],
            'features': features
        }

        if global_id is None:
            # Create new vehicle
            global_id = self.next_global_id
            self.next_global_id += 1

            self.vehicle_database[global_id] = {
                'global_id': global_id,
                'vehicle_class': detection['class'],
                'first_seen': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                'last_seen': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                'appearances': [appearance],
                'representative_features': features.copy(),
                'cameras_seen': {camera_id},
                'total_appearances': 1
            }
        else:
            # Update existing vehicle
            vehicle_info = self.vehicle_database[global_id]
            vehicle_info['appearances'].append(appearance)
            vehicle_info['last_seen'] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
            vehicle_info['cameras_seen'].add(camera_id)
            vehicle_info['total_appearances'] += 1

            # Update representative features (weighted average)
            alpha = 0.7  # Weight for existing features
            vehicle_info['representative_features'] = (
                    alpha * vehicle_info['representative_features'] +
                    (1 - alpha) * features
            )

        return global_id

    def process_single_video(self, video_path, camera_id, output_dir):
        """Process single camera video"""
        print(f"\nProcessing camera {camera_id}: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output directories
        camera_output_dir = os.path.join(output_dir, f'camera_{camera_id}')
        crops_dir = os.path.join(camera_output_dir, 'crops')
        os.makedirs(crops_dir, exist_ok=True)

        frame_idx = 0
        detections_data = []

        # Simulate timestamp (you can modify this based on actual video timestamp)
        start_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            current_time = start_time + timedelta(seconds=frame_idx / fps)

            # Detect vehicles in frame
            detections = self.detect_vehicles_in_frame(frame, camera_id, current_time)

            for det_idx, detection in enumerate(detections):
                # Extract vehicle crop
                x1, y1, x2, y2 = detection['bbox']
                vehicle_crop = frame[y1:y2, x1:x2]

                if vehicle_crop.size > 0:
                    # Extract features
                    features = self.extract_vehicle_features(vehicle_crop, camera_id)

                    # Find matching vehicle or create new one
                    match_id, similarity = self.find_matching_vehicle(
                        detection, features, camera_id, current_time
                    )

                    global_id = self.add_or_update_vehicle(
                        detection, features, camera_id, current_time, frame_idx, match_id
                    )

                    # Save vehicle crop
                    crop_filename = f'vehicle_{global_id}_frame_{frame_idx}_det_{det_idx}.jpg'
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, vehicle_crop)

                    # Record detection data
                    
                    detection_record = {
                        'global_id': int(global_id),
                        'camera_id': str(camera_id),
                        'frame_idx': int(frame_idx),
                        'timestamp': current_time.isoformat(),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(detection['confidence']),
                        'class': str(detection['class']),
                        'crop_path': str(crop_path),
                        'match_similarity': float(similarity) if match_id else 1.0
                    }
                    detections_data.append(detection_record)
    

                    if match_id:
                        print(f"Frame {frame_idx}: Matched vehicle {global_id} (similarity: {similarity:.3f})")
                    else:
                        print(f"Frame {frame_idx}: New vehicle {global_id}")

            frame_idx += 1

            # Progress update
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()

        # Save camera-specific results
        camera_results = {
            'camera_id': camera_id,
            'video_path': video_path,
            'total_frames': frame_idx,
            'detections': detections_data
        }

        results_path = os.path.join(camera_output_dir, 'detections.json')
        with open(results_path, 'w') as f:
            json.dump(camera_results, f, indent=2, default=_json_default)

        print(f"Camera {camera_id} processed: {len(detections_data)} detections")
        return camera_results

    def process_multi_camera_dataset(self, dataset_config, output_dir):
        """
        Process multiple camera videos

        dataset_config format:
        {
            "cameras": {
                "cam_001": {
                    "video_path": "path/to/video1.mp4",
                    "location": {"x": 100, "y": 200},
                    "start_time": "2024-01-01T10:00:00"
                },
                "cam_002": {
                    "video_path": "path/to/video2.mp4",
                    "location": {"x": 500, "y": 300},
                    "start_time": "2024-01-01T10:05:00"
                }
            }
        }
        """

        print("=== MULTI-CAMERA VEHICLE RE-IDENTIFICATION ===")
        os.makedirs(output_dir, exist_ok=True)

        # Load dataset configuration
        with open(dataset_config, 'r') as f:
            config = json.load(f)

        self.cameras = config.get('cameras', {})
        # Build mapping camera_id (string) -> numeric label
        self.cam_id_to_label = {str(cid): idx for idx, cid in enumerate(sorted(self.cameras.keys()))}
        all_results = []

        # Process each camera sequentially
        for camera_id, camera_info in self.cameras.items():
            video_path = camera_info['video_path']

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            # Process single camera
            camera_results = self.process_single_video(video_path, camera_id, output_dir)
            if camera_results:
                all_results.append(camera_results)

        # Generate cross-camera analysis
        cross_camera_analysis = self.analyze_cross_camera_tracks(output_dir)

        # Save complete results
        complete_results = {
            'dataset_info': {
                'total_cameras': len(self.cameras),
                'processed_cameras': len(all_results),
                'total_vehicles': len(self.vehicle_database),
                'processing_time': datetime.now().isoformat()
            },
            'cameras': all_results,
            'cross_camera_analysis': cross_camera_analysis,
            'vehicle_database_summary': self.get_database_summary()
        }

        # Save to files
        results_path = os.path.join(output_dir, 'multi_camera_results.json')
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=_json_default)

        # Save vehicle database
        db_path = os.path.join(output_dir, 'vehicle_database.pkl')
        with open(db_path, 'wb') as f:
            pickle.dump(self.vehicle_database, f)

        # Generate reports
        self.generate_reports(output_dir)

        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total vehicles identified: {len(self.vehicle_database)}")
        print(f"Results saved to: {output_dir}")

        return complete_results

    def analyze_cross_camera_tracks(self, output_dir):
        """Analyze vehicle tracks across multiple cameras"""

        cross_camera_vehicles = []
        single_camera_vehicles = []

        for global_id, vehicle_info in self.vehicle_database.items():
            cameras_seen = list(vehicle_info['cameras_seen'])

            if len(cameras_seen) > 1:
                # Vehicle seen in multiple cameras
                track_info = {
                    'global_id': global_id,
                    'vehicle_class': vehicle_info['vehicle_class'],
                    'cameras': cameras_seen,
                    'num_cameras': len(cameras_seen),
                    'total_appearances': vehicle_info['total_appearances'],
                    'first_seen': vehicle_info['first_seen'],
                    'last_seen': vehicle_info['last_seen'],
                    'track_duration': self.calculate_track_duration(vehicle_info),
                    'camera_transitions': self.analyze_camera_transitions(vehicle_info)
                }
                cross_camera_vehicles.append(track_info)
            else:
                single_camera_vehicles.append({
                    'global_id': global_id,
                    'camera': cameras_seen[0],
                    'appearances': vehicle_info['total_appearances']
                })

        analysis = {
            'cross_camera_vehicles': cross_camera_vehicles,
            'single_camera_vehicles': single_camera_vehicles,
            'statistics': {
                'total_cross_camera_tracks': len(cross_camera_vehicles),
                'total_single_camera_tracks': len(single_camera_vehicles),
                'cross_camera_ratio': len(cross_camera_vehicles) / len(
                    self.vehicle_database) if self.vehicle_database else 0
            }
        }

        return analysis

    def calculate_track_duration(self, vehicle_info):
        """Calculate total tracking duration for a vehicle"""
        first_time = datetime.fromisoformat(vehicle_info['first_seen'])
        last_time = datetime.fromisoformat(vehicle_info['last_seen'])
        return (last_time - first_time).total_seconds()

    def analyze_camera_transitions(self, vehicle_info):
        """Analyze transitions between cameras"""
        appearances = sorted(vehicle_info['appearances'],
                             key=lambda x: datetime.fromisoformat(x['timestamp']))

        transitions = []
        for i in range(1, len(appearances)):
            prev_app = appearances[i - 1]
            curr_app = appearances[i]

            if prev_app['camera_id'] != curr_app['camera_id']:
                transition_time = (
                        datetime.fromisoformat(curr_app['timestamp']) -
                        datetime.fromisoformat(prev_app['timestamp'])
                ).total_seconds()

                transitions.append({
                    'from_camera': prev_app['camera_id'],
                    'to_camera': curr_app['camera_id'],
                    'transition_time': transition_time,
                    'from_timestamp': prev_app['timestamp'],
                    'to_timestamp': curr_app['timestamp']
                })

        return transitions

    def get_database_summary(self):
        """Get summary statistics of vehicle database"""
        if not self.vehicle_database:
            return {}

        # Class distribution
        class_counts = defaultdict(int)
        camera_counts = defaultdict(int)
        appearance_counts = []

        for vehicle_info in self.vehicle_database.values():
            class_counts[vehicle_info['vehicle_class']] += 1
            appearance_counts.append(vehicle_info['total_appearances'])

            for camera_id in vehicle_info['cameras_seen']:
                camera_counts[camera_id] += 1

        return {
            'total_vehicles': len(self.vehicle_database),
            'class_distribution': dict(class_counts),
            'camera_distribution': dict(camera_counts),
            'appearance_statistics': {
                'mean_appearances': np.mean(appearance_counts),
                'max_appearances': max(appearance_counts),
                'min_appearances': min(appearance_counts)
            }
        }

    def generate_reports(self, output_dir):
        """Generate detailed reports and visualizations"""

        # 1. Cross-camera tracking report
        cross_camera_df_data = []
        for global_id, vehicle_info in self.vehicle_database.items():
            if len(vehicle_info['cameras_seen']) > 1:
                row = {
                    'Vehicle_ID': global_id,
                    'Class': vehicle_info['vehicle_class'],
                    'Cameras': ','.join(vehicle_info['cameras_seen']),
                    'Total_Appearances': vehicle_info['total_appearances'],
                    'First_Seen': vehicle_info['first_seen'],
                    'Last_Seen': vehicle_info['last_seen'],
                    'Track_Duration_Seconds': self.calculate_track_duration(vehicle_info)
                }
                cross_camera_df_data.append(row)

        if cross_camera_df_data:
            df = pd.DataFrame(cross_camera_df_data)
            csv_path = os.path.join(output_dir, 'cross_camera_tracks.csv')
            df.to_csv(csv_path, index=False)
            print(f"Cross-camera tracking report saved to: {csv_path}")

        # 2. Camera activity report
        camera_activity = defaultdict(lambda: {'total_detections': 0, 'unique_vehicles': set()})

        for global_id, vehicle_info in self.vehicle_database.items():
            for appearance in vehicle_info['appearances']:
                camera_id = appearance['camera_id']
                camera_activity[camera_id]['total_detections'] += 1
                camera_activity[camera_id]['unique_vehicles'].add(global_id)

        camera_df_data = []
        for camera_id, activity in camera_activity.items():
            row = {
                'Camera_ID': camera_id,
                'Total_Detections': activity['total_detections'],
                'Unique_Vehicles': len(activity['unique_vehicles'])
            }
            camera_df_data.append(row)

        camera_df = pd.DataFrame(camera_df_data)
        camera_csv_path = os.path.join(output_dir, 'camera_activity.csv')
        camera_df.to_csv(camera_csv_path, index=False)
        print(f"Camera activity report saved to: {camera_csv_path}")

    def query_vehicle_trajectory(self, global_id, output_dir):
        """Query complete trajectory of a specific vehicle"""

        if global_id not in self.vehicle_database:
            print(f"Vehicle {global_id} not found in database")
            return None

        vehicle_info = self.vehicle_database[global_id]

        # Sort appearances by timestamp
        appearances = sorted(vehicle_info['appearances'],
                             key=lambda x: datetime.fromisoformat(x['timestamp']))

        trajectory_data = {
            'global_id': global_id,
            'vehicle_class': vehicle_info['vehicle_class'],
            'summary': {
                'total_appearances': len(appearances),
                'cameras_visited': list(vehicle_info['cameras_seen']),
                'first_seen': vehicle_info['first_seen'],
                'last_seen': vehicle_info['last_seen'],
                'total_duration': self.calculate_track_duration(vehicle_info)
            },
            'trajectory': appearances
        }

        # Save trajectory
        trajectory_path = os.path.join(output_dir, f'vehicle_{global_id}_trajectory.json')
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"Vehicle {global_id} trajectory saved to: {trajectory_path}")
        return trajectory_data


def create_sample_dataset_config():
    """Create sample dataset configuration file"""

    sample_config = {
        "dataset_name": "Sample Multi-Camera Dataset",
        "description": "Sample configuration for multi-camera vehicle tracking",
        "cameras": {
            "cam_001": {
                "video_path": "videos/camera_001.mp4",
                "location": {"x": 0, "y": 0, "description": "Main street entrance"},
                "start_time": "2024-01-01T08:00:00",
                "fps": 30
            },
            "cam_002": {
                "video_path": "videos/camera_002.mp4",
                "location": {"x": 500, "y": 100, "description": "Main street middle"},
                "start_time": "2024-01-01T08:00:00",
                "fps": 30
            },
            "cam_003": {
                "video_path": "videos/camera_003.mp4",
                "location": {"x": 1000, "y": 200, "description": "Main street exit"},
                "start_time": "2024-01-01T08:00:00",
                "fps": 25
            }
        }
    }

    config_path = 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"Sample dataset config created: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Vehicle Re-Identification")
    parser.add_argument("--config_file", required=True, help="ReID model config file")
    parser.add_argument("--model_weight", required=True, help="ReID model weights")
    parser.add_argument("--dataset_config", help="Dataset configuration JSON file")
    parser.add_argument("--output_dir", default="./multi_camera_results", help="Output directory")
    parser.add_argument("--mode", choices=['process', 'query', 'setup'], default='setup',
                        help="Operation mode")
    parser.add_argument("--vehicle_id", type=int, help="Vehicle ID for trajectory query")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                        help="Similarity threshold for matching")

    args = parser.parse_args()

    if args.mode == 'setup':
        print("Setting up Multi-Camera Vehicle ReID...")
        config_path = create_sample_dataset_config()
        print(f"\nSample dataset config created: {config_path}")
        print("\nNext steps:")
        print("1. Update dataset_config.json with your actual video paths")
        print(
            "2. Run: python multi_camera_vehicle_reid.py --mode process --config_file your_config.yaml --model_weight your_model.pth --dataset_config dataset_config.json")
        return

    # Initialize system
    reid_system = MultiCameraVehicleReID(args.config_file, args.model_weight)
    reid_system.similarity_threshold = args.similarity_threshold

    if args.mode == 'process':
        if not args.dataset_config:
            print("Error: dataset_config is required for process mode")
            return

        # Process multi-camera dataset
        results = reid_system.process_multi_camera_dataset(args.dataset_config, args.output_dir)

        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total vehicles identified: {results['dataset_info']['total_vehicles']}")
        print(f"Cross-camera tracks: {results['cross_camera_analysis']['statistics']['total_cross_camera_tracks']}")
        print(f"Single-camera tracks: {results['cross_camera_analysis']['statistics']['total_single_camera_tracks']}")

    elif args.mode == 'query':
        if args.vehicle_id is None:
            print("Error: vehicle_id is required for query mode")
            return

        # Load existing database
        db_path = os.path.join(args.output_dir, 'vehicle_database.pkl')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                reid_system.vehicle_database = pickle.load(f)

        # Query vehicle trajectory
        trajectory = reid_system.query_vehicle_trajectory(args.vehicle_id, args.output_dir)

        if trajectory:
            print(f"\nVehicle {args.vehicle_id} Summary:")
            print(f"Class: {trajectory['vehicle_class']}")
            print(f"Total appearances: {trajectory['summary']['total_appearances']}")
            print(f"Cameras visited: {trajectory['summary']['cameras_visited']}")
            print(f"Duration: {trajectory['summary']['total_duration']:.1f} seconds")


if __name__ == "__main__":
    main()