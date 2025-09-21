#!/usr/bin/env python3
"""
Test extraction cho ReID scenario thực tế
"""

import re
from collections import defaultdict


def extract_vehicle_id(filename):
    """Extract vehicle ID (view ID) từ filename"""
    filename_lower = filename.lower()

    # Pattern goc*_v* - v* là vehicle identity
    if 'goc' in filename_lower and '_v' in filename_lower:
        match = re.search(r'_v(\d+)', filename_lower)
        if match:
            return int(match.group(1))

    return None


def extract_camera_id(filename):
    """Extract camera ID từ filename"""
    filename_lower = filename.lower()

    # Extract camera từ goc*
    match = re.search(r'goc[_]?(\d+)', filename_lower)
    if match:
        return int(match.group(1))

    return None


# Test với data thực tế của bạn
actual_files = [
    "goc1_v1.MOV",  # Vehicle 1, Camera 1
    "goc_2_v1.MOV",  # Vehicle 1, Camera 2
    "goc_3_v1.MOV",  # Vehicle 1, Camera 3
    "goc4_v1.MOV",  # Vehicle 1, Camera 4
    "VID_20241030_150638.mp4"  # Skip
]

# Giả lập thêm data để test ReID
simulated_files = [
    "goc1_v1.MOV",  # Vehicle 1, Camera 1
    "goc_2_v1.MOV",  # Vehicle 1, Camera 2
    "goc_3_v1.MOV",  # Vehicle 1, Camera 3
    "goc4_v1.MOV",  # Vehicle 1, Camera 4
    "goc1_v2.MOV",  # Vehicle 2, Camera 1
    "goc_2_v2.MOV",  # Vehicle 2, Camera 2
    "goc_3_v3.MOV",  # Vehicle 3, Camera 3
    "goc4_v3.MOV",  # Vehicle 3, Camera 4
    "goc1_v4.MOV",  # Vehicle 4, Camera 1
]

print("ReID Extraction Test")
print("=" * 60)

print("\n1. Test với files thực tế của bạn:")
for filename in actual_files:
    vid = extract_vehicle_id(filename)
    cam = extract_camera_id(filename)
    print(f"    {filename:<30} -> Vehicle: {vid}, Camera: {cam}")

print(f"\n2. Test với simulated data (để hiểu ReID scenario):")
vehicle_cameras = defaultdict(list)

for filename in simulated_files:
    vid = extract_vehicle_id(filename)
    cam = extract_camera_id(filename)
    if vid is not None and cam is not None:
        vehicle_cameras[vid].append((filename, cam))
        print(f"    {filename:<20} -> Vehicle: {vid}, Camera: {cam}")

print(f"\n3. Query/Gallery Split Analysis:")
print("-" * 40)

query_count = 0
gallery_count = 0
valid_vehicles = 0

for vid, cam_list in vehicle_cameras.items():
    if len(cam_list) >= 2:  # Cần ít nhất 2 cameras để evaluate
        valid_vehicles += 1
        query_count += 1  # 1 query per vehicle
        gallery_count += len(cam_list) - 1  # Còn lại làm gallery

        print(f"Vehicle {vid}: {len(cam_list)} cameras")
        for filename, cam in sorted(cam_list, key=lambda x: x[1]):
            print(f"    Camera {cam}: {filename}")
    else:
        print(f"Vehicle {vid}: {len(cam_list)} cameras (SKIP - không đủ để evaluate)")

print(f"\n4. ReID Evaluation Potential:")
print(f"   Valid vehicles: {valid_vehicles}")
print(f"   Query videos: {query_count}")
print(f"   Gallery videos: {gallery_count}")
print(f"   Total videos for evaluation: {query_count + gallery_count}")

if valid_vehicles >= 2:
    print(f"   ✅ Dataset suitable for ReID evaluation")
else:
    print(f"   ❌ Dataset needs more vehicles with multiple camera views")

print(f"\n5. ReID Scenario Explanation:")
print("   - Query: 1 video per vehicle (e.g., goc1_v1.MOV)")
print("   - Gallery: remaining videos of same vehicle (goc_2_v1.MOV, goc_3_v1.MOV, goc4_v1.MOV)")
print("   - Goal: Model should rank gallery videos of same vehicle higher than other vehicles")
print("   - Success: If querying Vehicle 1 from Camera 1, videos of Vehicle 1 from other cameras")
print("            should appear at top of ranking (Rank-1, Rank-5, mAP metrics)")