#!/usr/bin/env python3
"""
Complete Workflow Script for Multi-Camera Vehicle ReID
Script tổng hợp chạy toàn bộ pipeline từ A-Z
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


class MultiCameraWorkflow:
    def __init__(self, dataset_root, model_config, model_weight, output_root):
        self.dataset_root = Path(dataset_root)
        self.model_config = model_config
        self.model_weight = model_weight
        self.output_root = Path(output_root)

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Multi-Camera Vehicle ReID Workflow")
        print(f"📁 Dataset: {self.dataset_root}")
        print(f"🤖 Model: {self.model_weight}")
        print(f"📊 Output: {self.run_dir}")
        print("=" * 60)

    def step_1_scan_dataset(self):
        """Step 1: Scan và validate dataset"""
        print("\n📋 Step 1: Scanning Dataset...")

        dataset_config = self.run_dir / "dataset_config.json"

        # Scan dataset directory
        cmd = [
            sys.executable, "dataset_processor.py",
            "--mode", "scan",
            "--dataset_root", str(self.dataset_root),
            "--output_config", str(dataset_config)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Dataset scan failed: {result.stderr}")
            return False

        print(f"✅ Dataset config created: {dataset_config}")

        # Validate dataset
        print("\n🔍 Validating dataset...")
        cmd = [
            sys.executable, "dataset_processor.py",
            "--mode", "validate",
            "--config_file", str(dataset_config)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Dataset validation failed: {result.stderr}")
            return False

        print("✅ Dataset validation passed")
        self.dataset_config = dataset_config
        return True

    def step_2_process_videos(self):
        """Step 2: Process multi-camera videos"""
        print("\n🎥 Step 2: Processing Multi-Camera Videos...")

        processing_dir = self.run_dir / "processing"

        cmd = [
            sys.executable, "multi_camera_vehicle_reid.py",
            "--mode", "process",
            "--config_file", self.model_config,
            "--model_weight", self.model_weight,
            "--dataset_config", str(self.dataset_config),
            "--output_dir", str(processing_dir),
            "--similarity_threshold", "0.7"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("❌ Video processing failed")
            return False

        print("✅ Video processing completed")
        self.processing_dir = processing_dir
        return True

    def step_3_generate_visualizations(self):
        """Step 3: Generate visualizations"""
        print("\n📊 Step 3: Generating Visualizations...")

        viz_dir = self.run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "visualization_tools.py",
            "--results_dir", str(self.processing_dir),
            "--mode", "all",
            "--output_dir", str(viz_dir)
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("❌ Visualization generation failed")
            return False

        print("✅ Visualizations generated")
        self.viz_dir = viz_dir
        return True

    def step_4_generate_reports(self):
        """Step 4: Generate final reports"""
        print("\n📋 Step 4: Generating Final Reports...")

        # Generate processing summary
        summary_file = self.run_dir / "processing_summary.csv"
        cmd = [
            sys.executable, "dataset_processor.py",
            "--mode", "summary",
            "--config_file", str(self.dataset_config),
            "--output_summary", str(summary_file)
        ]

        subprocess.run(cmd)

        # Copy important results to main directory
        import shutil

        # Copy main results
        src_results = self.processing_dir / "multi_camera_results.json"
        if src_results.exists():
            shutil.copy(src_results, self.run_dir / "results.json")

        # Copy vehicle database
        src_db = self.processing_dir / "vehicle_database.pkl"
        if src_db.exists():
            shutil.copy(src_db, self.run_dir / "vehicle_database.pkl")

        # Copy HTML report
        src_report = self.viz_dir / "report.html"
        if src_report.exists():
            shutil.copy(src_report, self.run_dir / "report.html")

            # Copy visualization images
            for img_file in self.viz_dir.glob("*.png"):
                shutil.copy(img_file, self.run_dir / img_file.name)

        print("✅ Reports generated")
        return True

    def step_5_summary(self):
        """Step 5: Print summary"""
        print("\n📈 Step 5: Final Summary")
        print("=" * 60)

        # Load and print results summary
        results_file = self.run_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            dataset_info = results.get('dataset_info', {})
            cross_camera = results.get('cross_camera_analysis', {}).get('statistics', {})

            print(f"🎯 Processing Results:")
            print(f"   • Total Vehicles: {dataset_info.get('total_vehicles', 0)}")
            print(f"   • Cross-Camera Tracks: {cross_camera.get('total_cross_camera_tracks', 0)}")
            print(f"   • Single-Camera Tracks: {cross_camera.get('total_single_camera_tracks', 0)}")
            print(f"   • Cameras Processed: {dataset_info.get('processed_cameras', 0)}")

            cross_camera_ratio = cross_camera.get('cross_camera_ratio', 0)
            print(f"   • Cross-Camera Ratio: {cross_camera_ratio:.1%}")

        print(f"\n📁 Output Files:")
        print(f"   • Main Directory: {self.run_dir}")
        print(f"   • HTML Report: {self.run_dir}/report.html")
        print(f"   • Results JSON: {self.run_dir}/results.json")
        print(f"   • Vehicle Database: {self.run_dir}/vehicle_database.pkl")
        print(f"   • Processing Summary: {self.run_dir}/processing_summary.csv")

        # List visualization files
        viz_files = list(self.run_dir.glob("*.png"))
        if viz_files:
            print(f"   • Visualizations: {len(viz_files)} PNG files")

        print(f"\n🌐 To view results, open: {self.run_dir}/report.html")
        print("=" * 60)
        print("✅ Workflow Complete! 🎉")

    def run_complete_workflow(self):
        """Run complete workflow"""
        try:
            # Step 1: Dataset scanning
            if not self.step_1_scan_dataset():
                return False

            # Step 2: Video processing
            if not self.step_2_process_videos():
                return False

            # Step 3: Visualizations
            if not self.step_3_generate_visualizations():
                return False

            # Step 4: Reports
            if not self.step_4_generate_reports():
                return False

            # Step 5: Summary
            self.step_5_summary()

            return True

        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            return False

    def run_interactive_mode(self):
        """Interactive mode with step-by-step confirmation"""
        print("🎮 Interactive Mode - You can run each step individually")

        steps = [
            ("Scan Dataset", self.step_1_scan_dataset),
            ("Process Videos", self.step_2_process_videos),
            ("Generate Visualizations", self.step_3_generate_visualizations),
            ("Generate Reports", self.step_4_generate_reports),
            ("Show Summary", self.step_5_summary)
        ]

        for i, (step_name, step_func) in enumerate(steps):
            print(f"\n{'=' * 60}")
            print(f"Step {i + 1}: {step_name}")

            while True:
                choice = input(f"Run {step_name}? [y/n/q]: ").lower().strip()
                if choice == 'q':
                    print("Workflow terminated by user")
                    return False
                elif choice == 'y':
                    success = step_func()
                    if not success and i < len(steps) - 1:  # Last step is summary, always continue
                        print(f"Step {i + 1} failed. Continue anyway? [y/n]: ", end="")
                        if input().lower().strip() != 'y':
                            return False
                    break
                elif choice == 'n':
                    print(f"Skipping {step_name}")
                    break
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")

        return True


def create_sample_directory_structure():
    """Create sample directory structure for testing"""
    sample_dir = Path("sample_dataset")
    sample_dir.mkdir(exist_ok=True)

    # Create sample structure
    (sample_dir / "camera_001").mkdir(exist_ok=True)
    (sample_dir / "camera_002").mkdir(exist_ok=True)
    (sample_dir / "camera_003").mkdir(exist_ok=True)

    # Create sample metadata
    sample_metadata = {
        "location": {"x": 0, "y": 0, "description": "Street entrance"},
        "start_time": "2024-01-01T08:00:00",
        "fps": 30
    }

    for i in range(1, 4):
        metadata_file = sample_dir / f"camera_{i:03d}" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f, indent=2)

    print(f"📁 Sample directory structure created: {sample_dir}")
    print("Place your video files in the camera_xxx folders and run the workflow")

    return sample_dir


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Vehicle ReID Complete Workflow")
    parser.add_argument("--dataset_root", help="Root directory containing camera videos")
    parser.add_argument("--model_config", required=True, help="ReID model config file")
    parser.add_argument("--model_weight", required=True, help="ReID model weights")
    parser.add_argument("--output_root", default="./workflow_outputs", help="Output root directory")
    parser.add_argument("--mode", choices=['auto', 'interactive', 'setup'], default='auto',
                        help="Workflow mode")

    args = parser.parse_args()

    if args.mode == 'setup':
        print("🛠️  Setting up sample directory structure...")
        sample_dir = create_sample_directory_structure()
        print(f"\nSample structure created in: {sample_dir}")
        print("\nNext steps:")
        print(f"1. Place your video files in the camera folders")
        print(
            f"2. Run: python run_complete_workflow.py --dataset_root {sample_dir} --model_config your_config.yaml --model_weight your_model.pth")
        return

    if not args.dataset_root:
        print("Error: dataset_root is required for auto and interactive modes")
        print("Run with --mode setup to create sample structure first")
        return

    # Check required files
    if not Path(args.model_config).exists():
        print(f"Error: Model config file not found: {args.model_config}")
        return

    if not Path(args.model_weight).exists():
        print(f"Error: Model weight file not found: {args.model_weight}")
        return

    if not Path(args.dataset_root).exists():
        print(f"Error: Dataset root directory not found: {args.dataset_root}")
        return

    # Initialize workflow
    workflow = MultiCameraWorkflow(
        args.dataset_root,
        args.model_config,
        args.model_weight,
        args.output_root
    )

    # Run workflow
    if args.mode == 'auto':
        success = workflow.run_complete_workflow()
    elif args.mode == 'interactive':
        success = workflow.run_interactive_mode()

    if success:
        print("\n🎉 Workflow completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Workflow failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()