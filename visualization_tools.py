#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless-safe Visualization Tools
- For environments without X/Qt (fixes: "Could not load the Qt platform plugin 'xcb'")
- Forces Matplotlib backend to 'Agg' before importing pyplot
Usage:
  python visualization_tools_headless.py --results_dir <processing_dir> --mode all
"""
import os
# Force a non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")

import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

try:
    import cv2  # optional; we don't call imshow so no Qt needed
except Exception:
    cv2 = None

class ReIDVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.vehicle_database = {}
        self._load()

    def _load(self):
        res_json = self.results_dir / "multi_camera_results.json"
        if res_json.exists():
            self.results = json.loads(res_json.read_text(encoding="utf-8"))
        db_pkl = self.results_dir / "vehicle_database.pkl"
        if db_pkl.exists():
            with db_pkl.open("rb") as f:
                self.vehicle_database = pickle.load(f)
        print(f"[viz] Loaded {len(self.vehicle_database)} vehicles from {self.results_dir}")

    # ---------- Plots ----------
    def plot_camera_network(self, out_path: Path):
        out_path = Path(out_path)
        G = nx.DiGraph()
        camera_vehicle_counts = {}

        for vid, vinfo in self.vehicle_database.items():
            if len(vinfo.get("cameras_seen", [])) > 1:
                appearances = sorted(vinfo["appearances"], key=lambda x: x["timestamp"])
                for i in range(1, len(appearances)):
                    a = appearances[i-1]; b = appearances[i]
                    c1, c2 = a["camera_id"], b["camera_id"]
                    if c1 != c2:
                        if G.has_edge(c1, c2):
                            G[c1][c2]["weight"] += 1
                        else:
                            G.add_edge(c1, c2, weight=1)
            for cid in vinfo.get("cameras_seen", []):
                camera_vehicle_counts[cid] = camera_vehicle_counts.get(cid, 0) + 1

        for cid in camera_vehicle_counts.keys():
            if cid not in G.nodes:
                G.add_node(cid)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, seed=42)
        node_sizes = [camera_vehicle_counts.get(n, 1) * 120 for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#9BD1F2", alpha=0.9)
        if G.edges:
            weights = [G[u][v]["weight"] for u,v in G.edges]
            maxw = max(weights) if weights else 1
            widths = [w / maxw * 5 for w in weights]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color="#555", arrows=True, arrowsize=16, alpha=0.6)
            nx.draw_networkx_edge_labels(G, pos, {(u,v): G[u][v]["weight"] for u,v in G.edges}, font_size=9)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")
        plt.title("Camera Network - Vehicle Movement", fontsize=14, fontweight="bold")
        plt.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[viz] camera network -> {out_path}")

    def plot_timeline(self, out_path: Path):
        out_path = Path(out_path)
        rows = []
        for vid, vinfo in self.vehicle_database.items():
            if len(vinfo.get("cameras_seen", [])) > 1:
                for ap in vinfo["appearances"]:
                    rows.append({
                        "vehicle_id": vid,
                        "camera_id": ap["camera_id"],
                        "timestamp": ap["timestamp"]
                    })
        if not rows:
            print("[viz] No cross-camera vehicles; timeline skipped")
            return
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["vehicle_id", "timestamp"])

        plt.figure(figsize=(15, 8))
        cameras = df["camera_id"].unique().tolist()
        cmap = plt.cm.get_cmap("Set3", len(cameras))
        color_map = {c: cmap(i) for i, c in enumerate(cameras)}

        y = 0
        for vid in df["vehicle_id"].unique():
            dd = df[df["vehicle_id"] == vid]
            plt.plot(dd["timestamp"], [y]*len(dd), linestyle="-", color="#222", alpha=0.25, linewidth=1)
            plt.scatter(dd["timestamp"], [y]*len(dd), s=90, alpha=0.85,
                        c=[color_map[c] for c in dd["camera_id"]])
            plt.text(dd["timestamp"].iloc[0], y+0.15, f"V{vid}", fontsize=8)
            y += 1

        handles = [plt.Line2D([0],[0], marker="o", linestyle="", label=c, markerfacecolor=color_map[c], markeredgecolor=color_map[c]) for c in cameras]
        plt.legend(handles=handles, title="Cameras", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.title("Vehicle Timeline - Cross-Camera Appearances", fontsize=14, fontweight="bold")
        plt.xlabel("Time"); plt.ylabel("Vehicle (row index)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[viz] timeline -> {out_path}")

    def plot_stats(self, out_path: Path):
        out_path = Path(out_path)
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Multi-Camera Vehicle ReID - Statistics", fontsize=16, fontweight="bold")

        # (1) class distribution
        cls_counts = {}
        for v in self.vehicle_database.values():
            cls = v.get("vehicle_class", "unknown")
            cls_counts[cls] = cls_counts.get(cls, 0) + 1
        axes[0,0].pie(cls_counts.values(), labels=cls_counts.keys(), autopct="%1.1f%%")
        axes[0,0].set_title("Vehicle Class Distribution")

        # (2) vehicles per camera
        cam_counts = {}
        for v in self.vehicle_database.values():
            for cid in v.get("cameras_seen", []):
                cam_counts[cid] = cam_counts.get(cid, 0) + 1
        axes[0,1].bar(list(cam_counts.keys()), list(cam_counts.values()))
        axes[0,1].set_title("Vehicles per Camera")
        axes[0,1].tick_params(axis="x", rotation=45)

        # (3) cross vs single
        cross = sum(1 for v in self.vehicle_database.values() if len(v.get("cameras_seen", [])) > 1)
        single = len(self.vehicle_database) - cross
        axes[0,2].pie([cross, single], labels=["Cross-camera", "Single-camera"], autopct="%1.1f%%")
        axes[0,2].set_title("Track Distribution")

        # (4) appearances histogram
        app_counts = [len(v.get("appearances", [])) for v in self.vehicle_database.values()]
        axes[1,0].hist(app_counts, bins=20, alpha=0.8)
        axes[1,0].set_title("Appearance Frequency")
        axes[1,0].set_xlabel("#Appearances"); axes[1,0].set_ylabel("#Vehicles")

        # (5) durations for cross-camera
        durations = []
        for v in self.vehicle_database.values():
            if len(v.get("cameras_seen", [])) > 1 and v.get("first_seen") and v.get("last_seen"):
                try:
                    t1 = pd.to_datetime(v["first_seen"])
                    t2 = pd.to_datetime(v["last_seen"])
                    durations.append((t2 - t1).total_seconds()/60.0)
                except Exception:
                    pass
        if durations:
            axes[1,1].hist(durations, bins=15, alpha=0.8)
            axes[1,1].set_title("Track Duration (minutes)")
        else:
            axes[1,1].text(0.5,0.5,"No cross-camera\ntracks", ha="center", va="center", transform=axes[1,1].transAxes)
            axes[1,1].set_title("Track Duration")

        # (6) summary text
        import numpy as _np
        stats_txt = f"""Total Vehicles: {len(self.vehicle_database)}
Cross-camera: {cross}
Single-camera: {single}
Total Cameras: {len(cam_counts)}
Avg Appearances/Vehicle: {(_np.mean(app_counts) if app_counts else 0):.1f}
Max Appearances: {max(app_counts) if app_counts else 0}
"""
        axes[1,2].axis("off")
        axes[1,2].text(0.02, 0.98, stats_txt, ha="left", va="top", fontsize=12, family="monospace")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[viz] stats -> {out_path}")

    def write_report(self, out_html: Path):
        out_html = Path(out_html)
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Multi-Camera Vehicle ReID - Report</title>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; margin: 24px; }}
    h1,h2 {{ margin: 0 0 10px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    .imgbox {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #fafafa; }}
    img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Multi-Camera Vehicle ReID - Report</h1>
  <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <div class="grid">
    <div class="imgbox">
      <h2>Statistics Dashboard</h2>
      <img src="statistics_dashboard.png" />
    </div>
    <div class="imgbox">
      <h2>Camera Network</h2>
      <img src="camera_network.png" />
    </div>
    <div class="imgbox">
      <h2>Vehicle Timeline</h2>
      <img src="vehicle_timeline.png" />
    </div>
  </div>
</body>
</html>
        """.strip()
        out_html.write_text(html, encoding="utf-8")
        print(f"[viz] report -> {out_html}")

def main():
    ap = argparse.ArgumentParser(description="Visualization tools (headless-safe)")
    ap.add_argument("--results_dir", required=True, help="Directory that contains multi_camera_results.json + vehicle_database.pkl")
    ap.add_argument("--mode", choices=["all", "network", "timeline", "stats", "report"], default="all")
    ap.add_argument("--output_dir", help="Directory to write images/report (default = results_dir)")
    args = ap.parse_args()

    viz = ReIDVisualizer(args.results_dir)
    if not viz.vehicle_database:
        print("[viz] No vehicle database found. Aborting.")
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else viz.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("all", "network"):
        viz.plot_camera_network(out_dir / "camera_network.png")
    if args.mode in ("all", "timeline"):
        viz.plot_timeline(out_dir / "vehicle_timeline.png")
    if args.mode in ("all", "stats"):
        viz.plot_stats(out_dir / "statistics_dashboard.png")
    if args.mode in ("all", "report"):
        viz.write_report(out_dir / "report.html")
    print("[viz] done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
