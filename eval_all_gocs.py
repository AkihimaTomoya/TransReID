import os
import argparse
from glob import glob

import torch
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference


def load_weights_filtered(model, weight_path: str):
    """Nạp checkpoint an toàn: bỏ 'module.', chỉ copy tensor trùng tên & trùng shape."""
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(f"[load_weights_filtered] Not found: {weight_path}")
    state = torch.load(weight_path, map_location='cpu')
    sd = state.get('model_state_dict', state.get('state_dict', state))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    model_sd = model.state_dict()
    ok = {k: v for k, v in sd.items() if (k in model_sd and model_sd[k].shape == v.shape)}
    model_sd.update(ok)
    model.load_state_dict(model_sd, strict=False)
    print(f"[load_weights_filtered] loaded {len(ok)} tensors from {weight_path}, skipped={len(sd)-len(ok)}")


def run_one(cfg_file, root_dir, weight, query_ver, goc_name, device_id=0,
            ims_per_batch=32, num_workers=2, rerank=True, neck_feat='after'):
    """Chạy eval cho 1 gốc, trả về dict: {'goc', 'mAP', 'R1','R5','R10'}"""
    # 1) Cấu hình
    cfg.defrost()
    if cfg_file:
        cfg.merge_from_file(cfg_file)
    # override
    cfg.merge_from_list([
        'DATASETS.NAMES', 'realdata',
        'DATASETS.ROOT_DIR', root_dir,
        'TEST.WEIGHT', weight,
        'TEST.NECK_FEAT', neck_feat,
        'TEST.RE_RANKING', str(bool(rerank)),
        'TEST.IMS_PER_BATCH', str(int(ims_per_batch)),
        'DATALOADER.NUM_WORKERS', str(int(num_workers)),
        'MODEL.DEVICE_ID', f"('{device_id}')",
    ])
    cfg.freeze()

    # 2) Env để RealData split đúng gốc/phiên bản
    os.environ['REALDATA_QUERY_VER'] = query_ver
    os.environ['REALDATA_QUERY_GOC'] = goc_name

    # 3) Dataloader / Model
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    load_weights_filtered(model, cfg.TEST.WEIGHT)

    # 4) Inference (y/c processor.do_inference trả về mAP, R1, R5, R10)
    mAP, R1, R5, R10 = do_inference(cfg, model, val_loader, num_query)
    return {'goc': goc_name, 'mAP': mAP * 100.0, 'R1': R1 * 100.0, 'R5': R5 * 100.0, 'R10': R10 * 100.0}


def main():
    ap = argparse.ArgumentParser("Eval all goc_* and compute mean")
    ap.add_argument("--config_file", type=str, default="configs/VeRi/vit_base.yml")
    ap.add_argument("--root_dir", type=str, required=True, help="Folder cha chứa RealData/")
    ap.add_argument("--weight", type=str, required=True)
    ap.add_argument("--query_ver", type=str, default="v1", choices=["v1", "v2"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ims_per_batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_rerank", action="store_true")
    ap.add_argument("--neck_feat", type=str, default="after", choices=["before","after"])
    args = ap.parse_args()

    # liệt kê goc_*
    goc_pattern = os.path.join(args.root_dir, "RealData", "dataID", "goc_*")
    goc_dirs = sorted([d for d in glob(goc_pattern) if os.path.isdir(d)])
    if not goc_dirs:
        raise RuntimeError(f"Không tìm thấy thư mục theo mẫu: {goc_pattern}")
    goc_names = [os.path.basename(d) for d in goc_dirs]
    print("Found gocs:", goc_names)

    results = []
    for goc in goc_names:
        print(f"\n>>> Running {goc} (query={args.query_ver})")
        rec = run_one(
            cfg_file=args.config_file,
            root_dir=args.root_dir,
            weight=args.weight,
            query_ver=args.query_ver,
            goc_name=goc,
            device_id=args.device,
            ims_per_batch=args.ims_per_batch,
            num_workers=args.num_workers,
            rerank=(not args.no_rerank),
            neck_feat=args.neck_feat,
        )
        print(f"[{goc}] mAP={rec['mAP']:.2f}%  R1={rec['R1']:.2f}%  R5={rec['R5']:.2f}%  R10={rec['R10']:.2f}%")
        results.append(rec)

    # compute mean
    mean = lambda k: sum(r[k] for r in results) / len(results)
    m_map, m_r1, m_r5, m_r10 = mean('mAP'), mean('R1'), mean('R5'), mean('R10')

    print("\n===== Per-goc results =====")
    for r in results:
        print(f"{r['goc']:>10s} | mAP={r['mAP']:6.2f}%  R1={r['R1']:6.2f}%  R5={r['R5']:6.2f}%  R10={r['R10']:6.2f}%")
    print("\n===== MEAN over gocs =====")
    print(f"mAP={m_map:.2f}%  R1={m_r1:.2f}%  R5={m_r5:.2f}%  R10={m_r10:.2f}%")


if __name__ == "__main__":
    main()
