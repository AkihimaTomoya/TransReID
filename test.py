import os
import argparse
import torch  # <-- thêm
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


def load_weights_filtered(model, weight_path: str):
    """
    Nạp checkpoint 'an toàn' cho cross-dataset eval:
    - Tự nhận {model_state_dict}/{state_dict}/raw dict
    - Bỏ prefix 'module.'
    - Chỉ nạp các tensor trùng tên & trùng shape, skip head lệch shape (classifier/bnneck)
    """
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(f"[load_weights_filtered] Weight not found: {weight_path}")

    state = torch.load(weight_path, map_location='cpu')
    sd = state.get('model_state_dict', state.get('state_dict', state))
    # normalize keys
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    model_sd = model.state_dict()
    ok = {k: v for k, v in sd.items() if (k in model_sd and model_sd[k].shape == v.shape)}
    skip_shape = [(k, tuple(v.shape), tuple(model_sd[k].shape)) for k, v in sd.items()
                  if k in model_sd and model_sd[k].shape != v.shape]
    skip_missing = [k for k in sd.keys() if k not in model_sd]

    # update and load
    model_sd.update(ok)
    model.load_state_dict(model_sd, strict=False)

    print(f"[load_weights_filtered] loaded {len(ok)} tensors from {weight_path}, "
          f"skipped_shape={len(skip_shape)}, skipped_missing={len(skip_missing)}")
    for i, (k, s_src, s_dst) in enumerate(skip_shape[:12]):
        print(f"  - skip shape {i+1:02d}: {k}  ckpt{s_src} -> model{s_dst}")
    for i, k in enumerate(skip_missing[:6]):
        print(f"  - skip missing {i+1:02d}: {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    # model.load_param(cfg.TEST.WEIGHT)  # <-- bỏ
    load_weights_filtered(model, cfg.TEST.WEIGHT)  # <-- nạp an toàn

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg, model, val_loader, num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
        do_inference(cfg, model, val_loader, num_query)
