from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_checkpoint(model, optimizer, optimizer_center, scheduler, checkpoint_path, logger):
    """
    Load checkpoint and resume training state
    Returns: start_epoch
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return 1
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded successfully")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Model state loaded successfully")
        else:
            # If checkpoint only contains model weights (old format)
            model.load_state_dict(checkpoint)
            logger.info("Model weights loaded (old format)")
            return 1  # Cannot resume epoch info from old format
        
        start_epoch = 1
        
        # Load optimizer state
        if cfg.RESUME.RESUME_OPTIMIZER and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
            
        if cfg.RESUME.RESUME_OPTIMIZER and 'optimizer_center_state_dict' in checkpoint:
            optimizer_center.load_state_dict(checkpoint['optimizer_center_state_dict'])
            logger.info("Center optimizer state loaded successfully")
        
        # Load scheduler state
        if cfg.RESUME.RESUME_SCHEDULER and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded successfully")
        
        # Get start epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        
        return start_epoch
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 1

def save_checkpoint(model, optimizer, optimizer_center, scheduler, epoch, save_path, logger):
    """
    Save checkpoint with all training states
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_center_state_dict': optimizer_center.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved at epoch {epoch}: {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    # Check if we need to resume from checkpoint
    start_epoch = 1
    if cfg.RESUME.ENABLED and cfg.RESUME.CHECKPOINT_PATH:
        start_epoch = load_checkpoint(model, optimizer, optimizer_center, scheduler, 
                                    cfg.RESUME.CHECKPOINT_PATH, logger)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, 
        args.local_rank,
        start_epoch=start_epoch  # Pass start_epoch to do_train
    )
