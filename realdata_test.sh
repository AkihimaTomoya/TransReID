python eval_all_gocs.py \
  --config_file configs/VeRi/vit_base.yml \
  --root_dir /kaggle/working \
  --weight /kaggle/input/temp/pytorch/default/1/transformer_120.pth \
  --query_ver v1 \
  --device 0 \
  --ims_per_batch 32 \
  --num_workers 2 \
  --neck_feat after
