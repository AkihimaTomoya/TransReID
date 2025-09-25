python eval_all.py \
  --config_file configs/VeRi/vit_base.yml \
  --root_dir ./ \
  --weight transformer_120.pth \
  --query_ver v1 \
  --device 0 \
  --ims_per_batch 32 \
  --num_workers 2 \
  --neck_feat before \
  --tta_flip
