REALDATA_QUERY_VER=v1 \
python test.py \
  --config_file configs/VeRi/vit_base.yml \
  DATASETS.NAMES realdata \
  DATASETS.ROOT_DIR ./ \
  TEST.WEIGHT transformer_120.pth \
  OUTPUT_DIR logs/test_realdata_v2q \
  MODEL.DEVICE_ID "('0')"
