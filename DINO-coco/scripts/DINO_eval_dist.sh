
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --output_dir logs/DINO/R50-MS4-%j \
	-c config/DINO/DINO_4scale.py  --coco_path ../dataset/coco/  \
	--eval --resume '../Frozen-DETR-ckpt/dino_4scale_r50_1x_coco_checkpoint0011.pth' \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
