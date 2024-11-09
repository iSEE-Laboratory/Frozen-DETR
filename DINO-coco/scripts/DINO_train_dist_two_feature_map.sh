
python -m torch.distributed.launch --nproc_per_node=4 main.py \
	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale.py --coco_path ../dataset/coco/ \
	--pretrained_checkpoint ../pretrained_models/torchvision_resnet50.pth --two_feature_map \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
