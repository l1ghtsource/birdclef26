from configs.bc_pretrain_perch_distill_5s import cfg

cfg.exp_name = "bc_pretrain_perch_distill_5s_tf_efficientnetv2_s_in21k_ft_in1k"
cfg.model.backbone.backbone_name = "models/tf_efficientnetv2_s.in21k_ft_in1k"
