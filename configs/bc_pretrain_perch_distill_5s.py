import pprint
from types import SimpleNamespace

# data: 2022-2025 data + old extra data + 2026 labeled/unlabeled data + xc/inat/tsa parsed data
# task: backbone perch distillation pretrain

cfg = SimpleNamespace(**{})
cfg.exp_name = "bc_pretrain_perch_distill_5s"

cfg.data_root = "data"
cfg.train_path = "train.csv"
cfg.train_soundscapes_path = "train_soundscapes"
cfg.test_soundscapes_path = "test_soundscapes"

cfg.is_reversed_audio = False
cfg.mel_spec_params = SimpleNamespace(
    mel_top_db=80,
    mel_image_size=(384, 384),
    mel_delta_stack=False,
    sample_rate=32000,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    f_min=0,
    f_max=16000,
    norm_method="per_sample_absmax",
)

cfg.dataset = SimpleNamespace(
    taxonomy_csv="old_extra_data/pretrain_taxonomy.csv",
    train_csv="train.csv",
    build_manifest_out="old_extra_data/pretrain_manifest.csv",
    soundscapes_labels_csv="train_soundscapes_labels.csv",
    train_audio_subdir="train_audio",
    additional_train_csv=[
        {"train_csv": "old_extra_data/pretrain_manifest.csv", "train_audio_subdir": ""},
    ],
    train_soundscapes_subdir="train_soundscapes",
    use_train_audio=True,
    use_train_soundscapes=True,
    pl_path="../TOP_SVALKA/top13_raw_pseudos_25april.csv",
    pl_filter=False,
    pl_filter_thr=0.5,
    pl_zero_unconf=False,
    pl_zero_unconf_thr=0.1,
    sample_rate=32000,
    chunk_duration_s=5.0,
    soundscape_label_bin_s=5.0,
    crop_mode="random",
    padding_mode="random_place",
    secondary_label_weight=1.0,
    audio_blocklist_rel=None,
    audio_blocklist_rels=(
        "old_extra_data/pretrain_manifest_rejected_audio.txt",
        "old_extra_data/extra_sources_rejected_audio.txt",
    ),
    filter_invalid_background_noise=False,
    use_reshape_waves_not_melspec=False,
    wave_reshape_width=80,
    replace_corrupted=True,
    extra_sources_data=[
        "inat_downloaded",
        "tsa_downloaded",
        "xc_downloaded",
    ],
    extra_sources_meta=[
        "inat_parsed",
        "tsa_parsed",
        "xc_parsed",
    ],
    extra_filter_geo="none",
    extra_rare_thr="none",
    extra_max_class_num="none",
    extra_stats_csv=None,
    extra_merge_taxonomy_from_folders=True,
)

cfg.sampler = "none"
cfg.ss_sampling_weight = "none"
cfg.do_upsampling = False
cfg.upsampling_n = 100

cfg.mel_spec_aug = SimpleNamespace(
    p_random_gain_db=1.0,
    random_gain_min_db=-6.0,
    random_gain_max_db=6.0,
    p_filt_aug=1.0,
    filt_aug_db_range=(-6.0, 6.0),
    filt_aug_n_band=(3, 6),
    filt_aug_min_bw=6,
    filt_aug_filter_type="linear",
    p_spec_aug=0.25,
    spec_aug_num_freq_masks=1,
    spec_aug_num_time_masks=2,
    stretch_global_prob=0.0,
    stretch_local_prob=0.0,
    stretch_max_global=0.2,
    stretch_max_local=0.3,
    stretch_max_local_regions=3,
    p_time_shift=0.2,
    time_shift_max_pct=0.1,
    p_freq_shift=0.0,
    freq_shift_max_pct=0.05,
    p_gaussian_noise=0.0,
    gaussian_noise_std=0.01,
    p_random_erasing=0.0,
    random_erasing_scale=(0.02, 0.08),
    random_erasing_ratio=(0.5, 2.0),
)
cfg.wave_aug = SimpleNamespace(
    background_noise_prob=0.0,
    background_noise_dirs=("background_noises",),
    background_noise_min_snr_db=3.0,
    background_noise_max_snr_db=30.0,
)

cfg.online_aug = SimpleNamespace(
    p_mixup=0.0,
    mixup_alpha=0.0,
    mixup_use_max_label=False,
    p_sumix_freq=0.0,
    wave_level=False,
    use_ss_bank=False,
    ss_bank_share=0.0,
)

cfg.is_train = True
cfg.is_infer = False

cfg.num_classes = None
cfg.do_full_retrain = True
cfg.n_splits = 5
cfg.curr_folds = [0]
cfg.val_strategy = "skf"
cfg.val_split_pool = "all"
cfg.seed = 69

cfg.model = SimpleNamespace(
    model_type="pretrain_perch_distill",
    backbone=SimpleNamespace(
        backbone_name="models/tf_efficientnet_b0.ns_jft_in1k",
        init_checkpoint=None,
        pretrained=True,
        drop_rate=0.3,
        drop_path_rate=0.15,
    ),
    attn_block=SimpleNamespace(
        head_type="att",
        activation="sigmoid",
        dropout=0.5,
        hidden_dim=512,
        att_activation="tanh",
        norm="softmax",
        eps=1e-7,
        use_se=False,
        use_gru_before=False,
        use_gru_after=False,
        use_complex_convs=False,
        se_reduction=16,
        gru_before_hidden=None,
        gru_after_hidden=None,
        gru_layers=1,
        channel_smoothing="max_plus_avg",
        segwise_pooling="max",
    ),
    multicontext=False,
)

cfg.distill_perch = True
cfg.distill_perch_coef = 1.0
cfg.distill_perch_coef_min = 1.0
cfg.distill_perch_coef_scheduler = "none"
cfg.distill_perch_emb_loss_alpha = 0.5
cfg.distill_perch_do_norm = True

cfg.pretrain_encoder_filename = "encoder_efficientnet_b0.pt"
cfg.pretrain_save_lightning_checkpoints = True

cfg.bs = 128
cfg.n_epochs = 30
cfg.lr = 1e-4
cfg.weight_decay = 1e-4
cfg.num_warmup_steps_ratio = 0.05
cfg.max_norm = 1.0

cfg.scheduler = "cosine"
cfg.optim_type = "adamw"

cfg.loss_name = "mse"
cfg.label_smoothing = 0.0
cfg.focal_alpha = 0.25
cfg.focal_gamma = 2.0
cfg.asymmetric_gamma_neg = 4.0
cfg.asymmetric_gamma_pos = 1.0
cfg.asymmetric_clip = 0.05

cfg.use_ema = False
cfg.ema_decay = 0.999
cfg.use_awp = False
cfg.awp_lr = 1e-3
cfg.awp_eps = 1e-2
cfg.use_rdrop = False
cfg.rdrop_alpha = 4.0

cfg.num_workers = 32
cfg.devices = "0"

cfg.checkpoint_save_last_k = 5
cfg.log_dir = "logs"
cfg.model_dir = "weights"
cfg.oof_dir = "oofs"
cfg.do_tensorboard_log = True
cfg.log_dir_steps = 1
cfg.tensorboard_project = "birdclef-2026"

pprint.pprint(vars(cfg))
