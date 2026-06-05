import pprint
from types import SimpleNamespace

# --- cfg ---
cfg = SimpleNamespace(**{})
cfg.exp_name = "0_920_best_full"

# --- pathes ---
# TODO: add extra datasets (xc)
cfg.data_root = "data"
cfg.train_path = "train.csv"
cfg.train_soundscapes_path = "train_soundscapes"
cfg.test_soundscapes_path = "test_soundscapes"

# --- data ---
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
    norm_method="per_sample_absmax",  # db_scaled, per_sample_minmax, per_sample_absmax [best norm ?]
)

# --- dataset---
cfg.dataset = SimpleNamespace(
    taxonomy_csv="taxonomy.csv",
    train_csv="train.csv",
    soundscapes_labels_csv="train_soundscapes_labels.csv",
    train_audio_subdir="train_audio",
    train_soundscapes_subdir="train_soundscapes",
    use_train_audio=True,
    use_train_soundscapes=True,
    sample_rate=32000,
    chunk_duration_s=5.0,
    soundscape_label_bin_s=5.0,
    crop_mode="random",  # random, rms, start_random, end_random
    padding_mode="repeated",  # left, centered, random_place, repeated
    secondary_label_weight=1.0,
    use_reshape_waves_not_melspec=False,
    wave_reshape_width=80,
    replace_corrupted=True,
)

# --- sampler ---
cfg.sampler = "none"  # balanced, square_balanced, none
cfg.ss_sampling_weight = "none"  # [true bad ?]
cfg.do_upsampling = False  # [true bad ?]
cfg.upsampling_n = 100  # test 10 with augs

# --- offline augs ---
cfg.mel_spec_aug = SimpleNamespace(
    # random gain [good ?]
    p_random_gain_db=1.0,
    random_gain_min_db=-6.0,
    random_gain_max_db=6.0,
    # filter augmentation [good ?]
    p_filt_aug=1.0,
    filt_aug_db_range=(-6.0, 6.0),
    filt_aug_n_band=(3, 6),
    filt_aug_min_bw=6,
    filt_aug_filter_type="linear",
    # spec augmentation [bad ?]
    p_spec_aug=0.0,
    spec_aug_num_freq_masks=1,
    spec_aug_num_time_masks=2,
    # stretch augmentation [bad ?]
    stretch_global_prob=0.0,
    stretch_local_prob=0.0,
    stretch_max_global=0.2,
    stretch_max_local=0.3,
    stretch_max_local_regions=3,
    # time shift augmentation [idk ?]]
    p_time_shift=0.0,
    time_shift_max_pct=0.15,
    # freq shift augmentation
    p_freq_shift=0.0,
    freq_shift_max_pct=0.05,
    # gaussian noise augmentation [bad ?]
    p_gaussian_noise=0.0,
    gaussian_noise_std=0.01,
    # random erasing augmentation
    p_random_erasing=0.0,
    random_erasing_scale=(0.02, 0.08),
    random_erasing_ratio=(0.5, 2.0),
)
cfg.wave_aug = SimpleNamespace(
    background_noise_prob=0.0,  # [bad ?]
    background_noise_dirs=("background_noises",),
    background_noise_min_snr_db=3.0,
    background_noise_max_snr_db=30.0,
)

# --- online augs ---
cfg.online_aug = SimpleNamespace(
    p_mixup=0.0,  # wave 1.0 w/ 1.5 ok
    mixup_alpha=1.5,
    mixup_use_max_label=False,
    p_sumix_freq=0.5,  # mel 0.5 ok
    p_horizontal_cutmix=0.0,
    horizontal_cutmix_alpha=1.0,
    wave_level=False,  # good only for mixup
    use_ss_bank=False,  # [bad ?]
    ss_bank_share=0.0,
)

# --- train/infer flags ---
cfg.is_train = True
cfg.is_infer = False

# --- important vars ---
cfg.num_classes = 234
cfg.do_full_retrain = True
cfg.n_splits = 5
cfg.curr_folds = [0, 1, 2, 3, 4]
cfg.val_strategy = "skf"  # kf, skf
cfg.val_split_pool = "soundscape"  # all, train_audio, soundscape
cfg.seed = 69

# --- model ---
cfg.model = SimpleNamespace(
    model_type="sed",
    backbone=SimpleNamespace(
        backbone_name="models/tf_efficientnet_b0.ns_jft_in1k",
        init_checkpoint="weights/bc_pretrain_perch_distill_5s/encoder_efficientnet_b0.pt",
        pretrained=False,
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

# --- train params ---
cfg.bs = 64
cfg.n_epochs = 20
cfg.lr = 1e-3
cfg.head_to_bb_lr_ratio = "10:1"
cfg.weight_decay = 1e-4
cfg.num_warmup_steps_ratio = 0.03
cfg.max_norm = 2.0

cfg.scheduler = "cosine"  # cosine, linear, constant
cfg.optim_type = "adamw"  # adamw, muon, radam

cfg.loss_name = "focal_loss_plus_bce"  # [best loss ?]
cfg.label_smoothing = 0.005
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

# --- saving and logging ---
cfg.checkpoint_save_last_k = 5
cfg.log_dir = "logs"
cfg.model_dir = "weights"
cfg.oof_dir = "oofs"
cfg.do_tensorboard_log = True
cfg.log_dir_steps = 1
cfg.tensorboard_project = "birdclef-2026"

pprint.pprint(vars(cfg))
