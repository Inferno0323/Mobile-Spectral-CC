cfg = dict(
    exp_name = "SpectralLSMI_SVWB_Unet_ft_MobileSpectralAWB",

    train = True,
    test = True,

    # Model to be trained
    model_type = "IE",
    model_name = "SVWB_Unet",
    model_parameters = dict(
    ),

    # Dataset settings
    data_type = "LSMI",
    dataset_root = "./data/SpectralLSMI/",
    rgb_camera = "Canon R5",
    spectral_camera = "Spectricity S1",
    gt_type = "xyz",
    
    # LSMI-specific settings
    lsmi_output_type = "RGB",
    rgb_size = (512, 512),
    ms_size = (64,64),
    misaligned = False,
    
    # Data splits
    train_list = "data/lsmi_splits/train.txt",
    val_list = "data/lsmi_splits/val.txt",
    test_list = "data/lsmi_splits/test.txt",

    # Experiment settings
    seed = 42,
    device = 0,
    n_epochs = 300,
    n_workers = 8,
    lr = 4e-4,
    train_batch_size = 8,
    val_batch_size = 8,
    test_batch_size = 1,
    early_stop = 5,
    criterion = "AngularErrorLoss",
    metrics = ["ReproductionError", "deltaE00", "PSNR"],

    exp_dir = None,
    train_checkpoint = None,
    # pretrained_weights = None,
    pretrained_weights = "experiments/260216_180355_SpectralLSMI_SVWB_Unet/best.pth",
    val_viz_list = ["sony_Place959_12_GT_AS","sony_Place1236_13_GT_AS","sony_Place149_12_GT_AS","sony_Place240_12_GT_AS","sony_Place252_12_GT_AS"],
    test_viz_list = ["nikon_Place935_12_GT_AS","sony_Place932_12_GT_AS","sony_Place72_12_GT_AS","sony_Place787_12_GT_AS","sony_Place422_12_GT_AS","sony_Place74_12_GT_AS","sony_Place915_12_GT_AS","sony_Place34_12_GT_AS","sony_Place79_12_GT_AS","nikon_Place940_12_GT_AS","sony_Place1147_123_GT_AS","sony_Place324_12_GT_AS","sony_Place256_12_GT_AS","sony_Place375_12_GT_AS","sony_Place71_12_GT_AS","nikon_Place932_13_GT_AS","sony_Place855_12_GT_AS","sony_Place599_12_GT_AS","sony_Place398_12_GT_AS","nikon_Place61_12_GT_AS","sony_Place800_12_GT_AS","sony_Place68_12_GT_AS","sony_Place801_12_GT_AS","sony_Place395_12_GT_AS","sony_Place507_12_GT_AS","nikon_Place559_12_GT_AS","sony_Place935_12_GT_AS","sony_Place1132_12_GT_AS","sony_Place851_12_GT_AS","nikon_Place941_123_GT_AS","sony_Place255_12_GT_AS","nikon_Place949_12_GT_AS","nikon_Place288_12_GT_AS","nikon_Place951_13_GT_AS","sony_Place872_12_GT_AS","nikon_Place911_12_GT_AS","sony_Place840_12_GT_AS","sony_Place380_12_GT_AS","sony_Place1152_13_GT_AS","nikon_Place144_12_GT_AS","sony_Place1201_13_GT_AS","galaxy_Place74_12_GT_AS","nikon_Place944_12_GT_AS","sony_Place899_12_GT_AS","sony_Place895_12_GT_AS","sony_Place921_12_GT_AS","nikon_Place946_13_GT_AS","nikon_Place950_12_GT_AS","nikon_Place947_12_GT_AS","sony_Place583_12_GT_AS"],
    test_viz_de00_range = [10, 100],
)
