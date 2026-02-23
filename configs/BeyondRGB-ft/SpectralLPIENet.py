cfg = dict(
    exp_name = "BeyondRGB_finetuned_SpectralLPIENet_reproduction",

    train = True, # True for training, False for testing only
    test = True, # True for testing after training or during inference mode

    # Model to be trained
    model_type = "J_MSI", # "IE" (Illuminant Estimation) or "J" (Joint)
    model_name = "SpectralLPIENet",
    model_parameters = dict(
        rgb_input_channels = 3,
        spectral_input_channels = 16,
        output_channels = 3,
        encoder_dims = [16, 32, 64],
        decoder_dims = [32, 16],
        illuminant_head = False,
    ), # additional model parameters

    # Dataset settings
    data_type = "BeyondRGB", # "RGB" or "MS" or "RGB+MS"
    dataset_root = "./data/beyondRGB_final_dataset",
    rgb_camera = "RGB",
    gt_type = "xyz", # "xyz" or "srgb" or "raw"
    spectral_camera = "MS",
    train_list = "data/beyond_rgb_splits/train.txt",
    val_list = "data/beyond_rgb_splits/val.txt",
    test_list = "data/beyond_rgb_splits/test.txt",
   
    # Experiment settings
    seed = 42,
    device = 0, # choose GPU id (-1 for CPU)
    n_epochs = 300,
    n_workers = 16,
    lr = 4e-4,
    train_batch_size = 16,
    val_batch_size = 16,
    test_batch_size = 1,
    early_stop = 5,
    criterion = "ReproductionErrorLoss", # "L1Loss" or "L2Loss", "deltaE76Loss"
    metrics = ["ReproductionError", 
    "deltaE00", 
    # "LPIPS", 
    "PSNR"], # list of metrics to be evaluated during testing

    exp_dir = None, # will be set automatically if None
    train_checkpoint= None, # path to checkpoint to resume training (default: None)
    pretrained_weights = "experiments/260127_225008_BeyondRGB_SpectralLPIENet/best.pth", 
    val_viz_list = ["205","276","154","60","244","1","452","62"],
    test_viz_list = ["157","21","203","44","429","48","333","270","217","460"],
    test_viz_de00_range = [10,100],
)