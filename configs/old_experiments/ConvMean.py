cfg = dict(
    exp_name = "ConvMean_test",

    train = True, # True for training, False for testing only
    test = True, # True for testing after training or during inference mode

    # Model to be trained
    model_type = "IE", # "IE" (Illuminant Estimation) or "J" (Joint)
    model_name = "ConvMean",
    model_parameters = dict(
        inp_size=3
    ), # additional model parameters

    # Dataset settings
    data_type = "RGB", # "RGB" or "MS" or "RGB+MS"
    dataset_root = "./data/MobileSpectralAWBDataset/",
    rgb_camera = "GOOGLE_PIXEL_3",
    gt_type = "xyz", # "xyz" or "srgb" or "raw"
    spectral_camera = "SPECTRICITY_S1",
    train_list = "data/data_splits/scene_wise/train.txt",
    val_list = "data/data_splits/scene_wise/val.txt",
    test_list = "data/data_splits/scene_wise/test.txt",
   
    # Experiment settings
    seed = 42,
    device = 0, # choose GPU id (-1 for CPU)
    n_epochs = 300,
    n_workers = 16,
    lr = 4e-4,
    train_batch_size = 16,
    val_batch_size = 16,
    test_batch_size = 16,
    early_stop = 5,
    criterion = "AngularErrorLoss", # "MAE" or "MSE"
    metrics = ["ReproductionError", 
    "deltaE00", 
    # "LPIPS", 
    "PSNR"], # list of metrics to be evaluated during testing

    exp_dir = None, # will be set automatically if None
    train_checkpoint= None, # path to checkpoint to resume training (default: None)
    pretrained_weights = None, 
    val_viz_list = ["SC002_B_ILL000", "SC076_B_ILL099"],
    test_viz_list = ["SC068_B_ILL010", "SC003_K_ILL067"]
    )