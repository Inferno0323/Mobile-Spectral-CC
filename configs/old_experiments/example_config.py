cfg = dict(
    exp_name = "experiment_name",


    # Model to be trained
    model_type = "IE" # "IE" (Illuminant Estimation) or "J" (Joint)
    model_name = "name of_model_or_path_to_custom_model.py",
    model_parameters = dict(), # additional model parameters


    # Dataset settings
    data_type = "RGB+MS" # "RGB" or "MS" or "RGB+MS"
    dataset_root = "./data/MobileSpectralAWBDataset/",
    rgb_camera = "GOOGLE_PIXEL_3",
    gt_type = "srgb", # "xyz" or "srgb" or "raw"
    spectral_camera = "SPECTRICITY_S1",
    train_list = "path_to_train_list.txt",
    val_list = "path_to_val_list.txt",
    test_list = "path_to_test_list.txt",
   

    # Experiment settings
    seed = 42,
    device = 0, # choose GPU id (-1 for CPU)
    n_epochs = 300,
    n_workers = 8,
    lr = 4e-4,
    train_batch_size = 8,
    val_batch_size = 1,
    test_batch_size = 1,
    early_stop = None, # number of epochs for early stopping (default: -1, no early stopping)
    criterion = "MAE", # "MAE" or "MSE"

    exp_dir = None, # will be set automatically if None
    checkpoint = None, # path to checkpoint to resume training (default: None)

    )