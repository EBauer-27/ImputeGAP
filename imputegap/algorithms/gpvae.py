import time

from imputegap.wrapper.AlgoPython.GPVAE.recoverGPVAE import recoverGPVAE

def gpvae(incomp_data, model_type="gp-vae", seq_len=-1, batch_size=-1, epochs=40, length_scale=2.0, window_size=-1, sliding_windows=1, norm=True, tr_ratio=0.7, seed=27, logs=True, verbose=True):
    """
    Perform imputation using GP-VAE

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    model_type: str, optional
        Variational Autoencoder type, can choose between ["vae", "hi-vae", "gp-vae"] (default: "gp-vae") -> so far only "gp-vae" is running
    
    seq_len : int, optional
        Length of the input sequence for the encoder. If -1, it will be automatically determined (default: -1).

    batch_size : int, optional
        Number of samples per batch during training/inference. If -1, it will be auto-set (default: -1).

    epochs : int, optional
        Number of epoch for training the model (default: 40).

    length_scale: float, optional
        Length Scale for the cauchy kernel (fine tunes how fast the correlation drops) (deflaut: 2.0)

    window_size: int, optional
        Kernel size for the CNN layers. If -1, it will be set to half the sequence length (default: -1)

    sliding_windows: int, optional
        Stride between consecutive training windows (default is 1). If set to -1, the window size is equal to seq_len.
        Use values ≥ 1 for univariate datasets (window strategy) and -1 for multivariate datasets (sample strategy).

    norm: bool, optional
        If True Dataloader normalizes incomp_data with z_normalization, based on teh training set (deflaut: True)
        Should be set to True if incomp_data are unnormalized (needed to avoid data leakage)

    seed : int, optional
        Random seed for reproducibility (default: 27).

    tr_ratio: float, optional
        Split ratio between training and testing sets (default is 0.7).

    logs : bool, optional
        Whether to print/log execution time and key events (default: True).

    verbose : bool, optional
        Whether to print detailed output information during execution (default: True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values filled in. (If input unnormalized Matrix, output Matrix also unnormalized)

    """
    start_time = time.time()  # Record start time

    recov_data = recoverGPVAE(ts_m=incomp_data, model_type=model_type, seq_len=seq_len, batch_size=batch_size, epochs=epochs, length_scale=length_scale, window_size=window_size, sliding_windows=sliding_windows, norm=norm, tr_ratio=tr_ratio, seed=seed, verbose=verbose)

    end_time = time.time()

    if logs and verbose:
        print(f"\n> logs: imputation GP-VAE - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data
