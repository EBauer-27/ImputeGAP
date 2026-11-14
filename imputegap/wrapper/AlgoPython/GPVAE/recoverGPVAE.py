import os
import yaml
import numpy as np
from imputegap.wrapper.AlgoPython.GPVAE.utils import dataloader, dataloader_physio, train, evaluate
import imputegap.tools.utils as ult_imp
from imputegap.wrapper.AlgoPython.GPVAE.models import *


# TODO: include Norm as adaptable parameter
def recoverGPVAE(ts_m, model_type="gp-vae", seq_len=-1, batch_size=-1, epochs=40, length_scale=2.0, window_size=-1, sliding_windows=1, norm=True, tr_ratio=0.7, seed=27, verbose=True):
    """Main function to recover the ImputeGAP Time Series Datasets.
    
    Input:
        - ts_m: Time Series 
        - model_type: Model Type can choose between ["vae", "hi-vae", "gp-vae"] -> so far only "gp-vae" is running
        - seq_len: Length of each window (number of time steps per segment)
        - batch_size: batch size
        - epochs: training epochs
        - length_scale: fine tunes how fast the correlation drops
        - window_size: kernel size for the CNN layers (default half the sequence length)
        - sliding_windows: Step size between the starting indices of consecutive windows
        - norm: If True normalizes the TS, based on the training set
        - seed: seed
        - tr_ratio: ration of the training dataset 
        - verbose: If True, prints a summary of the computed windows

    Output:
        - imp_matrix: Imputed Matix (missing values filled with GP-VAE prediction)
    """
    data_m = np.copy(ts_m)
    #recov = np.copy(ts_m)
    #m_mask = np.isnan(ts_m)

    multivariate, seq_len, batch_size, sliding_windows, error = ult_imp.prepare_deep_learning_params(incomp_data=ts_m, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)

    # TEST: define window_size based on seq_len (half the sequence length) 
    if window_size == -1:
        window_size = int(seq_len/2)
    
    if error:
        return ts_m

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config/" + "base.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    data_dim = data_m.shape[1]

    # TODO so far decoder always GaussianDecoder; adapt if other datasets are implemented (hmnist=BernoulliDecoder)
    decoder = GaussianDecoder
    
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs

    # true if windows already exists (e.g. Physionet, split in patients)
    if multivariate:
        # TODO: fully implement the physio dataloader
        print("The Dataloader for the Physiodataset is not implemented so far!")
        return ts_m
    else:
        train_loader, valid_loader, test_loader, imp_loader, n_train, norm_param = dataloader(ts_m=data_m, config=config, seq_len=seq_len, seed=seed, sliding_windows=sliding_windows, tr_ratio=tr_ratio, norm=norm, verbose=verbose)

    if verbose:
            print(f"(IMPUTATION) GP-VAE\n\tMatrix: {ts_m.shape[0]}, {ts_m.shape[1]}\n\tmodel_type: {model_type}\n\tseq_len: {seq_len}\n\tbatch_size: {config['train']['batch_size']}\n\tepochs: {config['train']['epochs']}\n\tlength scale: {length_scale}\n\twindow_size: {window_size}\n\tsliding_windows: {sliding_windows}\n\tsigma: {config['model']['sigma']}\n\tbeta: {config['model']['beta']}\n\tnorm: {norm}\n\ttr_ratio: {tr_ratio}\n\tseed: {seed}\n\tverbose: {verbose}\n")

    # TODO: only works with gp-vae so far, cause vae and hi-vae have no prior defined in __init__
    if model_type == "vae":
        model = VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                    encoder_sizes=config["model"]["encoder_sizes"], encoder=DiagonalEncoder,
                    decoder_sizes=config["model"]["decoder_sizes"], decoder=decoder,
                    image_preprocessor=None, window_size=window_size,
                    beta=config['model']['beta'], M=config["model"]["M"], K=config["model"]["K"])
        
    elif model_type == "hi-vae":
        model = HI_VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                       encoder_sizes=config["model"]["encoder_sizes"], encoder=DiagonalEncoder,
                       decoder_sizes=config["model"]["decoder_sizes"], decoder=decoder,
                       image_preprocessor=None, window_size=window_size,
                       beta=config['model']['beta'], M=config["model"]["M"], K=config["model"]["K"])
        
    elif model_type == "gp-vae":
        model = GP_VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                       encoder_sizes=config["model"]["encoder_sizes"], encoder=BandedJointEncoder,
                       decoder_sizes=config["model"]["decoder_sizes"], decoder=GaussianDecoder,
                       kernel=config["model"]["kernel"], sigma=config['model']['sigma'],
                       length_scale=length_scale, kernel_scales = config["model"]["kernel_scales"],
                       image_preprocessor=None, window_size=window_size,
                       beta=config['model']['beta'], M=config["model"]["M"], K=config["model"]["K"], data_type=config["model"]["data_type"])
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")


    train(model=model, config=config, train_loader=train_loader, n_train=n_train, valid_loader=valid_loader, verbose=verbose)

    imp_matrix = evaluate(model=model, config=config, test_loader=test_loader, imp_loader=imp_loader, norm_param=norm_param, n_timestamps=ts_m.shape, multivariate=multivariate, verbose=verbose, sliding_windows=sliding_windows)

    #recov[m_mask] = imp_matrix[m_mask]

    if verbose:
        print(f"Recovered Matrix Shape: {imp_matrix.shape = }")

    return imp_matrix
