import os
import yaml
import numpy as np
from imputegap.wrapper.AlgoPython.GPVAE.utils import dataloader, dataloader_physio, train, evaluate
import imputegap.tools.utils as ult_imp
from imputegap.wrapper.AlgoPython.GPVAE.models import *

# TODO: check if I need it ???
#def handle_parser(argv=None):
#
#    import argparse

#    parser = argparse.ArgumentParser(description='GPVAE')
#    parser.add_argument("--config", type=str, default="base.yaml")
#    #parser.add_argument("--testmissingratio", type=float, default=0.1)
#    #parser.add_argument("--sigma", type=int, default=1)
#    #parser.add_argument("--length_scale", type=int, default=7)
#    #parser.add_argument("--beta", type=float, default=0.2)
#    #parser.add_argument("--seed", type=int, default=27)

#    args, _unknown = parser.parse_known_args(argv)

#    return args

# TODO: include Norm as adaptable parameter
def recoverGPVAE(ts_m, model_type="gp-vae", seq_len=-1, batch_size=-1, epochs=100, sliding_windows=1, seed=27, tr_ratio=0.7, verbose=True):
    
    data_m = np.copy(ts_m)
    recov = np.copy(ts_m)
    m_mask = np.isnan(ts_m)

    # TODO: what does multivariate mean?
    multivariate, seq_len, batch_size, sliding_windows, error = ult_imp.prepare_deep_learning_params(incomp_data=ts_m, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)

    if error:
        return ts_m

    #args = handle_parser()
    #args.seq_len = seq_len
    #args.data_dim = ts_m.shape[1]
    #args.time_length = ts_m.shape[0]
    #print(ts_m.shape)

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "config/" + "base.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    data_dim = ts_m.shape[1]

    # TODO: Parameters to adapt to the given dataset - window_size, sigma, length_scale, beta
    sigma = 1.0
    length_scale = 2.0
    window_size = 3
    beta = config["train"]["missing_ratio"]
    decoder = GaussianDecoder



    #config["model"]["encoder_sizes"] = [int(size) for size in config["model"]["encoder_sizes"]]
    #config["model"]["decoder_sizes"] = [int(size) for size in config["model"]["decoder_sizes"]]
    
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    #config["model"]["test_missing_ratio"] = args.testmissingratio
    #config["model"]["beta"] = args.beta
    #config["model"]["sigma"] = args.sigma
    #config["model"]["length_scale"] = args.length_scale


    if verbose:
        print(f"(IMPUTATION) GP-VAE\n\tMatrix: {ts_m.shape[0]}, {ts_m.shape[1]}\n\tseq_len: {seq_len}\n\tbatch_size: {config['train']['batch_size']}\n\tepochs: {config['train']['epochs']}\n\tsliding_windows: {sliding_windows}\n\tmodel_type: {model_type}\n\tsigma: {sigma}\n\tlength scale: {length_scale}\n\twindow_size: {window_size}\n\tbeta: {beta}\n\ttr_ratio: {tr_ratio}\n\tseed: {seed}\n\tverbose: {verbose}\n")

    if multivariate:
        # TODO: fully implement the physio dataloader
        print("The Dataloader for the Physiodataset is not implemented so far!")
        return ts_m
    else:
        train_loader, valid_loader, test_loader, imp_loader, n_train, norm_param = dataloader(ts_m=data_m, seq_len=seq_len, missing_ratio=config["train"]["missing_ratio"], seed=seed, sliding_windows=sliding_windows, tr_ratio=tr_ratio, norm=True, batch_size=batch_size, verbose=verbose)

    print(len(test_loader))

    # TODO: only works with gp-vae so far, cause vae and hi-vae have no prior defined in __init__
    if model_type == "vae":
        model = VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                    encoder_sizes=config["model"]["encoder_sizes"], encoder=DiagonalEncoder,
                    decoder_sizes=config["model"]["decoder_sizes"], decoder=decoder,
                    image_preprocessor=None, window_size=window_size,
                    beta=beta, M=config["model"]["M"], K=config["model"]["K"])
        
    elif model_type == "hi-vae":
        model = HI_VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                       encoder_sizes=config["model"]["encoder_sizes"], encoder=DiagonalEncoder,
                       decoder_sizes=config["model"]["decoder_sizes"], decoder=decoder,
                       image_preprocessor=None, window_size=window_size,
                       beta=beta, M=config["model"]["M"], K=config["model"]["K"])
        
    elif model_type == "gp-vae":
        model = GP_VAE(latent_dim=config["model"]["latent_dim"], data_dim=data_dim, time_length=seq_len,
                       encoder_sizes=config["model"]["encoder_sizes"], encoder=BandedJointEncoder,
                       decoder_sizes=config["model"]["decoder_sizes"], decoder=GaussianDecoder,
                       kernel=config["model"]["kernel"], sigma=sigma,
                       length_scale=length_scale, kernel_scales = config["model"]["kernel_scales"],
                       image_preprocessor=None, window_size=window_size,
                       beta=beta, M=config["model"]["M"], K=config["model"]["K"], data_type=config["model"]["data_type"])
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")


    train(model=model, config=config, train_loader=train_loader, n_train=n_train, valid_loader=valid_loader, verbose=verbose)

    imp_matrix = evaluate(model=model, config=config, test_loader=test_loader, imp_loader=imp_loader, norm_param=norm_param, n_timestamps=ts_m.shape, multivariate=multivariate, verbose=verbose, sliding_windows=sliding_windows)

    #recov[m_mask] = imp_matrix[m_mask]

    if verbose:
        print(f"Recovered Matrix Shape: {imp_matrix.shape = }")

    return imp_matrix
