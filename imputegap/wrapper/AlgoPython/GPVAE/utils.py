import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd
import tf_keras
from tqdm import tqdm 
import math

import imputegap.tools.utils as ult_imp


''' TF utils '''


def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.math.log(tf.reduce_mean(
            tf.exp(x - x_max), axis=axis, keepdims=True) + eps) + x_max


def multiply_tfd_gaussians(gaussians):
    """Multiplies two tfd.MultivariateNormal distributions."""
    mus = [gauss.mean() for gauss in gaussians]
    Sigmas = [gauss.covariance() for gauss in gaussians]
    mu_3, Sigma_3, _ = multiply_gaussians(mus, Sigmas)
    return tfd.MultivariateNormalFullCovariance(loc=mu_3, covariance_matrix=Sigma_3)


def multiply_inv_gaussians(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [n, d]
    lambdas: list of precision matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = int(mus[0].shape[0])
    d_z = int(lambdas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), [-1,d_z,d_z])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bij, bj -> bi", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bij, bj -> bi", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_inv_gaussians_batch(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [..., d]
    lambdas: list of precision matrices with shape [..., d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = mus[0].shape.as_list()[:-1]
    d_z = lambdas[0].shape.as_list()[-1]
    identity_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(d_z), axis=0), axis=0), batch_size+[1,1])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bcij, bcj -> bci", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bcij, bcj -> bci", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_gaussians(mus, sigmas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of covariance matrices.
    mus: list of mean with shape [n, d]
    sigmas: list of covariance matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(sigmas)
    batch_size = [int(n) for n in mus[0].shape[0]]
    d_z = int(sigmas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), batch_size+[d_z,d_z])
    sigma_new = identity_matrix
    mu_new = tf.zeros((batch_size, d_z))
    for mu, sigma in zip(mus, sigmas):
        sigma_inv = tf.linalg.inv(sigma_new + sigma)
        sigma_prod = tf.matmul(tf.matmul(sigma_new, sigma_inv), sigma)
        mu_prod = (tf.einsum("bij,bj->bi", tf.matmul(sigma, sigma_inv), mu_new)
                   + tf.einsum("bij,bj->bi", tf.matmul(sigma_new, sigma_inv), mu))
        sigma_new = sigma_prod
        mu_new = mu_prod
    lambda_new = tf.linalg.inv(sigma_new)
    return mu_new, sigma_new, lambda_new

def normalize(data, parameters=None):
    """Z-normalization
  
    Args:
        - data: original data
  
    Returns:
        - norm_data: normalized data
        - norm_parameters: mean and standartdeviation of the given data
    """
    norm_data = data.copy()

    if parameters is None:
        mean = np.nanmean(norm_data, axis=0)
        std = np.nanstd(norm_data, axis=0)

        parameters = {"mean": mean,
                      "std": std}

        # Avoid division by zero: set std_dev to 1 where it is zero
        std[std == 0] = 1

    else: 
        mean = parameters["mean"]
        std = parameters["std"]

    norm_data = (norm_data - mean) / std

    return norm_data, parameters



def renormalize(norm_data, norm_parameters):
    """Renormalize data to the original range.
  
    Args:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
  
    Returns:
        - renorm_data: renormalized original data
    """
    mean = norm_parameters["mean"]
    std = norm_parameters["std"]

    norm_data = np.asarray(norm_data)
    renorm_data = norm_data.copy()

    renorm_data = (renorm_data * std) + mean

    return renorm_data

# function to create a custom dataloader for imputGAP Datasets
def dataloader(ts_m, seq_len=48, missing_ratio=0.1, seed=27, sliding_windows=1, tr_ratio=0.7, norm=True, batch_size=64, verbose=True):
    """Function to create a custom dataloader to be able to apply the model on imputGAP Datasets.
    
    ts_m: ImputeGAP dataset with missing values
    seq_len: sequence length of the windows
    missing_ratio: applied missing ratio for the training, testing, and evaluation
    seed: random seed
    sliding_window: sliding window for the window creation
    tr_ratio: training ratio
    batch_size: batch size
    verbose: verbose

    Returns 
    tf_x_train_miss: training dataset
    tf_x_val_miss: validation dataset
    tf_x_test_miss: testing dataset
    tf_impute_gap: full dataset for the final imputeGAP reconstruction
    train_end: number of training windows
    param: normalization parameter
    """
    # creates a mask for all observed values and put all nan values to 0
    obs_ts = np.asarray(ts_m, dtype=np.float32)
    obs_mask = ~np.isnan(obs_ts)     # Missing Mask (True: value, False: missing) - no artificially introduced missingness
    obs_ts = np.nan_to_num(obs_ts)  # Input Time-Series with 0.0 for nan values 
    T = obs_ts.shape[0]

    # randomly creates a missingnes mask
    rng = np.random.default_rng(seed)
    obs_idx = np.flatnonzero(obs_mask) 
    miss_idx = rng.choice(obs_idx, int(len(obs_idx) * missing_ratio), replace=False)    # randomly selects idx for artificial missing values
    mask_flat = obs_mask.flatten()
    mask_flat[miss_idx] = False     # introduce artificial missing to Missing Mask
    hide_mask = mask_flat.reshape(obs_mask.shape)    # 3D Missing Mask (True, False)

    # create train, test, validation split ratio based on the training ratio
    ts_r, val_r = ult_imp.sets_splitter_based_on_training(tr_ratio, verbose=False)

    # calculate train, test, validation indexes
    test_len= max(1, round(T*ts_r))
    val_len = max(1, round(T*val_r))
    test_start = T - test_len
    val_start = test_start - val_len

    sl_train, sl_val, sl_test = slice(0, val_start), slice(val_start, test_start), slice(test_start, T)

    # normalize dataset
    # get normalization parameter for train set and use this to normalize the time series to avido data leakage
    if norm:
        _, param = normalize(obs_ts[sl_train], None)

        obs_ts, _ = normalize(obs_ts, param)
        obs_ts = obs_ts.astype(np.float32)
    else:
        param = None
    

    win = lambda a: ult_imp.window_truncation(a, seq_len=seq_len, stride=sliding_windows, info=" |", verbose=verbose)

    def split(sl):
        """Creates the separate datasets.
        Input: slide indices.
        Output:
        x_miss: Time Series with artificially missing values
        m_miss: Missing Mask (missing values = 1)
        x_full: Time-Series without artificually missing values (given input)
        eval_w: Information which values are artificially hidden -> 1
        """
        ts_full, obs_m, hide_m = win(obs_ts[sl]), win(obs_mask[sl]), win(hide_mask[sl])
        return ts_full * hide_m, 1.0 - hide_m, ts_full, obs_m - hide_m

    x_train, m_train, _, _ = split(sl_train)
    x_val, m_val, _, _ = split(sl_val)
    x_test, m_test, x_full_test, eval_w_test = split(sl_test)

    x_imp, mask_imp = win(obs_ts), win(obs_mask)

    tf_train = tf.data.Dataset.from_tensor_slices((x_train, m_train))\
                                                  .shuffle(len(x_train))\
                                                  .batch(batch_size, drop_remainder=True)\
                                                  .repeat()

    tf_val = tf.data.Dataset.from_tensor_slices((x_val, m_val))\
                                                 .batch(batch_size, drop_remainder=True)\
                                                 .repeat()

    tf_test = tf.data.Dataset.from_tensor_slices((x_test, m_test, x_full_test, eval_w_test))\
                                                 .batch(batch_size, drop_remainder=True)

    tf_impute_gap = tf.data.Dataset.from_tensor_slices((x_imp, 1.0 - mask_imp))\
                                                        .batch(batch_size, drop_remainder=True)

    return tf_train, tf_val, tf_test, tf_impute_gap, len(x_train), param

    # normalize data
    #train_t_end = (train_end - 1) * sliding_windows + seq_len
    #vals  = observed_values[:train_t_end]      
    #known = gt_masks[:train_t_end]             

    #mean = np.array([v[k].mean() for v, k in zip(vals.T, known.T)])
    #std  = np.array([v[k].std()  for v, k in zip(vals.T, known.T)])
    #std[std == 0] = 1.0

    # apply the mask
    #x_full = (x_full - mean) / std
    #x_full = (x_full * observed_w).astype(np.float32)
    
    #x_miss = (x_full * gt_w).astype(np.float32)
    #m_miss = (1.0 - gt_w).astype(np.float32)    # GP-VAE convention: 1 = missing
    #eval_w = (observed_w - gt_w).astype(np.float32)      # 1 = known but artificially hidden

    # apply the mask
    #x_miss = x_full * gt_w
    #m_miss = 1.0 - gt_w             # GP-VAE convention: 1 = missing
    #eval_w = observed_w - gt_w      # 1 = known but artificially hidden
    
    # x_miss: Time-Series with artificially missing values
    # m_miss: Missing Mask (missing values = 1)
    # x_full: Time-Series without artificually missing values (given input)
    # eval_w: Information which values are artificially hidden (1)
    



# TODO How to use it and plot the recoverd stuff using imputeGAP 
def dataloader_physio(physio, seq_len=48, missing_ratio=0.1, seed=27, sliding_windows=1, tr_ratio=0.7, batch_size=64):
    # Load data
    data = lambda k: physio[k].astype(np.float32)
    x_train_full = data('x_train_full')
    x_train_miss = data('x_train_miss')
    m_train_miss = data('m_train_miss')

    x_val_full = data("x_val_full") 
    x_val_miss = data("x_val_miss")
    m_val_miss = data("m_val_miss")

    x_test_full = data('x_test_full')
    x_test_miss = data('x_test_miss')
    m_test_miss = data('m_test_miss')
    m_test_arti = data('m_test_artificial')

    tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss))\
                                     .shuffle(len(x_train_miss), seed=seed).batch(batch_size).repeat()

    tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss))\
                                   .batch(batch_size).repeat()

    tf_x_test_miss = tf.data.Dataset.from_tensor_slices((x_test_miss, m_test_miss, x_test_full, m_test_arti))\
                                    .batch(batch_size)

    tf_impute_gap = tf.data.Dataset.from_tensor_slices((x_test_miss, m_test_arti))\
                                   .batch(batch_size)

    return tf_x_train_miss, tf_x_val_miss, tf_x_test_miss, tf_impute_gap, len(x_train_miss)

    

def train(model, config, train_loader, n_train, valid_loader, foldername="", verbose=True):
    """Function to train the model on time-series data. 
    
    model: model (GP-VAE)
    config: pre configurations
    train_loader: train loader
    n_train: number of training windows
    valid_loader: validation loader
    foldername: name of the folder to save model (doesn't save so far)
    verbose: verbose
    """

    if verbose:
        print(f"\n\ntraining of the GP-VAE model:")

        # so far no GPU support
        print("GPU support: ", tf.config.list_physical_devices('GPU'))

    trainable_vars = model.get_trainable_vars()
    optimizer = tf_keras.optimizers.Adam(learning_rate=float(config["train"]["learning_rate"])) 
    print(float(config["train"]["learning_rate"]))
     
    if verbose:
        print("Encoder: ", model.encoder.net.summary())
        print("Decoder: ", model.decoder.net.summary())

    # TODO Doesn't save so far
    if foldername != "":
        output_path = foldername + "/model.path"


    if config["train"]["num_steps"] == 0:
        num_steps = config["train"]["epochs"] * n_train // config["train"]["batch_size"]
    else:
        num_steps = config["train"]["num_steps"]

    eval_interval = num_steps // config["train"]["epochs"]

    valid_loader = iter(valid_loader)

    pbar = tqdm(enumerate(train_loader.take(num_steps)), total=num_steps, desc="Training")
    for i, (x_seq, m_seq) in pbar:
        try:
            with tf.GradientTape() as tape:
                tape.watch(trainable_vars)
                loss = model.compute_loss(x_seq, m_mask=m_seq)
            grads = tape.gradient(loss, trainable_vars)
            grads = [np.nan_to_num(grad) for grad in grads]
            grads, global_norm = tf.clip_by_global_norm(grads, float(config["train"]["gradient_clip"])) 
            optimizer.apply_gradients(zip(grads, trainable_vars))

            if i % eval_interval == 0:
                x_val_batch, m_val_batch = next(valid_loader)
                val_loss, val_nll, val_kl = model.compute_loss(x_val_batch, m_mask=m_val_batch, return_parts=True)
                pbar.set_postfix(train=f"{float(loss):.3f}", val=f"{float(val_loss):.3f}",
                                 nll=f"{float(val_nll):.3f}", kl=f"{float(val_kl):.3f}")

                # Update learning rate (used only for physionet with decay=0.5)
                if i > 0 and i % (10*eval_interval) == 0:
                    optimizer.learning_rate = max(0.5 * optimizer.learning_rate, 0.1 * float(config["train"]["learning_rate"]))
        
        except KeyboardInterrupt:
            print("\n Training interrupted, saving current state.")

    # TODO Doesn't save so far
    # if foldername != "":

# TODO adapt the MSE Score
def evaluate(model, config, test_loader, imp_loader=None, norm_param=None, foldername="", n_timestamps=0, multivariate=False, verbose=True, sliding_windows=1):
    """Function to evaluate the trained model and recover the given time-series"
    

    Returns the recovered time series
    """
    n_batches = int(tf.data.experimental.cardinality(test_loader))

    if verbose:
        if n_timestamps == 0:
            print(f"\n\nevaluation of the GP-VAE model:")
        else:
            print(f"\n\nreconstruction of the imputed matrix:")

    nll_total = 0.0
    mse_total = 0.0
    n_missings = 0.0
    imputed = []

    pbar = tqdm(test_loader, total=n_batches, desc="Evaluation")
    for x_b, m_b, x_full_b, eval_b in pbar:
        # Compute NLL and MSE on missing values
        nll_total    += float(model.compute_nll(x_b, y=x_full_b, m_mask=eval_b))
        mse_total    += float(model.compute_mse(x_b, y=x_full_b, m_mask=eval_b))
        n_missings += float(tf.reduce_sum(eval_b))

        pbar.set_postfix(nll=f"{nll_total / n_missings:.4f}", mse=f"{mse_total / n_missings:.4f}")


        # Only needed is AUROC is implemented
        # impute time series
        #z_mean = model.encode(x_b).mean()   # center of the encoded distribution over the latent space z
        #x_imp = model.decode(z_mean).mean().numpy()    # mean of the decoded distribution over the data x

        #x_imp[m_b == 0] = x_b[m_b == 0]     # recovers the known values from the time-series

        #imputed.append(x_imp)

        # TODO AUROC - check if it is already implemented in imputeGAP evaluation
        # AUROC evaluation using Logistic Regression
        #x_val_imputed = x_imp.reshape([-1, config["time_length"] * config["data_dim"]])
        #val_split = len(x_val_imputed) // 2
        #cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
        #cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
        #probs = cls_model.predict_proba(x_val_imputed[val_split:])[:, 1]
        #auprc = average_precision_score(y_val[val_split:], probs)
        #auroc = roc_auc_score(y_val[val_split:], probs)


    if verbose:
        print("\nresult from the model\nNLL miss:{:.4f}".format(nll_total/n_missings))
        print("MSE miss:{:.4f}".format(mse_total/n_missings))

    if imp_loader is None:
        return 

    for x_b, m_b in imp_loader:
        z_mean = model.encode(x_b).mean()   # center of the encoded distribution over the latent space z
        x_imp = model.decode(z_mean).mean().numpy()    # mean of the decoded distribution over the data x

        # Do it later in recoverGPVAE.py
        x_imp[m_b == 0] = x_b[m_b == 0]     # recovers the known values from the time-series

        imputed.append(x_imp)

    recover = np.concatenate(imputed)

    if not multivariate:
        recover = ult_imp.reconstruction_window_based(preds=recover, nbr_timestamps=n_timestamps[0], sliding_windows=sliding_windows)

    if norm_param is not None:
        recover = renormalize(recover, norm_param)

    return recover