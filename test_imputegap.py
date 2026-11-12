from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils
from imputegap.wrapper.AlgoPython.GPVAE.utils import normalize, renormalize

import numpy as np
# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset from the library
# TODO: GP-VAE needs unnormalized data - Check !!!!!!!!!!!!!!
#ts.load_series(utils.search_path("meteo"), normalizer="z_score")
ts.load_series(utils.search_path("eeg-alcohol"), normalizer=None)
#ts.load_series(utils.search_path("meteo"), normalizer="z_score")
#ts.load_series(utils.search_path("sport-activity"), normalizer="z_score")



#imputer = Imputation.DeepLearning.GPVAE(ts_m)
#imputer.impute(user_def=True, params={"model_type": "gp-vae", "seq_len": -1, "batch_size": -1, "epoch": 40, "sliding_window": 1, "seed": 27})

# plot the recovered time series
#ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")

#my_algorithms = ["GPVAE"]

#my_opt = "default_params"

#my_datasets = ["eeg-alcohol"]

#my_patterns = ["mcar"]

#range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# launch the evaluation
#bench = Benchmark()
#bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizer=my_opt)

# display the plot
#bench.subplots.show()