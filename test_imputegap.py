from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils
import numpy as np

# initialize the time series object
ts = TimeSeries()



# load and normalize the dataset from the library
#ts.load_series(utils.search_path("meteo"), normalizer="z_score")
ts.load_series(utils.search_path("eeg-alcohol"), normalizer=None)
#ts.load_series(utils.search_path("meteo"), normalizer="z_score")
#ts.load_series(utils.search_path("sport-activity"), normalizer="z_score")

ts_m = GenGap.mcar(ts.data, rate_dataset=0.4, rate_series=0.2, block_size=10, seed=True)

# TODO enable stop also when the algorithms trains
imputer = Imputation.DeepLearning.GPVAE(ts_m)
#imputer.impute()
#imputer.impute(user_def=True, params={"model_type": "gp-vae", "seq_len": -1, "batch_size": -1,"epoch": 100, "length_scale": 7.0, "window_size": -1, "sliding_window": 1, "norm": True})
imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

#imputer = Imputation.DeepLearning.GAIN(ts_m)
#imputer.impute()
# plot the recovered time series
#ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")

# TODO fix problem with benchmark
#my_algorithms = ["MeanImpute", "GAIN", "GPVAE"]
#my_algorithms = ["MPIN"]
#my_algorithms = ["SAITS"] -> doesn't work

#my_opt = "default_params"

#my_datasets = ["eeg-alcohol"]

#my_patterns = ["mcar"]

#range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
#range = [0.05, 0.1]

# launch the evaluation
#bench = Benchmark()
#bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizer=my_opt, normalizer=None)

# display the plot
#bench.subplots.show()
