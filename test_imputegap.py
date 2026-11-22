from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils
import numpy as np

# initialize the time series object
ts = TimeSeries()



# load and normalize the dataset from the library
ts.load_series(utils.search_path("stock-exchange"), normalizer=None)
#ts.load_series(utils.search_path("eeg-alcohol"), normalizer=None)
#ts.load_series(utils.search_path("traffic"), normalizer=None)
#ts.load_series(utils.search_path("sport-activity"), normalizer="z_score")

ts_m = GenGap.mcar(ts.data, rate_dataset=0.4, rate_series=0.3, block_size=10, seed=True)
#ts.plot(input_data=ts.data, incomp_data=ts_m, nbr_series=1, nbr_val=2000, save_path="./imputegap_assets/imputation/Report", cont_rate="30",series_range=1)

# TODO enable stop also when the algorithms trains
imputer = Imputation.DeepLearning.GPVAE(ts_m)
imputer.impute()
#imputer.impute(user_def=True, params={"model_type": "gp-vae", "seq_len": -1, "batch_size": -1,"epoch": 10, "length_scale": 2.0, "window_size": -1, "sliding_window": 1, "norm": True})
#imputer.impute(user_def=False, params={"input_data": ts.data, "optimizer": "ray_tune"})

#imputer = Imputation.DeepLearning.GAIN(ts_m)
#imputer.impute()

imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=1, series_range=1, nbr_val=1000, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation/Report", cont_rate="30")

# TODO fix problem with benchmark
# Recomending to use GAIN, BRITS and MeanImpute and Forward Imputation
#my_algorithms = ["GPVAE", "MeanImpute", "GAIN"]
#my_algorithms = ["MPIN"]
#my_algorithms = ["SAITS"] -> doesn't work

#my_opt = "default_params"

#my_datasets = ["eeg-alcohol"]
#my_datasets = ["traffic"]
#my_datasets = ["stock-exchange"]

#my_patterns = ["mcar"]

#range = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
#range = [0.05, 0.1]

# launch the evaluation
#bench = Benchmark()
#bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizer=my_opt, normalizer=None, report_title="Benchmark_Report_Stock")

# display the plot
#bench.subplots.show()
