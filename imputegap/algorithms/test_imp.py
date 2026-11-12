import numpy as np
import time

def test_imp(incomp_data, mean=True, fix_value=0.0, logs=True, verbose=True):
    """
    Impute NaN values with the mean value of the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series.

    Notes
    -----
    This function finds the non-NaN value in the time series and replaces all NaN values with this mean value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
        >>> incomp_data = np.array([[5, 2, np.nan], [3, np.nan, 6]])
        >>> recov_data = mean_impute(incomp_data)
        >>> print(recov_data)
        array([[5., 2., 4.],
               [3., 4., 6.]])

    """

    # core of the algorithm
    if mean:
        recovery_value = np.nanmean(incomp_data)
    else:
        recovery_value = fix_value

    start_time = time.time()  # Record start time
    
    recov_data = np.nan_to_num(incomp_data, nan=recovery_value)
    
    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation miss net - Execution Time: {(end_time - start_time):.4f} seconds\n")
    
    # Imputation
    

    return recov_data
