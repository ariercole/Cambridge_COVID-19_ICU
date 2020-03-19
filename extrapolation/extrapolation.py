"""Extrapolation module."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from numpy.random import gamma, normal, rand
from sklearn import linear_model
from statsmodels.sandbox.regression.predstd import wls_prediction_std

mpl.rc('font', family = 'serif', size = 15)
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14


# Global preferences
n_obs    = 6  # UPDATE THIS
n_back   = 10-n_obs
n_future = 14
n_total  = n_back + n_obs + n_future


def load_data():
    daily_cases = pd.read_csv(os.path.join(os.pardir, 'model_data', 'uptodate_cases.csv'))
    prev_daily_cases = pd.read_csv(os.path.join(os.pardir, 'model_data', 'DailyConfirmedCases.csv'))
    prev_daily_cases = prev_daily_cases[42-n_back:42]  # Truncate to useful period
    prev_daily_cases = np.array(prev_daily_cases['CMODateCount'], dtype=np.float32)

    regions = daily_cases['region']
    dates = [f'{13+i-n_back}/03' for i in range(n_total)]

    X = sm.add_constant(np.arange(n_back+n_obs))
    X_pred = sm.add_constant(np.arange(n_total))

    daily_cases = np.array(daily_cases.iloc[:, 2:], dtype=np.float32)
    probs = (daily_cases / daily_cases.sum(axis=0)).mean(axis=1)
    daily_cases = np.concatenate((np.round(np.outer(probs, prev_daily_cases)), daily_cases), axis=1)

    return X, X_pred, daily_cases, regions, dates


def ci_plot(y, x_pred, y_pred, ci_l, ci_u,
            regions, dates, iter, ylabel='% ICU bed occupancy', obs=True, y_max=1000):
    ax = plt.subplot(331+iter)
    ax.grid(True)
    ax.set_xticks(np.arange(0, len(x_pred), 3))
    plt.plot(x_pred, y_pred, 'b-', label='Fit (OLS)')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]), np.concatenate([ci_l, ci_u[::-1]]),
             alpha=0.5, fc='b', ec='None', label='95% CI')
    if obs:
        plt.plot(np.arange(n_back), y[:n_back], 'b.', markersize=10, label='Estimated')
        plt.plot(np.arange(n_back, n_back+n_obs), y[n_back:n_back+n_obs], 'r.', markersize=10, label='Observed')
    plt.xlim(0, x_pred.max())
    plt.ylim(0, y_max)
    plt.xticks(range(0, n_total, 3), dates[::3]) if iter in [4, 5, 6] else ax.set_xticklabels([])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    if iter in [0, 3, 6]: plt.ylabel(ylabel)
    plt.title(regions[iter])
    plt.legend()


def _regional_prediction(X, X_pred, Y, log, i):
    y = Y[i]
    if log:
        y = np.log(y)
    mod = sm.OLS(y, X)
    res = mod.fit()
    y_pred = res.predict(X_pred)
    _, _, std_u = wls_prediction_std(res, exog=X_pred, alpha=1 - 0.6827)  # 1 s.d.
    _, ci_l, ci_u = wls_prediction_std(res, exog=X_pred, alpha=1 - 0.95)  # 95% CI

    return y_pred, std_u, ci_l, ci_u


def regional_predictions(X, X_pred, Y, log=True):
    x_true_list = []
    y_true_list = []
    x_pred_list = []
    y_pred_list = []
    ci_l_list = []
    ci_u_list = []
    avgs_list = []
    stds_list = []

    # parallelize model fitting
    parallel = Parallel(n_jobs=-1, prefer="threads")
    results = parallel(
        delayed(_regional_prediction)(
            X, X_pred, Y, log, i)
        for i in range(7))

    for i, (y_pred, std_u, ci_l, ci_u) in enumerate(results):  # 7 regions
        avgs_list += [y_pred]
        stds_list += [std_u - y_pred]
        if log:
            y_pred = np.exp(y_pred)
            ci_l = np.exp(ci_l)
            ci_u = np.exp(ci_u)
        x_true_list += [X[:, 1]]
        y_true_list += [Y[i]]
        x_pred_list += [X_pred[:, 1]]
        y_pred_list += [y_pred]
        ci_l_list += [ci_l]
        ci_u_list += [ci_u]

    return x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list, avgs_list, stds_list


def occupancy_arrays(log_means, log_stds, pct_need_icu, icu_delay_normal_loc=10.0, los_gamma_shape=8):
    log_means = np.stack(log_means)
    log_stds  = np.stack(log_stds)
    n_regions = log_means.shape[0]
    n_days    = log_means.shape[1]
    n_samples = 100
    arr = np.zeros((n_regions, n_days, n_samples))

    for k in range(n_samples):
        new_cases = np.exp(normal(log_means, log_stds))
        icu_cases = new_cases * pct_need_icu[:, np.newaxis]  # ICU cases = new cases each day * icu_per_case
        icu_cases = np.maximum(icu_cases, 1).astype(np.int32)

        for i in range(n_regions):
            for j in range(n_days):
                # Start
                delay_2_icu = normal(loc=icu_delay_normal_loc, scale=3.5, size=icu_cases[i, j])
                delay_2_icu = np.maximum(delay_2_icu, 0).astype(np.int32)
                icu_start = j + delay_2_icu

                # End
                los = gamma(shape=los_gamma_shape, scale=1.0, size=icu_cases[i, j])
                los = np.ceil(los).astype(np.int32)
                icu_end = icu_start + los

                for start, end in zip(icu_start, icu_end):
                    if start >= n_days:
                        continue
                    else:
                        arr[i, start:min(end, n_days), k] += 1

    return arr.mean(axis=2), arr.std(axis=2)


def main():
    delay = 10.0
    los = 8.0

    X, X_pred, new_cases, regions, dates = load_data()
    x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list, means, stds = regional_predictions(X, X_pred, Y=new_cases, log=False)
    x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list, log_means, log_stds = regional_predictions(X, X_pred, Y=new_cases, log=True)

    beds_info = pd.read_csv(os.path.join('model_data', 'ICU_beds_region.csv'))
    beds = beds_info['n_beds (2019)'].values

    death_and_icu_info = pd.read_csv(os.path.join('model_data', 'hospitalisation_and_fatalities.csv'))
    # cfr = death_and_icu_info['Mortality Rate']  # currently not using
    pct_need_icu = death_and_icu_info['Critical Care Needs Rate']

    mu, sig = occupancy_arrays(log_means, log_stds, pct_need_icu, icu_delay_normal_loc=delay, los_gamma_shape=los)

    fig = plt.figure(figsize=(15, 15))
    for i in range(len(regions)):
        ci_plot(new_cases[i], X_pred[:, 1], mu[i], mu[i]-1.96*sig[i], mu[i]+1.96*sig[i],
                regions, dates, i, ylabel='COVID-19 patients in ICU', obs=False)

    avg_occ = mu / beds[:, np.newaxis] * 100
    std_occ = sig / beds[:, np.newaxis] * 100
    fig = plt.figure(figsize=(15, 15))
    for i in range(len(regions)):
        ci_plot(new_cases[i], X_pred[:, 1], avg_occ[i], avg_occ[i]-1.96*std_occ[i], avg_occ[i]+1.96*std_occ[i],
                regions, dates, i, ylabel='% ICU occupancy', obs=False, y_max=100)

    plt.show()


if __name__ == "__main__":
    main()
