"""Graphs file."""

import dash_core_components as dcc
import datetime
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import os
import shutil
import statsmodels.api as sm

from dateutil import parser
from joblib import Parallel, delayed
from numpy.random import gamma, lognormal, normal
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from tqdm import tqdm


# Global variables
n_obs    = 11  # UPDATE THIS
n_future = 14
n_total  = n_obs + n_future


def load_data():
    daily_cases = pd.read_csv(os.path.join(os.pardir, 'data', 'model', 'uptodate_cases.csv'))

    regions = daily_cases['region']
    dates = [f'{13+i}/03' for i in range(n_total)]
    dates = dates[:19] + [f'{i}/04' for i in range(1, n_total-18)]

    X = sm.add_constant(np.arange(n_obs))
    X_pred = sm.add_constant(np.arange(n_total))

    cum_cases = np.array(daily_cases.iloc[:, 2:], dtype=np.float32)

    return X, X_pred, cum_cases, regions, dates


def _regional_prediction(X, X_pred, Y, i):
    mod = sm.OLS(np.log(Y[i]), X)
    res = mod.fit()

    y_pred = res.predict(X_pred)
    _, _, std_u = wls_prediction_std(res, exog=X_pred, alpha=1-0.6827)  # 1 s.d.
    _, ci_l, ci_u = wls_prediction_std(res, exog=X_pred, alpha=1-0.95)  # 95% CI

    return y_pred, std_u, ci_l, ci_u, res.params[1]


def regional_predictions(X, X_pred, Y):
    x_true_list = []
    y_true_list = []
    x_pred_list = []
    y_pred_list = []
    ci_l_list = []
    ci_u_list = []
    avgs_list = []
    stds_list = []
    exponent_list = []

    # Parallelize model fitting
    parallel = Parallel(n_jobs=-1, prefer="threads")
    results = parallel(delayed(_regional_prediction)(X, X_pred, Y, i) for i in range(7))

    for i, (y_pred, std_u, ci_l, ci_u, exponent) in enumerate(results):  # 7 regions
        avgs_list += [y_pred]
        stds_list += [std_u - y_pred]

        # Log
        y_pred = np.exp(y_pred)
        ci_l = np.exp(ci_l)
        ci_u = np.exp(ci_u)

        x_true_list += [X[:, 1]]
        y_true_list += [Y[i]]
        x_pred_list += [X_pred[:, 1]]
        y_pred_list += [y_pred]
        ci_l_list += [ci_l]
        ci_u_list += [ci_u]
        exponent_list += [exponent]

    return x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list, avgs_list, stds_list, exponent_list


def occupancy_arrays(means, stds, exponents, pct_need_icu,
                     icu_delay_normal_loc=2.0, los_gamma_shape=8.0, log=True):
    means = np.stack(means)
    stds  = np.stack(stds)
    n_regions = means.shape[0]
    n_days    = means.shape[1]
    n_samples = 500
    arr = np.zeros((n_regions, n_days, n_samples))

    for k in range(n_samples):
        if log:
            new_cases = exponents[:, np.newaxis] * lognormal(means, stds)
        else:
            new_cases = normal(means, stds)
        icu_cases = pct_need_icu[:, np.newaxis] * new_cases  # ICU cases = new cases each day * icu_per_case
        icu_cases = np.maximum(icu_cases, 1).astype(np.int32)

        for i in range(n_regions):
            for j in range(n_days):
                # Start
                delay_2_icu = normal(loc=icu_delay_normal_loc, scale=3.5, size=icu_cases[i, j])
                delay_2_icu = delay_2_icu.round().astype(np.int32)

                # End
                los = gamma(shape=los_gamma_shape, scale=1.0, size=icu_cases[i, j])
                los = np.maximum(los, 1).astype(np.int32)

                # Indices
                start_inds = j + delay_2_icu
                end_inds   = np.minimum(start_inds + los, n_days-1)
                start_inds = np.maximum(start_inds, 0)

                for start, end in zip(start_inds, end_inds):
                    if start >= n_days:
                        continue
                    else:
                        arr[i, start:end+1, k] += 1

    return arr.mean(axis=2), arr.std(axis=2)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def get_new_dates(dates):
    min_date_str = dates[0].split("/")[1] + "/" + dates[0].split("/")[0]
    min_date = parser.parse(min_date_str)
    today = datetime.datetime.today()
    max_date = today + datetime.timedelta(days=n_future+1)
    new_dates = []
    today_idx = None
    for i, date in enumerate(daterange(min_date, max_date)):
        if today.strftime("%d/%m") == date.strftime("%d/%m"):
            today_idx = i
        new_dates.append(date.strftime("%d/%m"))
    return new_dates, today_idx


def _make_fig(x_true, y_true, x_pred, y_pred, ci_l, ci_u,
              title, ylabel, obs, dates):
    if "patients" in ylabel:
        y_max = max(y_pred)
    elif "occupancy" in ylabel:
        y_max = min(100, max(y_pred))
    else:
        pass

    color = "blue"
    trace0 = go.Scatter(
        x=x_pred,
        y=ci_l.round(2),
        line=dict(color=color),
        name="95% CI",
        showlegend=False
    )
    trace1 = go.Scatter(
        x=x_pred,
        y=ci_u.round(2),
        fill='tonexty',
        name="95% CI",
        line=dict(color=color)
    )
    if obs:
        trace2 = go.Scatter(
            x=x_true,
            y=y_true.round(2),
            mode='markers',
            name='Recorded',
            line=dict(color="red")
        )
    trace3 = go.Scatter(
        x=x_pred,
        y=y_pred.round(2),
        mode='lines+markers',
        name='Mean',
        line=dict(color=color, width=2),
    )
    if obs:
        traces = [trace0, trace1, trace3, trace2]
    else:
        traces = [trace0, trace1, trace3]
    tickvals = list(range(0, len(dates), 3))
    # print_dates = [d if i % 3 == 0 else '' for i, d in enumerate(dates)]
    print_dates = [d for i, d in enumerate(dates) if i % 3 == 0]
    fig=dict(
        data=traces,
        layout=dict(
            showlegend=True,
            title=title,
            yaxis_title=ylabel,
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=print_dates,
                showgrid=True,
                range=[min(x_true), max(x_pred)],
            ),
            yaxis=dict(
                title=ylabel,
                range=[0, max(y_pred)]
            ),
        )
    )

    return fig


def _make_figs(x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list,
               dates, regions, title, ylabel, obs=True):
    new_dates, today_idx = get_new_dates(dates)

    # Parallelize figure making
    parallel = Parallel(n_jobs=-1, prefer="threads")
    fig_list = parallel(
        delayed(_make_fig)(
            x_true, y_true, x_pred, y_pred, ci_l, ci_u, title, ylabel, obs, new_dates)
        for x_true, y_true, x_pred, y_pred, ci_l, ci_u in zip(
            x_true_list, y_true_list, x_pred_list, y_pred_list, ci_l_list, ci_u_list))

    return regions, fig_list, today_idx


def patients_update(plot_dict):
    _, _, _, regions, dates = load_data()
    x_true, y_true, x_pred, y_pred, ci_l, ci_u = plot_dict["new_patients_loglinear_fit"]
    regions, fig_list_logged, today_idx = _make_figs(x_true, y_true, x_pred, y_pred, ci_l, ci_u,
                                          dates, regions, "Cumulative COVID-19 patients", "Number of patients")

    return regions, fig_list_logged, get_new_dates(dates)


def occupancy_update(plot_dict, delay=10, los=8, obs=True):
    _, _, _, regions, dates = load_data()

    x_true, y_true, x_pred, y_pred, ci_l, ci_u = plot_dict["icu_patients"][delay][los]
    regions, icu_fig_list, today_idx = _make_figs(x_true, y_true, x_pred, y_pred, ci_l, ci_u,
                                                  dates, regions, "ICU patients", "#patients", obs=False)

    x_true, y_true, x_pred, y_pred, ci_l, ci_u = plot_dict["icu_occupancy"][delay][los]
    regions, occ_fig_list, today_idx = _make_figs(x_true, y_true, x_pred, y_pred, ci_l, ci_u,
                                                  dates, regions, "% of ICU Bed Occupancy", "% occupancy", obs=False)

    return regions, icu_fig_list, occ_fig_list, get_new_dates(dates)


def save_dict_safely(dict_, f=os.path.join('data', 'plot_dict.pkl')):
    if os.path.exists(f):
        os.remove(f)
    with open(f, 'wb') as fp:
        pickle.dump(dict_, fp)


def load_dict_safely(f=os.path.join('data', 'plot_dict.pkl')):
    with open(f, 'rb') as fp:
        dict_ = pickle.load(fp)
    return dict_


def update_backend(icu_delay_normal_locs=list(range(1, 11)), los_gamma_shapes=list(range(3, 12))):
    print('Updating backend dictionary, this may take a while...')
    print('Loading data...')
    X, X_pred, Y, _, _ = load_data()
    death_and_icu_info = pd.read_csv(os.path.join(os.pardir, 'data', 'model', 'hospitalisation_and_fatalities.csv'))
    pct_need_icu = death_and_icu_info['Critical Care Needs Rate']
    beds_info = pd.read_csv(os.path.join(os.pardir, 'data', 'model', 'ICU_beds_region.csv'))
    beds = beds_info['n_beds (2019)'].values

    """
    Construct large dictionary to be indexed by the web user.
    big_dict:
        new_patients_loglinear_fit:     plot_tuple
        icu_patients:                   icu_patients_dict
        icu_occupancy:                  icu_occupancy_dict

    icu_patients_dict:
        delay: default = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            los: default = [3, 4, 5, 6, 7, 8, 9, 10, 11]

    Example indexing:
    - big_dict['new_patients_loglinear_fit'] --> return plot_tuple
    - big_dict['icu_occupancy'][10][8]   --> return plot_tuple
    """
    big_dict = {}

    # Update new patient
    print('Updating estimated new patients (LOG-LINEAR)...')
    x_true, y_true, x_pred, y_pred, ci_l, ci_u, log_means, log_stds, exponents = regional_predictions(X, X_pred, Y)
    big_dict['new_patients_loglinear_fit'] = x_true, y_true, x_pred, y_pred, ci_l, ci_u

    # Update ICU patient
    print('\nUpdating ICU patients info...')
    delay_dict = {}
    for delay in icu_delay_normal_locs:
        los_dict = {}
        for los in tqdm(los_gamma_shapes):
            mu, sig = occupancy_arrays(log_means, log_stds, np.array(exponents), pct_need_icu,
                icu_delay_normal_loc=delay, los_gamma_shape=los)
            ci_l = [np.maximum(mu[i] - 1.96 * sig[i], 0) for i in range(7)]
            ci_u = [mu[i] + 1.96 * sig[i] for i in range(7)]

            los_dict[los] = (x_true, y_true, x_pred, mu, ci_l, ci_u)
        delay_dict[delay] = los_dict
    big_dict['icu_patients'] = delay_dict

    # Update ICU occupancy
    print('\nUpdating ICU occupancy info...')
    delay_dict = {}
    for delay in icu_delay_normal_locs:
        los_dict = {}
        for los in tqdm(los_gamma_shapes):
            _, _, _, mu, ci_l, ci_u = big_dict['icu_patients'][delay][los]

            avg_occ = [100 * mu[i] / beds[i] for i in range(7)]
            ci_l = [np.maximum(100 * ci_l[i] / beds[i], 0) for i in range(7)]
            ci_u = [100 * ci_u[i] / beds[i] for i in range(7)]

            los_dict[los] = (x_true, y_true, x_pred, avg_occ, ci_l, ci_u)
        delay_dict[delay] = los_dict
    big_dict['icu_occupancy'] = delay_dict

    # Save
    print('Saving big dictionary to file with pickle...')
    save_dict_safely(big_dict)


def choroplet_plot(plot_dict, geo_data, geo_df, today=None, delay=2, los=8):
    _, _, _, y_pred, _, _ = plot_dict["icu_occupancy"][delay][los]
    if today is None:
        today = n_obs
    # Index regions `today`
    y_pred = np.array(y_pred)
    geo_df['% additional demand'] = y_pred[[3, 4, 3, 2, 1, 2, 0, 5, 6], today].round(2)

    fig = px.choropleth_mapbox(
        geo_df,
        geojson=geo_data,
        locations='ID',
        color='% additional demand',
        color_continuous_scale="Portland",
        featureidkey="properties.nuts118cd",
        range_color=(0, 100),
        mapbox_style="carto-positron",
        hover_data=["Region", "% additional demand"],
        zoom=4.7,
        center={"lat": 53, "lon": -2},
        opacity=0.7
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


if __name__ == "__main__":
    update_backend(icu_delay_normal_locs=list(range(1, 11)), los_gamma_shapes=list(range(3, 12)))
