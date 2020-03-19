import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from length_of_stay_model.prob_hospitalisation import project_path, total_demographics, age_cats, per_region

mpl.rc('font', family = 'serif', size = 30)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 30

def plots(total_demographics, icu_beds, per_region):

    # demographics plot
    fig, (ax11, ax21) = plt.subplots(1, 2)
    fig.set_size_inches(28, 10)
    plt.subplots_adjust(left=0.05, wspace=0.3, right=0.95, top=0.92, bottom=0.05)
    ax12 = ax11.twinx()
    n = 7
    ind1 = np.arange(start=0, stop=n*3, step=3)  # the x locations for the groups
    ind2 = np.arange(start=1, stop=n*3+1, step=3)
    title_ind = np.arange(0.5, stop=n*3+0.5, step=3)
    width = 0.70  # the width of the bars: can also be len(x) sequence
    cumultative_total = 0
    total_demographics.sort_values('Region', inplace=True)
    regions = ['EoE', 'London', 'Midlands', 'NE&Yorks.', 'NW', 'SE', 'SW']
    colors = ['red', 'hotpink', 'orange', 'gold', 'limegreen', 'darkgreen', 'darkblue', 'dodgerblue', 'mediumorchid']
    plots = []

    for i in range(9):
        millions = total_demographics.iloc[:, i] / 1000000  # convert to millions
        plots.append(ax11.bar(ind1, millions, width, bottom=cumultative_total, color=colors[i]))
        cumultative_total += millions

    plots.append(ax12.bar(ind2, icu_beds['n_beds (2019)']*100000/total_demographics.sum(axis=1).values, width=width, color='dimgray'))
    ax11.set_ylabel('Population (millions)')
    ax12.set_ylabel('ICU beds per 100,000')
    plt.title('Demographics', y=1.02)
    plt.xticks(title_ind, regions, rotation='vertical')
    ax11.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax12.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax11.set_ylim((0, 12.5))
    ax12.set_ylim((0, 12.5))
    ax11.legend(age_cats, loc='upper right', prop={'size': 22})

    # critical care needs plot
    ax22 = ax21.twinx()
    plots.append(ax21.bar(ind1, per_region['Critical Care Needs Rate'] * 100, width=width, color='lightblue'))
    plots.append(ax21.bar(ind1, per_region['Mortality Rate'] * 100 , width=width, color='dodgerblue'))
    plots.append(ax22.bar(ind2, icu_beds['n_beds (2019)'] * 100000 / total_demographics.sum(axis=1).values, width=width, color='dimgray'))
    ax21.set_ylabel('Percentage of cases requiring ICU')
    ax22.set_ylabel('ICU beds per 100,000')
    plt.title('Critical Care Demand Per Case', y=1.02)
    plt.xticks(title_ind, regions, rotation='vertical')
    ax21.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax22.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax21.set_ylim((0, 3.63))
    ax22.set_ylim((0, 12.5))
    ax21.legend(['Survivors', 'Non-Survivors'], loc='upper right', prop={'size': 22})
    plt.savefig(project_path + '/figs/demographics_cc_icu.png', dpi=300)

    return


icu_beds = pd.read_csv(project_path + '/model_data/ICU_beds_region.csv')
plots(total_demographics, icu_beds, per_region)