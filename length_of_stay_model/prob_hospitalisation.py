import pandas as pd

project_path = '/Users/emmarocheteau/PycharmProjects/CoViD-19_ICU'

def get_age():
    age_cats = ['<10', '10-20', '20-30', '30-40','40-50', '50-60', '60-70', '70-80', '80+']
    # data taken from Table 1: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
    prob_hosp = [0.1, 0.3, 1.2, 3.2, 4.9, 10.2, 16.6, 24.3, 27.3]  # percentage requiring hospitalisation
    prob_critical_care_if_hosp = [5.0, 5.0, 5.0, 5.0, 6.3, 12.2, 27.4, 43.2, 70.9]  # percentage needing critical care among those hospitalised
    prob_death = [0.002, 0.006, 0.03, 0.08, 0.15, 0.6, 2.2, 5.1, 9.3]  # mortality rate
    age = pd.DataFrame([prob_hosp, prob_critical_care_if_hosp, prob_death], columns=age_cats, index=['hosp', 'crit_care_if_hosp', 'death'])
    age /= 100  # convert percentages to probs
    return age, age_cats

def get_gender():
    # we need to take into account the extra risk of being male
    # I am assuming the extra risk applies on a univariate basis and equally to hospitalisations and deaths
    gender_cats = ['Male', 'Female']
    prob_death_gender = [[2.8, 1.7]]  # data taken from Wuhan on gender difference in mortality
    prob_death_total = 2.8*0.5117 + 1.7*0.4883  # males make up 51.17% of population China according to 2017 Census (http://www.stats.gov.cn/tjsj/ndsj/2018/indexeh.htm)
    gender = pd.DataFrame(prob_death_gender, columns=gender_cats)
    return gender, prob_death_total

def get_age_brackets(lower, upper):
    cats = []
    for i in range(lower, upper):
        cats.append(str(i))
    if lower == 80:
        cats.append('90+')
    return cats

def get_age_cat_demographics(age_cats, df):
    demographics_age = pd.DataFrame([], columns=age_cats)
    demographics_age['Area Codes'] = df['Area Codes ']
    demographics_age['Region'] = df['Unnamed: 1']
    for i, age_cat in enumerate(age_cats):
        demographics_age[age_cat] = df[get_age_brackets(i*10, i*10+10)].astype(int).sum(axis=1)
    return demographics_age

age, age_cats = get_age()
gender, prob_death_total = get_gender()
# prob critical care = prob hospitalisation x prob critical care if hospitalised
prob_critical_care_by_age_gender = {}
prob_critical_care_by_age_gender['Male'] = age.loc['hosp']*age.loc['crit_care_if_hosp']*(gender['Male'].values[0]/prob_death_total)
prob_critical_care_by_age_gender['Female'] = age.loc['hosp']*age.loc['crit_care_if_hosp']*(gender['Female'].values[0]/prob_death_total)
prob_death_by_age_gender = {}
prob_death_by_age_gender['Male'] = age.loc['death']*(gender['Male'].values[0]/prob_death_total)
prob_death_by_age_gender['Female'] = age.loc['death']*(gender['Female'].values[0]/prob_death_total)

female = pd.read_csv('{}female_age_demographics.csv'.format(project_path + '/background_data/'), thousands=',')
male = pd.read_csv('{}male_age_demographics.csv'.format(project_path + '/background_data/'), thousands=',')

male_demographics = get_age_cat_demographics(age_cats, male)
female_demographics = get_age_cat_demographics(age_cats, female)

male_deaths = male_demographics[age_cats].mul(prob_death_by_age_gender['Male'].values, axis=1)
female_deaths = female_demographics[age_cats].mul(prob_death_by_age_gender['Female'].values, axis=1)

male_critical_care = male_demographics[age_cats].mul(prob_critical_care_by_age_gender['Male'].values, axis=1)
female_critical_care = female_demographics[age_cats].mul(prob_critical_care_by_age_gender['Female'].values, axis=1)

total_deaths = male_deaths + female_deaths
total_critical_care = male_critical_care + female_critical_care
total_demographics = male_demographics + female_demographics

per_region = pd.DataFrame(total_deaths.sum(axis=1), columns=['Deaths'])
per_region['Critical Beds Required'] = total_critical_care.sum(axis=1)
per_region['Area Codes'] = male_demographics['Area Codes']
per_region['Region'] = male_demographics['Region']
per_region['Mortality Rate'] = per_region['Deaths']/(male['All Ages'] + female['All Ages'])
per_region['Critical Care Needs Rate'] = per_region['Critical Beds Required']/(male['All Ages'] + female['All Ages'])
per_region.drop(columns=['Deaths', 'Critical Beds Required'], inplace=True)
per_region.sort_values('Region', inplace=True)
per_region.to_csv(project_path + '/model_data/hospitalisation_and_fatalities.csv')