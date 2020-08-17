from utils.ontf import Online_NTF
import numpy as np
from sklearn.decomposition import SparseCoder
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

DEBUG = False


def read_data_JHU_countrywise(path):
    '''
    Read input time series as a narray
    '''
    data_full = pd.read_csv(path, delimiter=',').T
    data = data_full.values[1:, :]
    data = np.delete(data, [1, 2], 0)  # delete lattitue & altitude
    country_list = [i for i in set(data[0, :])]
    country_list = sorted(country_list)  # whole countries in alphabetical order

    ### merge data according to country
    data_new = np.zeros(shape=(data.shape[0] - 1, len(country_list)))
    for i in np.arange(len(country_list)):
        idx = np.where(data[0, :] == country_list[i])
        data_sub = data[1:, idx]
        data_sub = data_sub[:, 0, :]
        data_sub = np.sum(data_sub, axis=1)
        data_new[:, i] = data_sub
    data_new = data_new.astype(int)

    idx = np.where(data_new[-1, :] > 1000)
    data_new = data_new[:, idx]
    data_new = data_new[:, 0, :]
    # data_new[:,1] = np.zeros(data_new.shape[0])
    print('data_new', data_new)
    country_list = [country_list[idx[0][i]] for i in range(len(idx[0]))]
    print('country_list', country_list)

    data_new = np.diff(data_new, axis=0)

    return data_new.T,

def read_data_COVIDactnow_NYT():
    '''
    Read input time series data as a dictionary of pandas dataframe
    Get Hospotal related data from COVIDACTNOW and cases and deaths from NYT
    '''

    data_ACT = pd.read_csv("Data/states.NO_INTERVENTION.timeseries.csv", delimiter=',')
    data_NYT = pd.read_csv("Data/NYT_us-states.csv", delimiter=',')
    df = {}

    state_list = sorted([i for i in set([i for i in data_ACT['stateName']])])

    ### Find maximum earliest and the minimum latest date of both data
    start_dates = []
    end_dates = []
    for state in state_list:
        df1 = data_ACT.loc[data_ACT['stateName'] == state]
        df2 = data_NYT.loc[data_NYT['state'] == state]
        start_dates.append(min(df1['date']))
        start_dates.append(min(df2['date']))
        end_dates.append(max(df1['date']))
        end_dates.append(max(df2['date']))

    max_min_date = max(start_dates)
    min_max_date = min(end_dates)
    # print('!!! min_dates', max_min_date)

    for state in state_list:
        df1 = data_ACT.loc[data_ACT['stateName'] == state].set_index('date')
        lastUpdatedDate = df1['lastUpdatedDate'].iloc[0]
        df1 = df1[max_min_date:min(lastUpdatedDate, min_max_date)]
        df1['input_hospitalBedsRequired'] = df1['hospitalBedsRequired']
        df1['input_ICUBedsInUse'] = df1['ICUBedsInUse']
        df1['input_ventilatorsInUse'] = df1['ventilatorsInUse']

        df2 = data_NYT.loc[data_NYT['state'] == state].set_index('date')
        df1['input_Deaths'] = df2['deaths']
        # print('!!! df_NYT1', df2['deaths'])
        df1['input_Infected'] = df2['cases']
        # print('!!! df_NYT1_cases', df2['cases'])

        df1 = df1.fillna(0)

        print('!!! If any value is NAN:', df1.isnull().values.any())

        df.update({state: df1})

    return df

def read_data_COVIDtrackingProject(if_moving_avg_data=False, if_log_scale=False):
    '''
    Read input time series data as a dictionary of pandas dataframe
    '''
    path = "Data/us_states_COVID_tracking_project.csv"
    data = pd.read_csv(path, delimiter=',').sort_values(by="date")
    ### Convert the format of dates from string to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', utc=False)

    df = {}

    ### Use full state names
    state_list = sorted([i for i in set([i for i in data['state']])])

    ### Find maximum earliest and the minimum latest date of both data
    start_dates = []
    end_dates = []
    for state in state_list:
        df1 = data.loc[data['state'] == state]
        start_dates.append(min(df1['date']).strftime("%Y-%m-%d"))
        end_dates.append(max(df1['date']).strftime("%Y-%m-%d"))
        # print('State %s and end_date %s' % (state, max(df1['date']).strftime("%Y-%m-%d")))
    max_min_date = max(start_dates)
    min_max_date = min(end_dates)

    print('!!! max_min_date', max_min_date)
    print('!!! min_max_date', min_max_date)


    original_list_variables = data.keys().tolist()
    original_list_variables.remove('date')


    for state in state_list:
        df1 = data.loc[data['state'] == state].set_index('date')
        # lastUpdatedDate = df1['lastUpdateEt'].iloc[0]
        df1 = df1[max_min_date:min_max_date]
        ### making new columns to process columns of interest and preserve the original data
        df1['input_hospitalized_Currently'] = df1['hospitalizedCurrently']
        df1['input_inICU_Currently'] = df1['inIcuCurrently']
        df1['input_daily_test_positive_rate'] = df1['positive'].diff() / df1['totalTestResults'].diff()
        df1['input_daily_cases'] = df1['positive'].diff()
        df1['input_daily_deaths'] = df1['death'].diff()
        df1['input_daily_cases_pct_change'] = df1['positive'].pct_change()

        # print('!!! If any value is NAN: %r for state %s:' % (df1.isnull().values.any(), state))
        df.update({abbrev_us_state[state]: df1})

    """
    for variable in original_list_variables:
        for state in state_list:
            df1 = data.loc[data['state'] == state].set_index('date')
            if not df1[variable].isnull().values.any():
                df.update({'list_states_observed_' + variable: abbrev_us_state[state]})
    """

    return df

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))

def main():

    data_source_list = ['COVID_ACT_NOW', 'COVID_TRACKING_PROJECT', 'JHU']
    data_source = data_source_list[0]

    # df = read_data_COVIDactnow_NYT()
    # print('!!! df', df)

if __name__ == '__main__':
    main()
