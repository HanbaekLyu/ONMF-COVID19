from utils.ontf import Online_NTF
import numpy as np
from sklearn.decomposition import SparseCoder
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
import matplotlib.font_manager as font_manager
import covid_dataprocess
import itertools

DEBUG = False


class ONMF_timeseries_reconstructor():
    def __init__(self,
                 path,
                 source,
                 data_source,
                 country_list=None,
                 state_list=None,
                 state_list_train=None,
                 n_components=100,  # number of dictionary elements -- rank
                 ONMF_iterations=50,  # number of iterations for the ONMF algorithm
                 ONMF_sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
                 ONMF_batch_size=20,  # number of patches used in i.i.d. subsampling
                 num_patches_perbatch=1000,  # number of patches that ONTF algorithm learns from at each iteration
                 patch_size=7,  # length of sliding window
                 patches_file='',
                 learn_joint_dict=False,
                 prediction_length=1,
                 learnevery=5,
                 learning_window_cap=None,  # if not none, learn from the past "learning_window_cap" days to predict
                 alpha=None,
                 beta=None,
                 subsample=False,
                 if_covidactnow=False,
                 if_onlynewcases=False,
                 if_moving_avg_data=False,
                 if_log_scale=False):
        '''
        batch_size = number of patches used for training dictionaries per ONTF iteration
        sources: array of filenames to make patches out of
        patches_array_filename: numpy array file which contains already read-in images
        '''
        self.path = path
        self.source = source
        self.data_source = data_source
        self.state_list = state_list
        self.country_list = country_list
        self.n_components = n_components
        self.ONMF_iterations = ONMF_iterations
        self.ONMF_sub_iterations = ONMF_sub_iterations
        self.num_patches_perbatch = num_patches_perbatch
        self.ONMF_batch_size = ONMF_batch_size
        self.patch_size = patch_size
        self.patches_file = patches_file
        self.learn_joint_dict = learn_joint_dict
        self.prediction_length = prediction_length
        self.code = np.zeros(shape=(n_components, num_patches_perbatch))
        self.learnevery = learnevery
        self.alpha = alpha
        self.beta = beta
        self.subsample = subsample
        self.if_onlynewcases = if_onlynewcases
        self.if_moving_avg_data = if_moving_avg_data
        self.if_log_scale = if_log_scale
        self.input_variable_list = []
        self.result_dict = {}
        self.state_list_train = state_list_train
        self.learning_window_cap = learning_window_cap

        input_variable_list = []

        if data_source == 'COVID_ACT_NOW':
            print('LOADING.. COVID_ACT_NOW')
            self.input_variable_list = ['input_hospitalBedsRequired',
                                        'input_ICUBedsInUse',
                                        'input_ventilatorsInUse',
                                        'input_Deaths',
                                        'input_Infected']
            self.df = covid_dataprocess.read_data_COVIDactnow_NYT()
            self.df = self.truncate_NAN_DataFrame()
            self.df = self.moving_avg_log_scale()
            self.data = self.extract_ndarray_from_DataFrame()
            self.result_dict.update({'Data source': 'COVID_ACT_NOW'})
            self.result_dict.update({'Full DataFrame': self.df})
            self.result_dict.update({'Data array': self.data})
            self.result_dict.update({'List_states': self.state_list})
            self.result_dict.update({'List_variables': self.input_variable_list})



        elif data_source == 'COVID_TRACKING_PROJECT':
            print('LOADING.. COVID_TRACKING_PROJECT')
            self.input_variable_list = ['input_hospitalized_Currently',
                                        'input_inICU_Currently',
                                        'input_daily_test_positive_rate',
                                        'input_daily_cases',
                                        'input_daily_deaths']
            # 'input_daily_cases_pct_change']

            self.df = covid_dataprocess.read_data_COVIDtrackingProject()
            self.df = self.truncate_NAN_DataFrame()
            self.df = self.moving_avg_log_scale()
            self.data = self.extract_ndarray_from_DataFrame()
            # print('!!! df', self.df.get('Florida'))
            self.result_dict.update({'Full DataFrame': self.df})
            self.result_dict.update({'Data array': self.data})
            self.result_dict.update({'List_states': self.state_list})
            self.result_dict.update({'List_variables': self.input_variable_list})

            if state_list_train is not None:
                print('LOADING.. COVID_TRACKING_PROJECT for training set')
                self.df_train = covid_dataprocess.read_data_COVIDtrackingProject()
                self.df_train = self.truncate_NAN_DataFrame()
                self.df_train = self.moving_avg_log_scale()
                self.data_train = self.extract_ndarray_from_DataFrame()
                self.result_dict.update({'Full DataFrame (train)': self.df_train})
                self.result_dict.update({'Data array (train)': self.data_train})
                self.result_dict.update({'List_states (train)': state_list_train})


        else:  ### JHU data
            print('LOADING.. JHU Data')
            self.data = self.combine_data(self.source)

        print('data', self.data)
        print('data.shape', self.data.shape)
        self.ntf = Online_NTF(self.data, self.n_components,
                              iterations=self.ONMF_sub_iterations,
                              learn_joint_dict=True,
                              mode=3,
                              ini_dict=None,
                              ini_A=None,
                              ini_B=None,
                              batch_size=self.ONMF_batch_size)

        self.W = np.zeros(shape=(self.data.shape[0] * self.data.shape[2] * patch_size, n_components))

    def moving_avg_log_scale(self):
        df = self.df
        if self.if_moving_avg_data:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = df2.rolling(window=5, win_type=None).sum() / 5  ### moving average with backward window size 5
                df2 = df2.fillna(0)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        if self.if_log_scale:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = np.log(df2 + 1)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        return df

    def truncate_NAN_DataFrame(self):
        df = self.df.copy()
        ### Take the maximal sub-dataframe that does not contain NAN
        ### If some state has all NANs for some variable, that variable is dropped from input_list_variable
        start_dates = []
        end_dates = []

        input_variable_list_noNAN = self.input_variable_list.copy()
        for column in input_variable_list_noNAN:
            for state in self.state_list:
                df1 = df.get(state)
                if df1[column].isnull().all():
                    input_variable_list_noNAN.remove(column)
        self.input_variable_list = input_variable_list_noNAN
        print('!!! New input_variable_list', self.input_variable_list)

        for state in self.state_list:
            df1 = df.get(state)
            for column in self.input_variable_list:
                l_min = df1[column][df1[column].notnull()].index[0]
                l_max = df1[column][df1[column].notnull()].index[-1]
                start_dates.append(l_min)
                end_dates.append(l_max)

        max_min_date = max(start_dates)
        min_max_date = min(end_dates)

        for state in self.state_list:
            df1 = df.get(state)
            df1 = df1[max_min_date:min_max_date]
            print('!!! If any value is NAN:', df1.isnull())
            df.update({state: df1})
        return df

    def extract_ndarray_from_DataFrame(self):
        ## Make numpy array of shape States x Days x variables
        data_combined = []
        df = self.df
        if self.state_list == None:
            self.state_list = sorted([i for i in set(df.keys())])

        for state in self.state_list:
            df1 = df.get(state)

            if state == self.state_list[0]:
                data_combined = df1[self.input_variable_list].values  ## shape Days x variables
                data_combined = np.expand_dims(data_combined, axis=0)
                print('!!!Data_combined.shape', data_combined.shape)
            else:
                data_new = df1[self.input_variable_list].values  ## shape Days x variables
                data_new = np.expand_dims(data_new, axis=0)
                print('!!! Data_new.shape', data_new.shape)
                data_combined = np.append(data_combined, data_new, axis=0)

        data_combined = np.nan_to_num(data_combined, copy=True, nan=0, posinf=1, neginf=0)
        return data_combined

    def read_data_as_array_countrywise(self, path):
        '''
        Read input time series as a narray
        '''
        data_full = pd.read_csv(path, delimiter=',').T
        data = data_full.values[1:, :]
        data = np.delete(data, [1, 2], 0)  # delete lattitue & altitude
        if self.country_list == None:
            country_list = [i for i in set(data[0, :])]
            country_list = sorted(country_list)  # whole countries in alphabetical order
        else:
            country_list = self.country_list

        ### merge data according to country
        data_new = np.zeros(shape=(data.shape[0] - 1, len(country_list)))
        for i in np.arange(len(country_list)):
            idx = np.where(data[0, :] == country_list[i])
            data_sub = data[1:, idx]
            data_sub = data_sub[:, 0, :]
            data_sub = np.sum(data_sub, axis=1)
            data_new[:, i] = data_sub
        data_new = data_new.astype(int)

        if self.country_list == None:
            idx = np.where(data_new[-1, :] > 1000)
            data_new = data_new[:, idx]
            data_new = data_new[:, 0, :]
            # data_new[:,1] = np.zeros(data_new.shape[0])
            print('data_new', data_new)
            country_list = [country_list[idx[0][i]] for i in range(len(idx[0]))]
            print('country_list', country_list)

        if self.if_onlynewcases:
            data_new = np.diff(data_new, axis=0)

        if self.if_moving_avg_data:
            for i in np.arange(5, data_new.T.shape[1]):
                data_new.T[:, i] = (data_new.T[:, i] + data_new.T[:, i - 1] + data_new.T[:, i - 2] + data_new.T[:,
                                                                                                     i - 3] + data_new.T[
                                                                                                              :,
                                                                                                              i - 4]) / 5
                # A_recons[:, i] = (A_recons[:, i] + A_recons[:, i-1]) / 2

        if self.if_log_scale:
            data_new = np.log(data_new + 1)

        return data_new.T, country_list

    def read_data_COVIDactnow(self, path, use_NYT_cases=False):
        '''
        Read input time series data as a dictionary of pandas dataframe
        COVIDACTNOW is a SYNTHETIC data!!!
        That's why the cases and deaths are off from the real data, expecially for the NO_INTERVENTION case
        '''
        data = pd.read_csv(path, delimiter=',')
        df = {}
        data_NYT = pd.read_csv("Data/NYT_us-states.csv", delimiter=',')

        if self.state_list == None:
            self.state_list = sorted([i for i in set([i for i in data['stateName']])])

        ### Find earliest starting date of the data
        start_dates = []
        for state in self.state_list:
            df1 = data.loc[data['stateName'] == state]
            start_dates.append(min(df1['date']))
        max_min_date = max(start_dates)
        # print('!!! min_dates', max_min_date)

        for state in self.state_list:
            df1 = data.loc[data['stateName'] == state].set_index('date')
            lastUpdatedDate = df1['lastUpdatedDate'].iloc[0]
            df1 = df1[max_min_date:lastUpdatedDate]
            df1['input_hospitalBedsRequired'] = df1['hospitalBedsRequired']
            df1['input_ICUBedsInUse'] = df1['ICUBedsInUse']
            df1['input_ventilatorsInUse'] = df1['ventilatorsInUse']
            if not use_NYT_cases:
                df1['input_Deaths'] = df1['cumulativeDeaths']
                df1['input_Infected'] = df1['cumulativeInfected']
            else:
                df_NYT1 = data_NYT.loc[data_NYT['state'] == state].set_index('date')
                df1['input_Deaths'] = df_NYT1['deaths']
                print('!!! df_NYT1', df_NYT1['deaths'])
                df1['input_Infected'] = df_NYT1['cases']
                print('!!! df_NYT1_cases', df_NYT1['cases'])

            ### Take the maximal sub-dataframe that does not contain NAN
            max_index = []
            for column in self.input_variable_list:
                l = df1[column][df1[column].notnull()].index[-1]
                max_index.append(l)
            max_index = min(max_index)
            print('!!! max_index', max_index)
            df1 = df1[:max_index]

            print('!!! If any value is NAN:', df1.isnull())
            df.update({state: df1})

        if self.if_onlynewcases:
            for state in self.state_list:
                df1 = df.get(state)
                # df1[input_variable_list] contains 153 rows and 5 columns
                df1['input_Infected'] = df1['input_Infected'].diff()
                df1['input_Deaths'] = df1['input_Deaths'].diff()
                df1 = df1.fillna(0)
                df.update({state: df1})

        if self.if_moving_avg_data:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = df2.rolling(window=5, win_type=None).sum() / 5  ### moving average with backward window size 5
                df2 = df2.fillna(0)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        if self.if_log_scale:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = np.log(df2 + 1)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        self.df = df

        ## Make numpy array of shape States x Days x variables
        data_combined = []
        for state in self.state_list:
            df1 = df.get(state)

            if state == self.state_list[0]:
                data_combined = df1[self.input_variable_list].values  ## shape Days x variables
                data_combined = np.expand_dims(data_combined, axis=0)
                print('!!!Data_combined.shape', data_combined.shape)
            else:
                data_new = df1[self.input_variable_list].values  ## shape Days x variables
                data_new = np.expand_dims(data_new, axis=0)
                print('!!! Data_new.shape', data_new.shape)
                data_combined = np.append(data_combined, data_new, axis=0)

        self.data = data_combined
        return df, data_combined

    def read_data_COVIDtrackingProject(self, path):
        '''
        Read input time series data as a dictionary of pandas dataframe
        '''
        data = pd.read_csv(path, delimiter=',').sort_values(by="date")
        ### Convert the format of dates from string to datetime
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', utc=False)

        df = {}

        if self.state_list == None:
            self.state_list = sorted([i for i in set([i for i in data['state']])])

        ### Find earliest starting date of the data
        start_dates = []
        for state in self.state_list:
            df1 = data.loc[data['state'] == state]
            start_dates.append(min(df1['date']).strftime("%Y-%m-%d"))
        max_min_date = max(start_dates)
        print('!!! min_dates', max_min_date)

        for state in self.state_list:
            df1 = data.loc[data['state'] == state].set_index('date')
            # lastUpdatedDate = df1['lastUpdateEt'].iloc[0]
            df1 = df1[max_min_date:]
            ### making new columns to process columns of interest and preserve the original data
            df1['input_onVentilator_Increase'] = df1['onVentilatorCumulative']
            df1['input_inICU_Increase'] = df1['inIcuCumulative']
            df1['input_test_positive_rate'] = df1['positiveTestsViral'] / df1['totalTestsViral']
            df1['input_case_Increase'] = df1['positiveIncrease']
            df1['input_death_Increase'] = df1['deathIncrease']

            df.update({state: df1})

        if self.if_moving_avg_data:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = df2.rolling(window=5, win_type=None).sum() / 5  ### moving average with backward window size 5
                df2 = df2.fillna(0)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        if self.if_log_scale:
            for state in self.state_list:
                df1 = df.get(state)
                df2 = df1[self.input_variable_list]
                df2 = np.log(df2 + 1)
                df1[self.input_variable_list] = df2
                df.update({state: df1})

        self.df = df

        ## Make numpy array of shape States x Days x variables
        data_combined = []
        for state in self.state_list:
            df1 = df.get(state)

            if state == self.state_list[0]:
                data_combined = df1[self.input_variable_list].values  ## shape Days x variables
                data_combined = np.expand_dims(data_combined, axis=0)
                print('!!!Data_combined.shape', data_combined.shape)
            else:
                data_new = df1[self.input_variable_list].values  ## shape Days x variables
                data_new = np.expand_dims(data_new, axis=0)
                print('!!! Data_new.shape', data_new.shape)
                data_combined = np.append(data_combined, data_new, axis=0)

        self.data = data_combined
        return df, data_combined

    def read_data_as_array_citywise(self, path):
        '''
        Read input time series as an array
        '''
        data_full = pd.read_csv(path, delimiter=',').T
        data = data_full.values
        data = np.delete(data, [2, 3], 0)  # delete lattitue & altitude
        idx = np.where((data[1, :] == 'Korea, South') | (data[1, :] == 'Japan'))
        data_sub = data[:, idx]
        data_sub = data_sub[:, 0, :]
        data_new = data_sub[2:, :].astype(int)

        idx = np.where(data_new[-1, :] > 0)
        data_new = data_new[:, idx]
        data_new = data_new[:, 0, :]
        # data_new[:,1] = np.zeros(data_new.shape[0])
        city_list = data_sub[0, idx][0]
        print('city_list', city_list)

        return data_new.T, city_list

    def combine_data(self, source):
        if len(source) == 1:
            for path in source:
                data, self.country_list = self.read_data_as_array_countrywise(path)
                data_combined = np.expand_dims(data, axis=2)

        else:
            path = source[0]
            data, self.country_list = self.read_data_as_array_countrywise(path)

            data_combined = np.empty(shape=[data.shape[0], data.shape[1], 1])
            for path in source:
                data_new = self.read_data_as_array_countrywise(path)[0]
                data_new = np.expand_dims(data_new, axis=2)
                # print('data_new.shape', data_new.shape)
                min_length = np.minimum(data_combined.shape[1], data_new.shape[1])
                data_combined = np.append(data_combined[:, 0:min_length, :], data_new[:, 0:min_length, :], axis=2)
            data_combined = data_combined[:, :, 1:]

            print('data_combined.shape', data_combined.shape)
        # data_full.replace(np.nan, 0)  ### replace all NANs with 0

        ### Replace all NANs in data_combined with 0
        where_are_NaNs = np.isnan(data_combined)
        data_combined[where_are_NaNs] = 0
        return data_combined

    def extract_random_patches(self, batch_size=None, time_interval_initial=None):
        '''
        Extract 'num_patches_perbatch' (segments) of size 'patch_size'many random patches of given size
        '''
        x = self.data.shape  # shape = 2 (ask, bid) * time * country
        k = self.patch_size
        if batch_size is None:
            num_patches_perbatch = self.num_patches_perbatch
        else:
            num_patches_perbatch = batch_size

        X = np.zeros(shape=(x[0], k, x[2], 1))  # 1 * window length * country * num_patches_perbatch
        for i in np.arange(num_patches_perbatch):
            if time_interval_initial is None:
                a = np.random.choice(x[1] - k)  # starting time of a window patch of length k
            else:
                a = time_interval_initial + i

            Y = self.data[:, a:a + k, :]  # shape 2 * k * x[2]
            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X  # X.shape = (2, k, num_countries, num_patches_perbatch)

    def extract_patches_interval(self, time_interval_initial, time_interval_terminal):
        '''
        Extract a given number of patches (segments) of size 'patch_size' during the given interval
        X.shape = (# states) x (# window length) x (# variables) x (num_patches_perbatch)
        '''
        x = self.data.shape  # shape = (# states) x (# days) x (# variables)
        k = self.patch_size  # num of consecutive days to form a single patch = window length

        X = np.zeros(
            shape=(x[0], k, x[2], 1))  # (# states) x (# window length) x (# variables) x (num_patches_perbatch)
        for i in np.arange(self.num_patches_perbatch):
            a = np.random.choice(np.arange(time_interval_initial, time_interval_terminal - k + 1))
            Y = self.data[:, a:a + k, :]  # shape 2 * k * x[2]
            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X

    def data_to_patches(self):
        '''

        args:
            path (string): Path and filename of input time series data
            patch_size (int): length of sliding window we are extracting from the time series (data)
        returns:

        '''

        if DEBUG:
            print(np.asarray(self.data))

        patches = self.extract_random_patches()
        print('patches.shape=', patches.shape)
        return patches

    def display_dictionary(self, W, cases, if_show, if_save, foldername, filename=None, custom_code4ordering=None):
        k = self.patch_size
        x = self.data.shape
        rows = np.floor(np.sqrt(self.n_components)).astype(int)
        cols = np.ceil(np.sqrt(self.n_components)).astype(int)

        '''
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(4, 3.5),
                                subplot_kw={'xticks': [], 'yticks': []})
        '''
        fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(4, 4.5),
                                subplot_kw={'xticks': [], 'yticks': []})

        print('W.shape', W.shape)

        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if self.if_log_scale:
            W = np.exp(W) - 1

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        # print('W', W)
        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        for axs, i in zip(axs.flat, range(self.n_components)):
            dict = W[:, idx[i]].reshape(x[0], k, x[2])
            # print('x.shape', x)
            for j in np.arange(dict.shape[0]):
                country_name = self.country_list[j]
                marker = ''
                if country_name == 'Korea, South':
                    marker = '*'
                elif country_name == 'China':
                    marker = 'x'
                elif country_name == 'US':
                    marker = '^'
                axs.plot(np.arange(k), dict[j, :, c], marker=marker, label='' + str(country_name))
            axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')  ## bbox_to_anchor=(0,0)
        # plt.suptitle(cases + '-Temporal Dictionary of size %d'% k, fontsize=16)
        # plt.subplots_adjust(left=0.01, right=0.55, bottom=0.05, top=0.99, wspace=0.1, hspace=0.4)  # for 24 atoms

        plt.subplots_adjust(left=0.01, right=0.62, bottom=0.1, top=0.99, wspace=0.1, hspace=0.4)  # for 12 atoms
        # plt.tight_layout()

        if if_save:
            if filename is None:
                plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + cases + '.png')
            else:
                plt.savefig(
                    'Time_series_dictionary/' + str(foldername) + '/Dict-' + cases + '_' + str(filename) + '.png')
        if if_show:
            plt.show()

    def display_dictionary_Hospital(self, W, state_name, if_show, if_save, foldername, filename=None,
                                    custom_code4ordering=None):
        k = self.patch_size
        x = self.data.shape
        rows = np.floor(np.sqrt(self.n_components)).astype(int)
        cols = np.ceil(np.sqrt(self.n_components)).astype(int)

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(6, 6),
                                subplot_kw={'xticks': [], 'yticks': []})

        print('W.shape', W.shape)

        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if self.if_log_scale:
            W = np.exp(W) - 1

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        for axs, i in zip(axs.flat, range(self.n_components)):
            dict = W[:, idx[i]].reshape(x[0], k, x[2])
            # print('x.shape', x)
            j = self.state_list.index(state_name)
            marker_list = itertools.cycle(('*', 'x', '^', 'o', '|', '+'))

            for c in np.arange(dict.shape[2]):
                variable_name = self.input_variable_list[c]
                variable_name = variable_name.replace('input_', '')

                axs.plot(np.arange(k), dict[j, :, c], marker=next(marker_list), label=variable_name)

            axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=13)  # get the largest first
            axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')  ## bbox_to_anchor=(0,0)
        plt.suptitle(str(state_name) + '-Temporal Dictionary of size %d' % k, fontsize=16)
        # plt.subplots_adjust(left=0.01, right=0.55, bottom=0.05, top=0.99, wspace=0.1, hspace=0.4)  # for 24 atoms

        plt.subplots_adjust(left=0.01, right=0.62, bottom=0.1, top=0.8, wspace=0.1, hspace=0.4)  # for 12 atoms
        # plt.tight_layout()

        if if_save:
            if filename is None:
                plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + str(state_name) + '.png')
            else:
                plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + str(state_name) + '_' + str(
                    filename) + '.png')
        if if_show:
            plt.show()

    def display_dictionary_single(self, W, if_show, if_save, foldername, filename, custom_code4ordering=None):
        k = self.patch_size
        x = self.data.shape
        code = self.code
        # print('code', code)
        importance = np.sum(code, axis=1) / sum(sum(code))

        if self.if_log_scale:
            W = np.exp(W) - 1

        if custom_code4ordering is None:
            idx = np.argsort(importance)
            idx = np.flip(idx)
        else:
            custom_importance = np.sum(custom_code4ordering, axis=1) / sum(sum(custom_code4ordering))
            idx = np.argsort(custom_importance)
            idx = np.flip(idx)

        # rows = np.floor(np.sqrt(self.n_components)).astype(int)
        # cols = np.ceil(np.sqrt(self.n_components)).astype(int)
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(4, 3),
                                subplot_kw={'xticks': [], 'yticks': []})
        print('W.shape', W.shape)
        # print('W', W)
        for axs, i in zip(axs.flat, range(self.n_components)):
            for c in np.arange(x[2]):
                if c == 0:
                    cases = 'confirmed'
                elif c == 1:
                    cases = 'death'
                else:
                    cases = 'recovered'

                dict = W[:, idx[i]].reshape(x[0], k, x[2])  ### atoms with highest importance appears first
                for j in np.arange(dict.shape[0]):

                    if c == 0:
                        marker = '*'
                    elif c == 1:
                        marker = 'x'
                    else:
                        marker = 's'

                    axs.plot(np.arange(k), dict[j, :, c], marker=marker, label='' + str(cases))
                axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=14)  # get the largest first
                axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')  ## bbox_to_anchor=(0,0)
        # plt.suptitle(str(self.country_list[0]) + '-Temporal Dictionary of size %d'% k, fontsize=16)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.3, top=0.99, wspace=0.1, hspace=0.4)
        # plt.tight_layout()

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + str(self.country_list[0]) + '_' + str(
                filename) + '.png')
        if if_show:
            plt.show()

    def display_prediction_evaluation(self, prediction, if_show, if_save, foldername, filename, if_errorbar=True,
                                      if_evaluation=False, title=None):
        A = self.data
        k = self.patch_size
        A_recons = prediction
        print('!!!!!!!A_recons.shape', A_recons.shape)
        A_predict = A_recons.copy()

        if if_evaluation:
            A_predict1 = A_recons.copy()
            ### A_recons.shape = (# trials) x (# days) x (# states) x (Future Extrapolation Length) x (# variables)
            A_recons1 = np.zeros(shape=(A_predict1.shape[0], A.shape[1] + A_predict1.shape[3], A.shape[0], A.shape[2]))
            A_recons1 = A_predict1[:, :, :, -1, :]

            # for i in np.arange(0, A_predict1.shape[2]):
            #     A_recons1[:,i + A_predict1.shape[3],:,:] = A_predict1[:,i,:, -1,:]
            a = np.zeros(shape=(A_predict1.shape[0], A_predict1.shape[3], A.shape[0], A.shape[2]))
            A_recons1 = np.append(a, A_recons1, axis=1)
            print('!!!! A.shape[1]+A_predict1.shape[3]', A.shape[1] + A_predict1.shape[3])
            print('!!!! A_recons1.shape', A_recons1.shape)

            A_recons1 = np.swapaxes(A_recons1, axis1=1, axis2=2)
            for trial in np.arange(0, A_predict1.shape[0]):
                for j in np.arange(0, A_predict1.shape[3]):
                    A_recons1[trial, :, j, :] = A[:, j, :]

            A_recons = A_recons1
            A_predict = A_recons.copy()
            print('!!!!!!!! A_recons', A_recons)

        if self.if_log_scale:
            A = np.exp(A) - 1
            A_recons = np.exp(A_recons) - 1

        if if_errorbar:
            # print('!!!', A_predict.shape)  # trials x states x days x variables
            A_predict = np.sum(A_recons, axis=0) / A_recons.shape[0]  ### axis-0 : trials
            A_std = np.std(A_recons, axis=0)
            print('!!! A_std', A_std)

        ### Make gridspec
        fig1 = plt.figure(figsize=(15, 10), constrained_layout=False)
        gs1 = fig1.add_gridspec(nrows=A_predict.shape[2], ncols=A_predict.shape[0], wspace=0.2, hspace=0.2)

        # font = font_manager.FontProperties(family="Times New Roman", size=11)

        for i in range(A_predict.shape[0]):
            for c in range(A_predict.shape[2]):

                ax = fig1.add_subplot(gs1[c, i])

                variable_name = self.input_variable_list[c]
                variable_name = variable_name.replace('input_', '')

                ### get days xticks
                start_day = self.df.get(self.state_list[0]).index[0]
                x_data = pd.date_range(start_day, periods=A.shape[1], freq='D')
                x_data_recons = pd.date_range(start_day, periods=A_predict.shape[1] - self.patch_size, freq='D')
                x_data_recons += pd.DateOffset(self.patch_size)

                ### plot axs
                ax.plot(x_data, A[i, :, c], 'b-', marker='o', markevery=5, label='Original-' + str(variable_name))

                if not if_errorbar:
                    ax.plot(x_data_recons, A_predict[i, self.patch_size:A_predict.shape[1], c],
                            'r-', marker='x', markevery=5, label='Prediction-' + str(variable_name))
                else:
                    markers, caps, bars = ax.errorbar(x_data_recons,
                                                      A_predict[i, self.patch_size:A_predict.shape[1], c],
                                                      yerr=A_std[i, self.patch_size:A_predict.shape[1], c],
                                                      fmt='r-', label='Prediction-' + str(variable_name), errorevery=1)

                    [bar.set_alpha(0.5) for bar in bars]
                    # [cap.set_alpha(0.5) for cap in caps]

                ax.set_ylim(0, np.maximum(np.max(A[i, :, c]), np.max(A_predict[i, :, c] + A_std[i, :, c])) * 1.1)

                if c == 0:
                    if title is None:
                        ax.set_title(str(self.state_list[i]), fontsize=15)
                    else:
                        ax.set_title(title, fontsize=15)

                ax.yaxis.set_label_position("left")
                # ax.yaxis.set_label_coords(0, 2)
                # ax.set_ylabel(str(list[j]), rotation=90)
                ax.set_ylabel('population', fontsize=10)  # get the largest first
                ax.yaxis.set_label_position("left")
                ax.legend()

        fig1.autofmt_xdate()
        # fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19 : '+ str(self.country_list[0]) +
        #             "\n seg. length = %i, # temp. dict. atoms = %i, learning exponent = %1.3f" % (self.patch_size, self.n_components, self.beta),
        #             fontsize=12, y=0.96)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Plot-' + str(
                filename) + '.pdf')
        if if_show:
            plt.show()

    def display_prediction_single(self, source, prediction, if_show, if_save, foldername, filename):
        A = self.combine_data(source)[0]
        k = self.patch_size
        A_predict = prediction

        if self.if_log_scale:
            A = np.exp(A) - 1
            A_predict = np.exp(A_predict) - 1

        fig, axs = plt.subplots(nrows=A.shape[2], ncols=1, figsize=(5, 5))
        lims = [(np.datetime64('2020-01-21'), np.datetime64('2020-07-15')),
                (np.datetime64('2020-01-21'), np.datetime64('2020-07-15')),
                (np.datetime64('2020-01-21'), np.datetime64('2020-07-15'))]
        if A.shape[2] == 1:
            L = zip([axs], np.arange(A.shape[2]))
        else:
            L = zip(axs.flat, np.arange(A.shape[2]))
        for axs, c in L:
            if c == 0:
                cases = 'confirmed'
            elif c == 1:
                cases = 'death'
            else:
                cases = 'recovered'

            ### get days xticks
            x_data = pd.date_range('2020-01-21', periods=A.shape[1], freq='D')
            x_data_recons = pd.date_range('2020-01-21', periods=A_predict.shape[1] - self.patch_size, freq='D')
            x_data_recons += pd.DateOffset(self.patch_size)

            ### plot axs
            axs.plot(x_data, A[0, :, c], 'b-', marker='o', markevery=5, label='Original-' + str(cases))
            axs.plot(x_data_recons, A_predict[0, self.patch_size:A_predict.shape[1], c],
                     'r-', marker='x', markevery=5, label='Prediction-' + str(cases))
            axs.set_ylim(0, np.max(A_predict[0, :, c]) + 10)

            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)
            axs.set_ylabel('log(population)', fontsize=10)  # get the largest first
            axs.yaxis.set_label_position("right")
            axs.legend(fontsize=11)

        fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19 : ' + str(self.country_list[0]) +
                     "\n seg. length = %i, # temp. dict. atoms = %i, learning exponent = %1.3f" % (
                         self.patch_size, self.n_components, self.beta),
                     fontsize=12, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Plot-' + str(self.country_list[0]) + '-' + str(
                filename) + '.png')
        if if_show:
            plt.show()

    def display_prediction(self, source, prediction, cases, if_show, if_save, foldername, if_errorbar=False):
        A = self.combine_data(source)[0]
        k = self.patch_size
        A_recons = prediction
        A_predict = prediction

        if self.if_log_scale:
            A = np.exp(A) - 1
            A_recons = np.exp(A_recons) - 1
            A_predict = np.exp(A_predict) - 1

        A_std = np.zeros(shape=A_recons.shape)
        if if_errorbar:
            A_predict = np.sum(A_predict, axis=0) / A_predict.shape[0]  ### axis-0 : trials
            A_std = np.std(A_recons, axis=0)
            # print('A_std', A_std)

        L = len(self.country_list)  # number of countries
        rows = np.floor(np.sqrt(L)).astype(int)
        cols = np.ceil(np.sqrt(L)).astype(int)

        ### get days xticks
        x_data = pd.date_range('2020-01-21', periods=A.shape[1], freq='D')
        x_data_recons = pd.date_range('2020-01-21', periods=A_predict.shape[1] - self.patch_size, freq='D')
        x_data_recons += pd.DateOffset(self.patch_size)

        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 5))
        for axs, j in zip(axs.flat, range(L)):
            country_name = self.country_list[j]
            if self.country_list[j] == 'Korea, South':
                country_name = 'Korea, S.'

            axs_empty = axs.plot([], [], ' ', label=str(country_name))
            axs_original = axs.plot(x_data, A[j, :, c], 'b-', marker='o', markevery=5, label='Original')
            if not if_errorbar:
                axs_recons = axs.plot(x_data_recons, A_predict[j, self.patch_size:A_predict.shape[1], c],
                                      'r-', marker='x', markevery=5, label='Prediction-' + str(country_name))
            else:
                y = A_predict[j, self.patch_size:A_predict.shape[1], c]
                axs_recons = axs.errorbar(x_data_recons, y, yerr=A_std[j, self.patch_size:A_predict.shape[1], c],
                                          fmt='r-.', label='Prediction', errorevery=2, )
            axs.set_ylim(0, np.maximum(np.max(A[j, :, c]), np.max(A_predict[j, :, c] + A_std[j, :, c])) * 1.1)

            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)

            axs.legend(fontsize=9)

            fig.autofmt_xdate()
            fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19:' + cases +
                         "\n segment length = %i, # temporal dictionary atoms = %i" % (
                             self.patch_size, self.n_components),
                         fontsize=12, y=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Plot-' + cases + '.png')
        if if_show:
            plt.show()

    def train_dict(self,
                   mode,
                   alpha,
                   beta,
                   learn_joint_dict,
                   foldername,
                   iterations=None,
                   update_self=True,
                   if_save=True,
                   print_iter=False):
        print('training dictionaries from patches along mode %i...' % mode)
        '''
        Trains dictionary based on patches from an i.i.d. sequence of batch of patches 
        mode = 0, 1, 2
        learn_joint_dict = True or False parameter
        '''
        W = self.W
        At = []
        Bt = []
        code = self.code
        if iterations is not None:
            n_iter = iterations
        else:
            n_iter = self.ONMF_iterations

        for t in np.arange(n_iter):
            X = self.extract_random_patches()
            if t == 0:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.ONMF_sub_iterations,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      alpha=alpha,
                                      beta=beta,
                                      batch_size=self.ONMF_batch_size)  # max number of possible patches
                W, At, Bt, H = self.ntf.train_dict_single()
                code += H
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.ONMF_sub_iterations,
                                      batch_size=self.ONMF_batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      alpha=alpha,
                                      beta=beta,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      history=self.ntf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict_single()
                code += H

            if print_iter:
                print('Current minibatch training iteration %i out of %i' % (t, self.ONMF_iterations))

        if update_self:
            self.W = W
            self.code = code
        # print('code_right_after_training', self.code)
        if self.data_source != 'JHU':
            list = self.state_list
        else:
            list = self.country_list

        print('dict_shape:', W.shape)
        print('code_shape:', code.shape)
        if if_save:
            np.save('Time_series_dictionary/' + str(foldername) + '/dict_learned_' + str(
                mode) + '_' + 'pretraining' + '_' + str(list[0]), self.W)
            np.save('Time_series_dictionary/' + str(foldername) + '/code_learned_' + str(
                mode) + '_' + 'pretraining' + '_' + str(list[0]), self.code)
            np.save('Time_series_dictionary/' + str(foldername) + '/At_' + str(mode) + '_' + 'pretraining' + '_' + str(
                list[0]), At)
            np.save('Time_series_dictionary/' + str(foldername) + '/Bt_' + str(mode) + '_' + 'pretraining' + '_' + str(
                list[0]), Bt)
        return W, At, Bt, code

    def ONMF_predictor(self,
                       mode,
                       foldername,
                       data=None,
                       learn_from_future2past=False,
                       learn_from_training_set=False,
                       prelearned_dict = None, # if not none, use this dictionary for prediction
                       ini_dict=None,
                       ini_A=None,
                       ini_B=None,
                       beta=1,
                       a1=0,  # regularizer for the code in partial fitting
                       a2=0,  # regularizer for the code in recursive prediction
                       future_extrapolation_length=0,
                       if_learn_online=True,
                       if_save=True,
                       # if_recons=True,  # Reconstruct observed data using learned dictionary
                       learning_window_cap = None, # if not none, learn only from the past "learning_window_cap" days
                       minibatch_training_initialization=True,
                       minibatch_alpha=1,
                       minibatch_beta=1,
                       print_iter=False,
                       online_learning=True,
                       num_trials=1):
        print('online learning and predicting from patches along mode %i...' % mode)
        '''
        Trains dictionary along a continuously sliding window over the data stream 
        Predict forthcoming data on the fly. This could be made to affect learning rate 
        '''
        if data is None:
            A = self.data.copy()
        else:
            A = data.copy()

        if learn_from_training_set:
            A_test = A.copy()
            A = self.data_train[:, A_test.shape[1], :]

        if learning_window_cap is None:
            learning_window_cap = self.learning_window_cap

        # print('!!!!!!!!!! A.shape', A.shape)

        k = self.patch_size  # Window length
        L = self.prediction_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        # W = self.W
        # print('W.shape', self.W.shape)
        At = []
        Bt = []
        H = []
        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))

        list_full_predictions = []
        A_recons = A.copy()

        for trial in np.arange(num_trials):
            ### Initialize self parameters
            self.W = ini_dict
            # A_recons = A[:, 0:k + L - 1, :]
            At = []
            Bt = []

            if prelearned_dict is not None:
                self.W = prelearned_dict
            else:
                # Learn new dictionary to use for prediction
                if minibatch_training_initialization:
                    # print('!!! self.W right before minibatch training', self.W)
                    self.W, At, Bt, H = self.train_dict(mode=3,
                                                        alpha=minibatch_alpha,
                                                        beta=minibatch_beta,
                                                        iterations=self.ONMF_iterations,
                                                        learn_joint_dict=True,
                                                        foldername=None,
                                                        update_self=True,
                                                        if_save=False)

                # print('data.shape', self.data.shape)
                # iter = np.floor(A.shape[1]/self.num_patches_perbatch).astype(int)
                if online_learning:
                    T_start = k
                    if learning_window_cap is not None:
                        T_start = max(k, A.shape[1]-learning_window_cap)

                    for t in np.arange(T_start, A.shape[1]):
                        if not learn_from_future2past:
                            a = np.maximum(0, t - self.num_patches_perbatch)
                            X = self.extract_patches_interval(time_interval_initial=a,
                                                              time_interval_terminal=t)  # get patch from the past2future
                        else:
                            t1 = A.shape[1] - t
                            a = np.minimum(A.shape[1], t1 + self.num_patches_perbatch)
                            X = self.extract_patches_interval(time_interval_initial=t1,
                                                              time_interval_terminal=a)  # get patch from the future2past

                        # print('X.shape', X.shape)
                        # X.shape = (# states) x (# window length) x (# variables) x (num_patches_perbatch)
                        if t == k:
                            self.ntf = Online_NTF(X, self.n_components,
                                                  iterations=self.ONMF_sub_iterations,
                                                  learn_joint_dict=True,
                                                  mode=mode,
                                                  ini_dict=self.W,
                                                  ini_A=ini_A,
                                                  ini_B=ini_B,
                                                  batch_size=self.ONMF_batch_size,
                                                  subsample=self.subsample,
                                                  beta=beta)
                            self.W, At, Bt, H = self.ntf.train_dict_single()
                            self.code += H

                            """
                            # reconstruction step
                            patch = A[:, t - k + L:t, :]
                            if learn_from_future2past:
                                patch_recons = self.predict_joint_single(patch, a1)
                                A_recons = np.append(patch_recons, A_recons, axis=1)
                            else:
                                A_recons = np.append(A_recons, patch_recons, axis=1)
                            """

                        else:
                            if t % self.learnevery == 0 and if_learn_online:  # do not learn from zero data (make np.sum(X)>0 for online learning)
                                self.ntf = Online_NTF(X, self.n_components,
                                                      iterations=self.ONMF_sub_iterations,
                                                      batch_size=self.ONMF_batch_size,
                                                      ini_dict=self.W,
                                                      ini_A=At,
                                                      ini_B=Bt,
                                                      learn_joint_dict=True,
                                                      mode=mode,
                                                      history=self.ntf.history,
                                                      subsample=self.subsample,
                                                      beta=beta)

                                self.W, At, Bt, H = self.ntf.train_dict_single()
                                # print('dictionary_updated')
                                self.code += H

                            """
                            # reconstruction step
                            patch = A[:, t - k + L:t, :]  ### in the original time orientation
                            if learn_from_future2past:
                                patch_recons = self.predict_joint_single(patch, a1)
                                # print('patch_recons', patch_recons)
                                A_recons = np.append(A_recons, patch_recons, axis=1)
                            else:
                                patch_recons = patch[:, -1, :]
                                patch_recons = patch_recons[:, np.newaxis, :]
                                A_recons = np.append(patch_recons, A_recons, axis=1)
                            """

                        if print_iter:
                            print('Current (trial, day) for ONMF_predictor (%i, %i) out of (%i, %i)' % (
                                trial + 1, t, num_trials, A.shape[1] - 1))

                        # print('!!!!! A_recons.shape', A_recons.shape)

                if learn_from_training_set:
                    # concatenate state-wise dictionary to predict one state
                    # Assumes len(list_states)=1
                    self.W = np.concatenate(np.vsplit(self.W, len(self.state_list_train)), axis=1)


            #### forward recursive prediction begins
            for t in np.arange(A.shape[1], A.shape[1] + future_extrapolation_length):
                patch = A_recons[:, t - k + L:t, :]

                if t == self.data.shape[1]:
                    patch = self.data[:, t - k + L:t, :]

                # print('!!!!! patch.shape', patch.shape)
                patch_recons = self.predict_joint_single(patch, a2)
                A_recons = np.append(A_recons, patch_recons, axis=1)
            print('new cases predicted final', A_recons[0, -1, 0])

            ### initial regulation
            A_recons[:, 0:self.learnevery + L, :] = A[:, 0:self.learnevery + L, :]
            ### patch the two reconstructions
            # A_recons = np.append(A_recons, A_recons[:,A.shape[1]:, :], axis=1)

            # print('!!!!! A_recons', A_recons.shape)

            list_full_predictions.append(A_recons.copy())

        A_full_predictions_trials = np.asarray(
            list_full_predictions)  ## shape = (# trials) x (# states) x (# days + L) x (# varibles)

        self.result_dict.update({'Evaluation_num_trials': str(num_trials)})
        self.result_dict.update({'Evaluation_A_full_predictions_trials': A_full_predictions_trials})
        self.result_dict.update({'Evaluation_Dictionary': self.W})
        self.result_dict.update({'Evaluation_Code': self.code})

        if if_save:
            if self.data_source != 'JHU':
                list = self.state_list
            else:
                list = self.country_list

            np.save('Time_series_dictionary/' + str(foldername) + '/full_results_' + 'num_trials_' + str(num_trials), self.result_dict)

            """
            np.save('Time_series_dictionary/' + str(foldername) + '/dict_learned_tensor' + '_' + str(
                list[0]) + '_' + 'afteronline' + str(self.beta), self.W)
            np.save('Time_series_dictionary/' + str(foldername) + '/code_learned_tensor' + '_' + str(
                list[0]) + '_' + 'afteronline' + str(self.beta), self.code)
            np.save('Time_series_dictionary/' + str(foldername) + '/At_' + str(list[0]) + '_' + 'afteronline' + str(
                self.beta), At)
            np.save('Time_series_dictionary/' + str(foldername) + '/Bt_' + str(list[0]) + '_' + 'afteronline' + str(
                self.beta), Bt)
            np.save('Time_series_dictionary/' + str(foldername) + '/recons', A_recons)
            """

        return A_full_predictions_trials, self.W, At, Bt, self.code

    def ONMF_predictor_historic(self,
                                mode,
                                foldername,
                                prelearned_dict_seq = None, # if not none, use this seq of dict for prediction
                                learn_from_future2past=True,
                                learn_from_training_set=True,
                                ini_dict=None,
                                ini_A=None,
                                ini_B=None,
                                beta=1,
                                a1=0,  # regularizer for the code in partial fitting
                                a2=0,  # regularizer for the code in recursive prediction
                                future_extrapolation_length=0,
                                learning_window_cap = None,
                                if_save=True,
                                minibatch_training_initialization=False,
                                minibatch_alpha=1,
                                minibatch_beta=1,
                                online_learning=True,
                                num_trials=1):  # take a number of trials to generate empirical confidence interval

        print('Running ONMF_timeseries_predictor_historic along mode %i...' % mode)
        '''
        Apply online_learning_and_prediction for intervals [0,t] for every 1\le t\le T to make proper all-time predictions 
        for evaluation  
        '''

        A = self.data

        # print('A.shape', A.shape)
        k = self.patch_size
        L = self.prediction_length
        FEL = future_extrapolation_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        # W = self.W
        if learning_window_cap is None:
            learning_window_cap = self.learning_window_cap

        self.W = ini_dict
        if ini_dict is None:
            d = self.data.shape[0]*k*self.data.shape[2]     #(# states) x (# window length) x (# variables)
            self.W = np.random.rand(d, self.n_components)
        # print('W.shape', self.W.shape)

        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))
        # A_recons = A[:, 0:k + L - 1, :]

        list_full_predictions = []
        W_total_seq_trials = []
        for trial in np.arange(num_trials):
            W_total_seq = []
            ### A_total_prediction.shape = (# days) x (# states) x (FEL) x (# variables)
            ### W_total_seq.shape = (# days) x (# states * window length * # variables) x (n_components)
            A_total_prediction = []
            ### fill in predictions for the first k days with the raw data
            for i in np.arange(k + 1):
                A_total_prediction.append(A[:, i:i + FEL, :])
                W_total_seq.append(self.W.copy())
            for t in np.arange(k + 1, A.shape[1]):
                ### Set self.data to the truncated one during [1,t]
                A1 = A[:, :t, :]
                prelearned_dict = None
                if prelearned_dict_seq is not None:
                    prelearned_dict = prelearned_dict_seq[trial,t,:,:]

                A_recons, W, At, Bt, code = self.ONMF_predictor(mode,
                                                                foldername,
                                                                data=A1,
                                                                prelearned_dict=prelearned_dict,
                                                                learn_from_future2past=learn_from_future2past,
                                                                learn_from_training_set=learn_from_training_set,
                                                                ini_dict=ini_dict,
                                                                ini_A=ini_A,
                                                                ini_B=ini_B,
                                                                beta=beta,
                                                                a1=a1,
                                                                # regularizer for the code in partial fitting
                                                                a2=a2,
                                                                # regularizer for the code in recursive prediction
                                                                future_extrapolation_length=future_extrapolation_length,
                                                                learning_window_cap=learning_window_cap,
                                                                if_save=True,
                                                                minibatch_training_initialization=minibatch_training_initialization,
                                                                minibatch_alpha=minibatch_alpha,
                                                                minibatch_beta=minibatch_beta,
                                                                print_iter=False,
                                                                online_learning=online_learning,
                                                                num_trials=1)

                A_recons = A_recons[0, :, :, :]
                # print('!!!! A_recons.shape', A_recons.shape)
                ### A_recons.shape = (# states, t+FEL, # variables)
                # print('!!!!! A_recons[:, -FEL:, :].shape', A_recons[:, -FEL:, :].shape)
                A_total_prediction.append(A_recons[:, -FEL:, :])
                W_total_seq.append(W.copy())
                ### A_recons.shape = (# states, t+FEL, # variables)
                print('Current (trial, day) for ONMF_predictor_historic (%i, %i) out of (%i, %i)' % (
                    trial + 1, t - k, num_trials, A.shape[1] - k - 1))

            A_total_prediction = np.asarray(A_total_prediction)
            W_total_seq = np.asarray(W_total_seq)
            print('W_total_seq.shape', W_total_seq.shape)
            W_total_seq_trials.append(W_total_seq)
            list_full_predictions.append(A_total_prediction)

        W_total_seq_trials = np.asarray(W_total_seq_trials)

        self.result_dict.update({'Evaluation_num_trials': str(num_trials)})
        self.result_dict.update({'Evaluation_A_full_predictions_trials': np.asarray(list_full_predictions)})
        self.result_dict.update({'Evaluation_Dictionary_seq_trials': W_total_seq_trials})
        # sequence of dictionaries to be used for historic prediction : shape (trials, time, W.shape[0], W.shape[1])

        A_full_predictions_trials = np.asarray(list_full_predictions)
        print('!!! A_full_predictions_trials.shape', A_full_predictions_trials.shape)

        if if_save:

            np.save('Time_series_dictionary/' + str(foldername) + '/full_results_' + 'num_trials_' + str(
                num_trials), self.result_dict)

            """
            np.save('Time_series_dictionary/' + str(foldername) + '/dict_learned_tensor' + '_' + str(
                list[0]) + '_' + 'afteronline' + str(self.beta), self.W)
            np.save('Time_series_dictionary/' + str(foldername) + '/code_learned_tensor' + '_' + str(
                list[0]) + '_' + 'afteronline' + str(self.beta), self.code)
            np.save('Time_series_dictionary/' + str(foldername) + '/At' + str(list[0]) + '_' + 'afteronline' + str(
                self.beta), At)
            np.save('Time_series_dictionary/' + str(foldername) + '/Bt' + str(list[0]) + '_' + 'afteronline' + str(
                self.beta), Bt)
            np.save('Time_series_dictionary/' + str(foldername) + '/Full_prediction_trials_' + 'num_trials_' + str(
                num_trials), A_full_predictions_trials)
            np.save('Time_series_dictionary/' + str(foldername) + '/W_total_seq_' + 'num_trials_' + str(
                num_trials), W_total_seq)
            """


        return A_full_predictions_trials, W_total_seq_trials, code

    def predict_joint_single(self, data, a1):
        k = self.patch_size
        L = self.prediction_length
        A = data  # A.shape = (self.data.shape[0], k-L, self.data.shape[2])
        # A_recons = np.zeros(shape=(A.shape[0], k, A.shape[2]))
        # W_tensor = self.W.reshape((k, A.shape[0], -1))
        # print('A.shape', A.shape)
        W_tensor = self.W.reshape((self.data.shape[0], k, self.data.shape[2], -1))
        # print('W.shape', W_tensor.shape)

        # for missing data, not needed for the COVID-19 data set
        # extract only rows of nonnegative values (disregarding missing entries) (negative = N/A)

        J = np.where(np.min(A, axis=(0, 1)) >= -1)
        A_pos = A[:, :, J]
        # print('A_pos', A_pos)
        # print('np.min(A)', np.min(A))
        W_tensor = W_tensor[:, :, J, :]
        W_trimmed = W_tensor[:, 0:k - L, :, :]
        W_trimmed = W_trimmed.reshape((-1, self.n_components))

        patch = A_pos

        # print('patch', patch)

        patch = patch.reshape((-1, 1))
        # print('patch.shape', patch.shape)

        # print('patch', patch)

        coder = SparseCoder(dictionary=W_trimmed.T, transform_n_nonzero_coefs=None,
                            transform_alpha=a1, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter
        code = coder.transform(patch.T)
        patch_recons = np.dot(self.W, code.T).T  # This gives prediction on the last L missing entries
        patch_recons = patch_recons.reshape(-1, k, A.shape[2])

        # now paint the reconstruction canvas
        # only add the last predicted value
        A_recons = patch_recons[:, -1, :]
        return A_recons[:, np.newaxis, :]


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
    'Northern Mariana Islands': 'MP',
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