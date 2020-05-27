from utils.ontf import Online_NTF
import numpy as np
from sklearn.decomposition import SparseCoder
import pandas as pd
import matplotlib.pyplot as plt

DEBUG = False

class time_series_tensor():
    def __init__(self,
                 path,
                 country_list,
                 source,
                 n_components=100,  # number of dictionary elements -- rank
                 iterations=50,  # number of iterations for the ONTF algorithm
                 sub_iterations = 20,  # number of i.i.d. subsampling for each iteration of ONTF
                 batch_size=20,   # number of patches used in i.i.d. subsampling
                 num_patches_perbatch = 1000,   # number of patches that ONTF algorithm learns from at each iteration
                 patch_size=7,
                 patches_file='',
                 learn_joint_dict=False,
                 prediction_length=1,
                 learnevery=5,
                 alpha=None,
                 beta=None,
                 subsample=False,
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
        self.country_list = country_list
        self.n_components = n_components
        self.iterations = iterations
        self.sub_iterations = sub_iterations
        self.num_patches_perbatch = num_patches_perbatch
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patches_file = patches_file
        self.learn_joint_dict = learn_joint_dict
        self.prediction_length = prediction_length
        self.code = np.zeros(shape=(n_components, num_patches_perbatch))
        self.learnevery=learnevery
        self.alpha = alpha
        self.beta = beta
        self.subsample = subsample
        self.if_onlynewcases = if_onlynewcases
        self.if_moving_avg_data = if_moving_avg_data
        self.if_log_scale = if_log_scale

        # read in time series data as array
        # self.data = self.read_timeseries_as_array(self.path)
        self.data, self.country_list = self.combine_data(self.source)
        print('data.shape', self.data.shape)
        self.ntf = Online_NTF(self.data, self.n_components,
                              iterations=self.sub_iterations,
                              learn_joint_dict=True,
                              mode=3,
                              ini_dict=None,
                              ini_A=None,
                              ini_B=None,
                              batch_size=self.batch_size)

        self.W = np.zeros(shape=(self.data.shape[0] * self.data.shape[2] * patch_size, n_components))

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
                data_new.T[:, i] = (data_new.T[:,i] + data_new.T[:,i-1] + data_new.T[:,i-2]+data_new.T[:,i-3]+data_new.T[:,i-4])/5
                # A_recons[:, i] = (A_recons[:, i] + A_recons[:, i-1]) / 2

        if self.if_log_scale:
                data_new = np.log(data_new+1)

        return data_new.T, country_list

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
        data_new = data_new[:,idx]
        data_new = data_new[:,0,:]
        # data_new[:,1] = np.zeros(data_new.shape[0])
        city_list = data_sub[0,idx][0]
        print('city_list', city_list)

        return data_new.T, city_list

    def combine_data(self, source):
        if len(source) == 1:
            for path in source:
                data, country_list = self.read_data_as_array_countrywise(path)
                data_combined = np.expand_dims(data, axis=2)
        else:
            path = source[0]
            data, country_list = self.read_data_as_array_countrywise(path)
            data_combined = np.empty(shape=[data.shape[0], data.shape[1], 1])
            for path in source:
                data_new = self.read_data_as_array_countrywise(path)[0]
                data_new = np.expand_dims(data_new, axis=2)
                # print('data_new.shape', data_new.shape)
                min_length = np.minimum(data_combined.shape[1], data_new.shape[1])
                data_combined = np.append(data_combined[:, 0:min_length, :], data_new[:, 0:min_length, :], axis=2)
            data_combined = data_combined[:, :, 1:]

            print('data_combined.shape', data_combined.shape)
        return data_combined, country_list

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

            Y = self.data[:, a:a+k, :]  # shape 2 * k * x[2]
            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X  # X.shape = (2, k, num_countries, num_patches_perbatch)

    def extract_patches_interval(self, time_interval_initial, time_interval_terminal):
        '''
        Extract all patches (segments) of size 'patch_size' during the given interval
        '''
        x = self.data.shape  # shape = 2 (ask, bid) * time * country
        k = self.patch_size

        X = np.zeros(shape=(x[0], k, x[2], 1))  # 2 (ask, bid) * window length * country * num_patches_perbatch
        for i in np.arange(self.num_patches_perbatch):
            a = np.random.choice(np.arange(time_interval_initial, time_interval_terminal-k+1))
            Y = self.data[:, a:a+k, :]  # shape 2 * k * x[2]
            Y = Y[:, :, :, np.newaxis]
            # print('Y.shape', Y.shape)
            if i == 0:
                X = Y
            else:
                X = np.append(X, Y, axis=3)  # x is class ndarray
        return X  # X.shape = (2, k, num_countries, num_patches_perbatch)

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

    def display_dictionary(self, W, cases, if_show, if_save, foldername, custom_code4ordering=None):
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
        fig.legend(handles, labels, loc='center right') ## bbox_to_anchor=(0,0)
        # plt.suptitle(cases + '-Temporal Dictionary of size %d'% k, fontsize=16)
        # plt.subplots_adjust(left=0.01, right=0.55, bottom=0.05, top=0.99, wspace=0.1, hspace=0.4)  # for 24 atoms

        plt.subplots_adjust(left=0.01, right=0.62, bottom=0.1, top=0.99, wspace=0.1, hspace=0.4)  # for 12 atoms
        # plt.tight_layout()

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) + '/Dict-' + cases + '_.png')
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

                dict = W[:, idx[i]].reshape(x[0], k, x[2])   ### atoms with highest importance appears first
                for j in np.arange(dict.shape[0]):

                    if c == 0:
                        marker = '*'
                    elif c == 1:
                        marker = 'x'
                    else:
                        marker = 's'

                    axs.plot(np.arange(k), dict[j, :, c], marker=marker, label=''+str(cases))
                axs.set_xlabel('%1.2f' % importance[idx[i]], fontsize=14)  # get the largest first
                axs.xaxis.set_label_coords(0.5, -0.05)  # adjust location of importance appearing beneath patches

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center') ## bbox_to_anchor=(0,0)
        # plt.suptitle(str(self.country_list[0]) + '-Temporal Dictionary of size %d'% k, fontsize=16)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.3, top=0.99, wspace=0.1, hspace=0.4)
        # plt.tight_layout()

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) +'/Dict-' + str(self.country_list[0])+ '_' + str(filename) + '.png')
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


        fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19 : '+ str(self.country_list[0]) +
                     "\n seg. length = %i, # temp. dict. atoms = %i, learning exponent = %1.3f" % (self.patch_size, self.n_components, self.beta),
                     fontsize=12, y=0.96)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) +'/Plot-'+str(self.country_list[0])+'-'+str(filename)+'.png')
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
                axs_recons = axs.errorbar(x_data_recons, y, yerr= A_std[j, self.patch_size:A_predict.shape[1], c],
                                          fmt='r-.', label='Prediction', errorevery=2,)
            axs.set_ylim(0, np.maximum(np.max(A[j, :, c]), np.max(A_predict[j, :, c] + A_std[j,:,c]))*1.1)

            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)

            axs.legend(fontsize=9)

            fig.autofmt_xdate()
            fig.suptitle('Plot of original and 1-step prediction -- ' + 'COVID-19:'+ cases +
                         "\n segment length = %i, # temporal dictionary atoms = %i" % (self.patch_size, self.n_components),
                         fontsize=12, y=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/' + str(foldername) +'/Plot-'+ cases + '.png')
        if if_show:
            plt.show()

    def train_dict(self, mode, alpha, beta, learn_joint_dict, foldername):
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
        for t in np.arange(self.iterations):
            X = self.extract_random_patches()
            if t == 0:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      learn_joint_dict = learn_joint_dict,
                                      mode=mode,
                                      alpha=alpha,
                                      beta=beta,
                                      batch_size=self.batch_size)  # max number of possible patches
                W, At, Bt, H = self.ntf.train_dict_single()
                code += H
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
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
            # print('Current iteration %i out of %i' % (t, self.iterations))
        self.W = W
        self.code = code
        # print('code_right_after_training', self.code)
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Time_series_dictionary/' + str(foldername) +'/dict_learned_' + str(mode) +'_'+ 'pretraining' + '_'+ str(self.country_list[0]), self.W)
        np.save('Time_series_dictionary/' + str(foldername) +'/code_learned_' + str(mode) +'_'+ 'pretraining' + '_'+ str(self.country_list[0]), self.code)
        np.save('Time_series_dictionary/' + str(foldername) +'/At_' + str(mode) + '_' + 'pretraining' + '_' + str(self.country_list[0]), At)
        np.save('Time_series_dictionary/' + str(foldername) +'/Bt_' + str(mode) + '_' + 'pretraining' + '_' + str(self.country_list[0]), Bt)
        return W, At, Bt, self.code

    def online_learning_and_prediction(self,
                                       mode,
                                       foldername,
                                       ini_dict=None,
                                       ini_A=None,
                                       ini_B=None,
                                       beta=1,
                                       a1=1,   # regularizer for the code in partial fitting
                                       a2=5,   # regularizer for the code in recursive prediction
                                       future_extraploation_length=0,
                                       if_learn_online = True,
                                       if_save=True):
        print('online learning and predicting from patches along mode %i...' % mode)
        '''
        Trains dictionary along a continuously sliding window over the data stream 
        Predict forthcoming data on the fly. This could be made to affect learning rate 
        '''
        A = self.data
        # print('A.shape', A.shape)
        k = self.patch_size
        L = self.prediction_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        # W = self.W
        self.W = ini_dict
        # print('W.shape', self.W.shape)
        At = []
        Bt = []
        H = []
        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))
        A_recons = A[:, 0:k + L - 1, :]
        error = np.zeros(shape=(A.shape[0], k - 1, A.shape[2]))

        code = self.code
        # print('data.shape', self.data.shape)
        # iter = np.floor(A.shape[1]/self.num_patches_perbatch).astype(int)
        for t in np.arange(k, A.shape[1]):
            a = np.maximum(0, t - self.num_patches_perbatch)
            X = self.extract_patches_interval(time_interval_initial=a,
                                              time_interval_terminal=t)  # get patch from the past
            # print('X.shape', X.shape)
            if t == k:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      learn_joint_dict=True,
                                      mode=mode,
                                      ini_dict=self.W,
                                      ini_A=ini_A,
                                      ini_B=ini_B,
                                      batch_size=self.batch_size,
                                      subsample=self.subsample,
                                      beta=beta)
                self.W, At, Bt, H = self.ntf.train_dict_single()
                self.code += H
                # print('W', W)


                # prediction step
                patch = A[:, t - k + L:t, :]
                patch_recons = self.predict_joint_single(patch, a1)
                # print('patch_recons', patch_recons)
                A_recons = np.append(A_recons, patch_recons, axis=1)

            else:

                if t % self.learnevery == 0 and if_learn_online:  # do not learn from zero data (make np.sum(X)>0 for online learning)
                    self.ntf = Online_NTF(X, self.n_components,
                                          iterations=self.sub_iterations,
                                          batch_size=self.batch_size,
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

                # prediction step
                patch = A[:, t - k + L:t, :]
                # print('patch.shape', patch.shape)
                patch_recons = self.predict_joint_single(patch, a1)
                A_recons = np.append(A_recons, patch_recons, axis=1)

            print('Current iteration for online learning/prediction %i out of %i' % (t, self.iterations))

        # forward recursive prediction begins
        for t in np.arange(A.shape[1], A.shape[1] + future_extraploation_length):
            patch = A_recons[:, t - k + L:t, :]
            patch_recons = self.predict_joint_single(patch, a2)
            A_recons = np.append(A_recons, patch_recons, axis=1)
        print('new cases predicted final', A_recons[0, -1, 0])
        '''
        A_recons_future = A.copy()
        A_recons_future = np.append(A_recons_future, np.expand_dims(A_recons[:,-1,:], axis=1), axis=1)
        for t in np.arange(A.shape[1], A.shape[1] + future_extraploation_length):
            patch = A_recons_future[:, t - k + L:t, :]
            patch_recons = self.predict_joint_single(patch, a1)
            A_recons_future = np.append(A_recons_future, patch_recons, axis=1)
        print('new cases predicted final', A_recons_future[0, -1, 0])
        '''

            # print('Current iteration %i out of %i' % (t, self.iterations))

        ### initial regulation
        A_recons[:, 0:self.learnevery + L, :] = A[:, 0:self.learnevery + L, :]
        ### patch the two reconstructions
        # A_recons = np.append(A_recons, A_recons[:,A.shape[1]:, :], axis=1)


        # print('error.shape', error.shape)
        # print('error shape', error[:,0:learn_every+1,:].shape)
        # error[:,0:learn_every+1,:] = np.zeros(shape=(error.shape[0], learn_every, error.shape[2]))

        # print('A_recons', A_recons.shape)
        # print(W)
        # print('dict_shape:', self.W.shape)
        # print('code_shape:', self.code.shape)
        if if_save:
            np.save('Time_series_dictionary/' + str(foldername) +'/dict_learned_tensor' +'_'+ str(self.country_list[0]) + '_' +'afteronline' + str(self.beta), self.W)
            np.save('Time_series_dictionary/' + str(foldername) +'/code_learned_tensor' +'_'+ str(self.country_list[0]) + '_' +'afteronline' + str(self.beta), self.code)
            np.save('Time_series_dictionary/' + str(foldername) +'/At' + str(self.country_list[0]) + '_' +'afteronline' + str(self.beta), At)
            np.save('Time_series_dictionary/' + str(foldername) +'/Bt' + str(self.country_list[0]) + '_' +'afteronline' + str(self.beta), Bt)
            np.save('Time_series_dictionary/' + str(foldername) +'/recons', A_recons)
        return A_recons, error, self.W, At, Bt, self.code

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

        J = np.where(np.min(A, axis=(0,1)) >= -1)
        A_pos = A[:,:,J]
        # print('A_pos', A_pos)
        # print('np.min(A)', np.min(A))
        W_tensor = W_tensor[:,:,J,:]
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
        A_recons = patch_recons[:,k-1,:]
        return A_recons[:,np.newaxis,:]


def main_train_joint():
    print('pre-training temporal dictionary started...')

    path_confirmed = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'joint9_00_5_1000'  ## for saving files
    source = [path_confirmed, path_deaths, path_recovered]

    country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'Spain']
    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       country_list=country_list,
                                       alpha=3,  # L1 sparsity regularizer
                                       beta=1,  # default learning exponent --
                                       # customized in both trianing and online prediction functions
                                       # learning rate exponent in online learning -- smaller weighs new data more
                                       n_components=24,  # number of dictionary elements -- rank
                                       iterations=20,  # number of iterations for the ONTF algorithm
                                       sub_iterations=2,  # number of i.i.d. subsampling for each iteration of ONTF
                                       batch_size=50,  # number of patches used in i.i.d. subsampling
                                       num_patches_perbatch=100,
                                       # number of patches per ONMF iteration (size of mini batch)
                                       # number of patches that ONTF algorithm learns from at each iteration
                                       patch_size=6,
                                       prediction_length=1,
                                       learnevery=1,
                                       subsample=False,
                                       if_onlynewcases=True,   # take the derivate of the time-series of total to get new cases
                                       if_moving_avg_data=True,
                                       if_log_scale=True)

    L = 30
    avg_iter = 10
    A_recons = []
    # W_old = W.copy()
    for step in np.arange(avg_iter):
        W, At, Bt, H = reconstructor.train_dict(mode=3,
                                                foldername=foldername,
                                                alpha=1,  ## L1 regularizer for sparse coding
                                                beta=1,  ## learning rate exponent in pre-learning
                                                learn_joint_dict=True)

        A_predict, error, W1, At1, Bt1, H = reconstructor.online_learning_and_prediction(mode=3,
                                                                                         ini_dict=W,
                                                                                         foldername=foldername,
                                                                                         beta=5,  # no effect if "if_learn_online" is false
                                                                                         ini_A=At,
                                                                                         ini_B=Bt,
                                                                                         a1=0, # regularizer for training
                                                                                         a2=0, # regularizer for prediction
                                                                                         future_extraploation_length=L,
                                                                                         if_learn_online = True,
                                                                                         if_save=True)
        A_recons.append(A_predict.tolist())
        print('Current iteration %i out of %i' % (step, avg_iter))

    A_recons = np.asarray(A_recons)
    np.save('Time_series_dictionary/' + str(foldername) + '/recons_nononline', A_recons)

    # print('change in dictionary after online learning', np.linalg.norm(W_old - W1))

    '''
    ### For loading saved checkpoints just for plotting
    W1 = np.load('Time_series_dictionary/' + str(foldername) + '/dict_learned_3_pretraining_Korea, South.npy')
    code = np.load('Time_series_dictionary/' + str(foldername) + '/code_learned_3_pretraining_Korea, South.npy')
    reconstructor.code = code
    A_recons = np.load("Time_series_dictionary/" + str(foldername) + "/recons_nononline.npy")
    '''

    reconstructor.display_dictionary(W1, cases='confirmed', if_show=True, if_save=True, foldername=foldername)
    reconstructor.display_dictionary(W1, cases='death', if_show=True, if_save=True, foldername=foldername)
    reconstructor.display_dictionary(W1, cases='recovered', if_show=True, if_save=True, foldername=foldername)

    reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='recovered', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)

def main_train_joint_transfer():
    print('pre-training temporal dictionary started...')

    path_confirmed = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'joint_transfer1'  ## for saving files
    source = [path_confirmed, path_deaths]

    L = 30
    avg_iter = 10
    A_recons = []

    country_list = ['Korea, South']
    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       country_list=country_list,
                                       alpha=1,  # L1 sparsity regularizer
                                       beta=1,  # default learning exponent --
                                       # customized in both trianing and online prediction functions
                                       # learning rate exponent in online learning -- smaller weighs new data more
                                       n_components=12,  # number of dictionary elements -- rank
                                       iterations=2,  # number of iterations for the ONTF algorithm
                                       sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
                                       batch_size=50,  # number of patches used in i.i.d. subsampling
                                       num_patches_perbatch=100,
                                       # number of patches per ONMF iteration (size of mini batch)
                                       # number of patches that ONTF algorithm learns from at each iteration
                                       patch_size=6,
                                       prediction_length=1,
                                       learnevery=1,
                                       subsample=False,
                                       if_onlynewcases=True,
                                       # take the derivate of the time-series of total to get new cases
                                       if_moving_avg_data=True,
                                       if_log_scale=True)

    for step in np.arange(avg_iter):

        ### Learn dictionary from Korea
        W_Korea, At, Bt, H = reconstructor.train_dict(mode=3,
                                                foldername=foldername,
                                                alpha=1,  ## L1 regularizer for sparse coding
                                                beta=1,  ## learning rate exponent in pre-learning
                                                learn_joint_dict=True)

        # print('W_Korea.shape for joint transfer learning', W_Korea.shape)

        ### Learn joint dictionary from all countries
        country_list = ['US']
        reconstructor = time_series_tensor(path=path_confirmed,
                                           source=source,
                                           country_list=country_list,
                                           alpha=1,  # L1 sparsity regularizer
                                           beta=1,  # default learning exponent --
                                           # customized in both trianing and online prediction functions
                                           # learning rate exponent in online learning -- smaller weighs new data more
                                           n_components=12,  # number of dictionary elements -- rank
                                           iterations=20,  # number of iterations for the ONTF algorithm
                                           sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
                                           batch_size=50,  # number of patches used in i.i.d. subsampling
                                           num_patches_perbatch=100,
                                           # number of patches per ONMF iteration (size of mini batch)
                                           # number of patches that ONTF algorithm learns from at each iteration
                                           patch_size=6,
                                           prediction_length=1,
                                           learnevery=1,
                                           subsample=False,
                                           if_onlynewcases=True,
                                           # take the derivate of the time-series of total to get new cases
                                           if_moving_avg_data=True,
                                           if_log_scale=True)

        '''
        W_joint, At, Bt, H = reconstructor.train_dict(mode=3,
                                                      foldername=foldername,
                                                      alpha=1,  ## L1 regularizer for sparse coding
                                                      beta=1,  ## learning rate exponent in pre-learning
                                                      learn_joint_dict=True)
        '''

        W = np.vstack([W_Korea] * len(country_list))
        At = np.vstack([At] * len(country_list))
        Bt = np.vstack([Bt] * len(country_list))

        # W = np.hstack((W_Korea, W_joint))
        # print('W.shape', W.shape)

        ### Start online learning and prediction
        A_predict, error, W1, At1, Bt1, H = reconstructor.online_learning_and_prediction(mode=3,
                                                                                         ini_dict=W,
                                                                                         foldername=foldername,
                                                                                         beta=1,
                                                                                         ini_A=At,
                                                                                         ini_B=Bt,
                                                                                         a1=0,
                                                                                         # regularizer for training
                                                                                         a2=0.2,
                                                                                         # regularizer for prediction
                                                                                         future_extraploation_length=L,
                                                                                         if_save=True,
                                                                                         if_learn_online=False)
        A_recons.append(A_predict.tolist())
        print('A_recons.max', np.max(np.asarray(A_recons)))
    A_recons = np.asarray(A_recons)
    print('A_recons.shape', A_recons.shape)
    print('A_recons.shape', np.max(A_recons))

    # A_recons = np.sum(A_recons, axis=0) / avg_iter

    # reconstructor.display_dictionary(W, cases=cases)
    reconstructor.display_dictionary(W1, cases='confirmed', if_show=True, if_save=True, foldername=foldername)
    reconstructor.display_dictionary(W1, cases='death', if_show=True, if_save=True, foldername=foldername)
    # reconstructor.display_dictionary(W1, cases='recovered', if_show=True, if_save=True, foldername=foldername)

    reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    # reconstructor.display_prediction(source, A_recons, cases='recovered', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)


def main_train_single():
    print('pre-training temporal dictionary started...')

    path_confirmed = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'joint4'  ## for saving files
    source = [path_confirmed, path_deaths]

    country_list = ['Korea, South']
    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       country_list=country_list,
                                       alpha=10,  # L1 sparsity regularizer
                                       beta=1,  # default learning exponent --
                                       # customized in both trianing and online prediction functions
                                       # learning rate exponent in online learning -- smaller weighs new data more
                                       n_components=12,  # number of dictionary elements -- rank
                                       iterations=2,  # number of iterations for the ONTF algorithm
                                       sub_iterations=20,  # number of i.i.d. subsampling for each iteration of ONTF
                                       batch_size=50,  # number of patches used in i.i.d. subsampling
                                       num_patches_perbatch=100,
                                       # number of patches per ONMF iteration (size of mini batch)
                                       # number of patches that ONTF algorithm learns from at each iteration
                                       patch_size=6,
                                       prediction_length=1,
                                       learnevery=1,
                                       subsample=False,
                                       if_onlynewcases=True,   # take the derivate of the time-series of total to get new cases
                                       if_moving_avg_data=True,
                                       if_log_scale=False)

    W, At, Bt, H = reconstructor.train_dict(mode=3,
                                            foldername=foldername,
                                            alpha=1,  ## L1 regularizer for sparse coding
                                            beta=1,  ## learning rate exponent in pre-learning
                                            learn_joint_dict=True)

    reconstructor.display_dictionary_single(W, if_show=True, if_save=True, foldername=foldername, filename=0)


def main_transfer_prediction():

    path_confirmed = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'joint2'  ## for saving files
    source = [path_confirmed, path_deaths]
    # source = [path_confirmed]

    # country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'France']
    country_list = ['US']
    for beta in np.arange(10, 0, -1):
        filenum = beta  ## for saving files
        print('%%%% current beta= ', beta)
        reconstructor = time_series_tensor(path=path_confirmed,
                                           source=source,
                                           country_list = country_list,
                                           alpha=1,   # L1 sparsity regularizer
                                           beta=beta,  # learning rate exponent in online learning -- smaller weighs new data more
                                           n_components=12,  # number of dictionary elements -- rank
                                           iterations=2,  # number of iterations for the ONTF algorithm
                                           sub_iterations=2,  # number of i.i.d. subsampling for each iteration of ONTF
                                           batch_size=50,  # number of patches used in i.i.d. subsampling
                                           num_patches_perbatch=100,  # number of patches per ONMF iteration (size of mini batch)
                                           # number of patches that ONTF algorithm learns from at each iteration
                                           patch_size=6,
                                           prediction_length=1,
                                           learnevery=1,
                                           subsample=False,
                                           if_onlynewcases=True, # take the derivate of the time-series of total to get new cases
                                           if_moving_avg_data=True,
                                           if_log_scale=False)


        W = np.load('Time_series_dictionary/' + str(foldername) +'/dict_learned_3_pretraining_Korea, South.npy')
        At = np.load('Time_series_dictionary/' + str(foldername) +'/At_3_pretraining_Korea, South.npy')
        Bt = np.load('Time_series_dictionary/' + str(foldername) +'/Bt_3_pretraining_Korea, South.npy')
        code = np.load('Time_series_dictionary/' + str(foldername) +'/code_learned_3_pretraining_Korea, South.npy')
        # reconstructor.code = code
        # reconstructor.display_dictionary_single(W, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filenum=filenum)

        # W = np.tile(W, (len(country_list),1))  ### initialize joint dictionary
        # W = np.vstack([W]*len(country_list))

        L = 90
        avg_iter = 1
        A_recons = []
        W_old = W.copy()
        for step in np.arange(avg_iter):
            A_predict, error, W1, At1, Bt1, H1 = reconstructor.online_learning_and_prediction(mode=3,
                                                                                              ini_dict=W,
                                                                                              foldername=foldername,
                                                                                              ini_A=None,
                                                                                              ini_B=None,
                                                                                              a1 = 1,
                                                                                              a2 = 1,
                                                                                              future_extraploation_length=L,
                                                                                              if_save=True)
            A_recons.append(A_predict.tolist())
        A_recons = np.asarray(A_recons)
        A_recons = np.sum(A_recons, axis=0)/avg_iter
        for i in np.arange(3, A_recons.shape[1]):
            if np.min(A_recons[:,i,:])>0:
                # print('np.mean(A_recons[:,i-2:i,:], axis=1)', A_recons[:,i-2:i,:])
                A_recons[:,i,:] = (A_recons[:,i,:] + A_recons[:,i-1,:] + A_recons[:,i-2,:] + A_recons[:,i-3,:])/4
                # A_recons[:, i, :] = (A_recons[:, i, :] + A_recons[:, i-1, :]) / 2

        print('change in dictionary after online learning', np.linalg.norm(W_old - W1))

        reconstructor.display_dictionary_single(W1, if_show=False, if_save=True, foldername=foldername, filename=filenum, custom_code4ordering=code)
        reconstructor.display_prediction_single(source, A_recons, if_show=False, if_save=True, foldername=foldername, filename=filenum)


def main():

    main_train_joint()
    # main_train_joint_transfer()
    # main_train_single()
    # main_transfer_prediction()

if __name__ == '__main__':
    main()

