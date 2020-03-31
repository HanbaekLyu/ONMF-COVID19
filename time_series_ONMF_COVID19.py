from ontf import Online_NTF
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import SparseCoder
import itertools
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

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
                 beta=None,
                 subsample=False):
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
        self.W = np.zeros(shape=(2* patch_size, n_components))  # 2 for ask and bid
        self.code = np.zeros(shape=(n_components, iterations*batch_size))
        self.learnevery=learnevery
        self.beta = beta
        self.subsample = subsample

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
        country_list

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
        #*****

        args:
            path (string): Path and filename of input time series data
            patch_size (int): length of sliding window we are extracting from the time series (data)
        returns: #***

        '''

        if DEBUG:
            print(np.asarray(self.data))

        patches = self.extract_random_patches()
        print('patches.shape=', patches.shape)
        return patches

    def display_dictionary(self, W, cases, if_show, if_save, filenum):
        k = self.patch_size
        x = self.data.shape
        rows = np.floor(np.sqrt(self.n_components)).astype(int)
        cols = np.ceil(np.sqrt(self.n_components)).astype(int)
        fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(5, 5),
                                subplot_kw={'xticks': [], 'yticks': []})
        print('W.shape', W.shape)
        # print('W', W)
        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        for axs, i in zip(axs.flat, range(self.n_components)):
            dict = W[:, i].reshape(x[0], k, x[2])
            for j in np.arange(dict.shape[0]):
                country_name = self.country_list[j]
                if self.country_list[j] == 'Korea, South':
                    country_name = 'Korea, S.'

                axs.plot(np.arange(k), dict[j, :, c], label=''+str(country_name))

        handles, labels = axs.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right') ## bbox_to_anchor=(0,0)
        # plt.suptitle(cases + '-Temporal Dictionary of size %d'% k, fontsize=16)
        plt.subplots_adjust(left=0.01, right=0.75, bottom=0, top=0.99, wspace=0.08, hspace=0.23)
        # plt.tight_layout()

        if if_save:
            plt.savefig('Time_series_dictionary/Dict-' + cases + '-' + str(filenum) + '.png')
        if if_show:
            plt.show()

    def display_prediction(self, source, prediction, cases, if_show, if_save, filenum):
        A = self.combine_data(source)[0]
        k = self.patch_size
        A_predict = prediction
        L = len(self.country_list)  # number of countries
        rows = np.floor(np.sqrt(L)).astype(int)
        cols = np.ceil(np.sqrt(L)).astype(int)

        if cases == 'confirmed':
            c = 0
        elif cases == 'death':
            c = 1
        else:
            c = 2

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
        for axs, j in zip(axs.flat, range(L)):
            country_name = self.country_list[j]
            if self.country_list[j] == 'Korea, South':
                country_name = 'Korea, S.'

            axs.plot(np.arange(A.shape[1]), A[j, :, c], 'b-', label='Original-' + str(country_name))
            axs.plot(np.arange(self.patch_size, A_predict.shape[1]),
                    A_predict[j, self.patch_size:A_predict.shape[1], c],
                    'r-', label='Recons.-' + str(country_name))
            axs.set_ylim(-10, np.max(A_predict[j, :, c]) + 10)
            # ax.text(2, 0.65, str(list[j]))
            axs.yaxis.set_label_position("right")
            # ax.yaxis.set_label_coords(0, 2)
            # ax.set_ylabel(str(list[j]), rotation=90)
            axs.legend(fontsize=11)
            fig.suptitle('Plot of original and reconstruction -- ' + 'COVID-19-'+ cases +
                         "\n segment length = %i, # temporal dictionary components = %i" % (self.patch_size, self.n_components),
                         fontsize=12, y=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            # plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.85, wspace=0.08, hspace=0.23)

        if if_save:
            plt.savefig('Time_series_dictionary/Plot-'+cases+'-'+str(filenum)+'.png')
        if if_show:
            plt.show()

    def train_dict(self, mode, beta, learn_joint_dict):
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
                                      beta=beta,
                                      batch_size=self.batch_size)  # max number of possible patches
                W, At, Bt, H = self.ntf.train_dict_single()
            else:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      batch_size=self.batch_size,
                                      ini_dict=W,
                                      ini_A=At,
                                      ini_B=Bt,
                                      beta=beta,
                                      learn_joint_dict=learn_joint_dict,
                                      mode=mode,
                                      history=self.ntf.history)
                # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                # for "iterations" iterations
                W, At, Bt, H = self.ntf.train_dict_single()
                # code += H
            print('Current iteration %i out of %i' % (t, self.iterations))
        self.W = W
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Time_series_dictionary/dict_learned_tensor_country_' + str(mode) + 'joint' + str(learn_joint_dict), self.W)
        np.save('Time_series_dictionary/code_learned_tensor_country_' + str(mode) + 'joint' + str(learn_joint_dict), self.code)
        return W, At, Bt, H

    def online_learning_and_prediction(self,
                                       mode,
                                       ini_dict=None,
                                       ini_A=None,
                                       ini_B=None,
                                       future_extraploation_length=0):
        print('online learning and predicting from patches along mode %i...' % mode)
        '''
        Trains dictionary along a continuously sliding window over the data stream 
        Predict forthcoming data on the fly. This could be made to affect learning rate 
        '''
        A = self.data
        print('A.shape', A.shape)
        k = self.patch_size
        L = self.prediction_length
        # A_recons = np.zeros(shape=A.shape)
        # print('A_recons.shape',A_recons.shape)
        W = self.W
        At = []
        Bt = []
        # A_recons = np.zeros(shape=(A.shape[0], k+L-1, A.shape[2]))
        A_recons = A[:, 0:k+L-1, :]
        error = np.zeros(shape=(A.shape[0], k-1, A.shape[2]))

        code = self.code
        print('data.shape', self.data.shape)
        # iter = np.floor(A.shape[1]/self.num_patches_perbatch).astype(int)
        for t in np.arange(k, np.minimum(self.iterations, A.shape[1])):
            a = np.maximum(0, t - self.num_patches_perbatch)
            X = self.extract_patches_interval(time_interval_initial=a, time_interval_terminal=t) # get patch from the past
            # print('X.shape', X.shape)
            if t == k:
                self.ntf = Online_NTF(X, self.n_components,
                                      iterations=self.sub_iterations,
                                      learn_joint_dict=True,
                                      mode=mode,
                                      ini_dict=ini_dict,
                                      ini_A=ini_A,
                                      ini_B=ini_B,
                                      batch_size=self.batch_size,
                                      subsample=self.subsample,
                                      beta=self.beta)  # max number of possible patches
                W, At, Bt, H = self.ntf.train_dict_single()
                self.W = W
                # print('W', W)

                # prediction step
                patch = A[:, t - k + L:t, :]
                patch_recons = self.predict_joint_single(patch)
                # print('patch_recons', patch_recons)
                A_recons = np.append(A_recons, patch_recons, axis=1)

            else:
                if t % self.learnevery == 0 and np.min(X) > 0: # do not learn from missing data
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
                                          beta=self.beta)
                    # out of "sample_size" columns in the data matrix, sample "batch_size" randomly and train the dictionary
                    # for "iterations" iterations
                    W, At, Bt, H = self.ntf.train_dict_single()
                    self.W = W
                    # code += H

                # prediction step
                patch = A[:, t-k+L:t, :]
                # print('patch.shape', patch.shape)
                patch_recons = self.predict_joint_single(patch)
                A_recons = np.append(A_recons, patch_recons, axis=1)
            print('Current iteration %i out of %i' % (t, self.iterations))

        for t in np.arange(A.shape[1], A.shape[1]+future_extraploation_length):
            ### predictive continuation into the future
            patch = A_recons[:, t - k + L:t, :]
            # print('patch.shape', patch.shape)
            patch_recons = self.predict_joint_single(patch)
            A_recons = np.append(A_recons, patch_recons, axis=1)

            print('Current iteration %i out of %i' % (t, self.iterations))

        # initial regulation
        A_recons[:,0:self.learnevery+L,:] = A[:,0:self.learnevery+L,:]
        # print('error.shape', error.shape)
        # print('error shape', error[:,0:learn_every+1,:].shape)
        # error[:,0:learn_every+1,:] = np.zeros(shape=(error.shape[0], learn_every, error.shape[2]))

        print('A_recons', A_recons.shape)
        self.W = W
        print(W)
        print('dict_shape:', self.W.shape)
        print('code_shape:', self.code.shape)
        np.save('Time_series_dictionary/dict_learned_tensor' + str(mode) + 'joint', self.W)
        np.save('Time_series_dictionary/code_learned_tensor' + str(mode) + 'joint', self.code)
        np.save('Time_series_dictionary/At' + str(mode) + 'joint', At)
        np.save('Time_series_dictionary/Bt' + str(mode) + 'joint', Bt)
        np.save('Time_series_dictionary/recons', A_recons)
        print('A_reconst.shape', A_recons.shape)
        return A_recons, error, W, At, Bt, H

    def predict_joint_single(self, data):
        k = self.patch_size
        L = self.prediction_length
        A = data  # A.shape = (self.data.shape[0], k-L, self.data.shape[2])
        is_prediction_correct = np.zeros(shape=(A.shape[0], k - 1, A.shape[2]))
        # A_recons = np.zeros(shape=(A.shape[0], k, A.shape[2]))
        # W_tensor = self.W.reshape((k, A.shape[0], -1))
        # print('A.shape', A.shape)
        W_tensor = self.W.reshape((self.data.shape[0], k, self.data.shape[2], -1))
        # print('W.shape', W_tensor.shape)

        # for missing data, not needed for the COVID-19 data set
        # extract only rows of nonnegative values (disregarding missing entries) (negative = N/A)
        J = np.where(np.min(A, axis=(0,1)) >= -1)  # damn, this took long
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
                            transform_alpha=1, transform_algorithm='lasso_lars', positive_code=True)
        # alpha = L1 regularization parameter
        code = coder.transform(patch.T)
        patch_recons = np.dot(self.W, code.T).T  # This gives prediction on the last L missing entries
        patch_recons = patch_recons.reshape(-1, k, A.shape[2])

        # now paint the reconstruction canvas
        # only add the last predicted value
        A_recons = patch_recons[:,k-1,:]
        return A_recons[:,np.newaxis,:]

def main():

    path_confirmed = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    # source = [path_deaths]
    source = [path_confirmed, path_deaths, path_recovered]
    # source = [path3]

    country_list = ['Korea, South', 'China', 'Japan', 'US', 'Italy', 'Spain']
    # country_list = ['US']
    cases = 'death'  ### cases = one of ['confirmed', 'death', 'recovered']


    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       country_list = country_list,
                                       beta = 0.01,  # learning rate exponent -- smaller weighs new data more
                                       n_components=36,  # number of dictionary elements -- rank
                                       iterations=200,  # number of iterations for the ONTF algorithm
                                       sub_iterations=2,  # number of i.i.d. subsampling for each iteration of ONTF
                                       batch_size=50,  # number of patches used in i.i.d. subsampling
                                       num_patches_perbatch=100,  # number of patches per ONMF iteration (size of mini batch)
                                       # number of patches that ONTF algorithm learns from at each iteration
                                       patch_size=20,
                                       patches_file='',
                                       learn_joint_dict=False,
                                       prediction_length=1,
                                       learnevery=1,
                                       subsample=False)

    W, At, Bt, H = reconstructor.train_dict(mode=3,
                                            beta=1,
                                            learn_joint_dict=True)

    # W = np.load("Time_series_dictionary\dict_learned_tensor3joint.npy")
    # At = np.load("Time_series_dictionary\At3joint.npy")
    # Bt = np.load("Time_series_dictionary\Bt3joint.npy")

    A_predict, error, W, At, Bt, H = reconstructor.online_learning_and_prediction(mode=3,
                                                                                  ini_dict=W,
                                                                                  ini_A=At,
                                                                                  ini_B=Bt,
                                                                                  future_extraploation_length = 5)

    # print(A_predict)
    # reconstructor.display_dictionary(W, cases=cases)
    filenum=8
    reconstructor.display_dictionary(W, cases='confirmed', if_show=False, if_save=True, filenum=filenum)
    reconstructor.display_dictionary(W, cases='death', if_show=False, if_save=True, filenum=filenum)
    reconstructor.display_dictionary(W, cases='recovered', if_show=False, if_save=True, filenum=filenum)
    reconstructor.display_prediction(source, A_predict, cases='confirmed', if_show=False, if_save=True, filenum=filenum)
    reconstructor.display_prediction(source, A_predict, cases='death', if_show=False, if_save=True, filenum=filenum)
    reconstructor.display_prediction(source, A_predict, cases='recovered', if_show=False, if_save=True, filenum=filenum)



if __name__ == '__main__':
    main()

