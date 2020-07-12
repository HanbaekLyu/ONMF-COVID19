from time_series_ONMF_COVID19 import time_series_tensor
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

def main_train_joint():
    print('pre-training temporal dictionary started...')

    path_confirmed = "Data/covid-19-data/time_series_covid19_confirmed_US.csv"
    path_deaths = "Data/covid-19-data/time_series_covid19_deaths_US.csv"
    path_residential = "Data/covid-19-data/df_residential_new.csv"
    path_weather = "Data/covid-19-data/d_weather.csv"
    #path_recovered = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = '05_11_2020'  ## for saving files
    #source = [path_confirmed, path_deaths, path_recovered]
    source = [path_confirmed, path_residential]
    #source = [path_deaths, path_residential]
    #source = [path_confirmed]

    n_components = 50

    #country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'Spain']
    #state_list = ["Massachusetts", "California", "New Hampshire", "New Jersey", "Pennsylvania", "Connecticut", "Maryland", "Ohio", "West Virginia", "New York"]
    #state_list = ["Mississippi", "Arkansas", "West Virginia", "Oklahoma", "Alabama", "Louisiana", "Kentucky",
    #              "Tennessee", "Wyoming"]

    state_list = ["California", "Texas", "Florida", "New York", "Pennsylvania", "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan", "Wyoming"]

    ## good healthcare -> predict bad healthcare
    #state_list = ["Hawaii", "Massachusetts", "Connecticut", "Washington", "Rhode Island", "New Jersey", "California", "Maryland", "Utah", "Minnesota", "Texas"]

    ## bad healthcare -> predict good healthcare
    #state_list = ["Arizona", "Alabama", "Texas", "Louisiana", "Oklahoma", "Georgia", "Arkansas", "South Carolina", "Mississippi", "North Carolina", "Maryland"]

    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       state_list=state_list,
                                       alpha=3,  # L1 sparsity regularizer
                                       beta=1,  # default learning exponent --
                                       # customized in both trianing and online prediction functions
                                       # learning rate exponent in online learning -- smaller weighs new data more
                                       n_components=n_components,  # number of dictionary elements -- rank
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
                                       if_log_scale=True,
                                       trainOrNot=True,
                                       mobilityOrNot=False,
                                       day_delay = 0)

    L = 0
    avg_iter = 50
    A_recons = []
    MSEs = []
    # W_old = W.copy()
    for step in np.arange(avg_iter):
        reconstructor.n_components = 50

        ### Clear up the fields
        reconstructor.W = None
        reconstructor.code = np.zeros(shape=(n_components, 100))

        ### Minibatch dictionary learning
        W, At, Bt, H = reconstructor.train_dict(mode=3,
                                                foldername=foldername,
                                                alpha=1,  ## L1 regularizer for sparse coding
                                                beta=1,  ## learning rate exponent in pre-learning
                                                learn_joint_dict=True)
        ### Online dictionary learning and prediction
        A_predict, error, MSE, W1, At1, Bt1, H = reconstructor.online_learning_and_prediction_statewise(mode=3,
                                                                                         ini_dict=W,
                                                                                         foldername=foldername,
                                                                                         beta=4,  # no effect if "if_learn_online" is false
                                                                                         ini_A=None,
                                                                                         ini_B=None,
                                                                                         a1=0, # regularizer for training
                                                                                         a2=2, # regularizer for prediction
                                                                                         future_extraploation_length=L,
                                                                                         if_learn_online = True,
                                                                                         if_save=True)
        A_recons.append(A_predict.tolist())
        #print("error", error)
        print('Current iteration %i out of %i' % (step, avg_iter))
        MSEs.append(MSE)

        if (step % 5 == 4):
            reconstructor.display_dictionary_state(W1, state="overall", if_show=True, if_save=True,
                                                   foldername=foldername, filename='online')

    A_recons = np.asarray(A_recons)


    # np.save('Time_series_dictionary/' + str(foldername) + '/recons_nononline', A_recons)

    # print('change in dictionary after online learning', np.linalg.norm(W_old - W1))

    '''
    ### For loading saved checkpoints just for plotting
    W1 = np.load('Time_series_dictionary/' + str(foldername) + '/dict_learned_3_pretraining_Korea, South.npy')
    code = np.load('Time_series_dictionary/' + str(foldername) + '/code_learned_3_pretraining_Korea, South.npy')
    reconstructor.code = code
    A_recons = np.load("Time_series_dictionary/" + str(foldername) + "/recons_nononline.npy")
    '''

    ### plot minibatch-trained dictionary (from last iteration)
    #reconstructor.display_dictionary(W, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    #reconstructor.display_dictionary(W, cases='death', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    #reconstructor.display_dictionary(W, cases='residential', if_show=True, if_save=True, foldername=foldername, filename='minibatch')

    ### plot minibatch-then-online-trained dictionary (from last iteration)
    reconstructor.display_dictionary_state(W1, state="overall", if_show=True, if_save=True,foldername=foldername, filename='online')

    ### plot original and prediction curves
    reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    #reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    #reconstructor.display_prediction(source, A_recons, cases='residential', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)

def main():

    main_train_joint()

if __name__ == '__main__':
    main()

