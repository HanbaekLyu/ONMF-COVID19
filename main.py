#!/users/colou/Miniconda3/tensorflow/python.exe

from utils.time_series_ONMF_COVID19 import time_series_tensor
import numpy as np


def main_train_joint():
    print('pre-training temporal dictionary started...')

    path_confirmed = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    foldername = 'test'  ## for saving files
    # source = [path_confirmed, path_deaths, path_recovered]
    source = [path_confirmed, path_deaths, path_recovered]

    n_components = 50

    country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'Spain']
    # country_list = ['Russia', 'Brazil']
    reconstructor = time_series_tensor(path=path_confirmed,
                                       source=source,
                                       country_list=country_list,
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
                                       if_log_scale=True)

    L = 30 ## prediction length
    avg_iter = 2
    A_recons = []
    # W_old = W.copy()
    for step in np.arange(avg_iter):

        ### Clear up the fields
        reconstructor.W = None
        reconstructor.code = np.zeros(shape=(n_components, 100))

        ### Minibatch dictionary learning
        W, At, Bt, H = reconstructor.train_dict(mode=3,
                                                foldername=foldername,
                                                alpha=1,  ## L1 regularizer for sparse coding
                                                beta=1,  ## learning rate exponent in pre-learning
                                                learn_joint_dict=True)

        ### Online dictionary learning and predictoin
        A_predict, error, W1, At1, Bt1, H = reconstructor.online_learning_and_prediction(mode=3,
                                                                                         ini_dict=W,
                                                                                         foldername=foldername,
                                                                                         beta=4,  # no effect if "if_learn_online" is false
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
    reconstructor.display_dictionary(W, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    reconstructor.display_dictionary(W, cases='death', if_show=True, if_save=True, foldername=foldername, filename='minibatch')
    reconstructor.display_dictionary(W, cases='recovered', if_show=True, if_save=True, foldername=foldername, filename='minibatch')

    ### plot minibatch-then-online-trained dictionary (from last iteration)
    reconstructor.display_dictionary(W1, cases='confirmed', if_show=True, if_save=True, foldername=foldername, filename='online')
    reconstructor.display_dictionary(W1, cases='death', if_show=True, if_save=True, foldername=foldername, filename='online')
    reconstructor.display_dictionary(W1, cases='recovered', if_show=True, if_save=True, foldername=foldername, filename='online')

    ### plot original and prediction curves
    reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)
    reconstructor.display_prediction(source, A_recons, cases='recovered', if_show=True, if_save=True, foldername=foldername, if_errorbar=True)

def main():

    main_train_joint()

if __name__ == '__main__':
    main()

