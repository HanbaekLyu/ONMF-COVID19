#!/users/colou/Miniconda3/tensorflow/python.exe

from utils.time_series_ONMF_COVID19 import ONMF_timeseries_reconstructor
import numpy as np


def main_train_joint():
    print('pre-training temporal dictionary started...')

    path_confirmed = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    path_deaths = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    path_recovered = "Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

    path_COVIDactnow = "Data/states.NO_INTERVENTION.timeseries.csv"
    path_COVID_tracking_proj = "Data/us_states_COVID_tracking_project.csv"
    # path_NYT_data  = "Data/NYT_us-states.csv"

    foldername = 'test3'  ## for saving files
    # source = [path_confirmed, path_deaths, path_recovered]
    # source = [path_confirmed, path_deaths, path_recovered]
    source = [path_COVIDactnow]

    n_components = 16

    state_list_1 = ['California']  ### List of states for learning dictionary from
    state_list_2 = ['New York']  ### List of states for transfer-prediction

    # state_list = ['California', 'Florida', 'Texas', 'New York']


    data_source_list = ['COVID_ACT_NOW', 'COVID_TRACKING_PROJECT', 'JHU']
    data_source = data_source_list[1]
    if_train_fresh = True
    if_display_dict = False
    if_recons = False
    if_ONMF_predictor_historic = True
    L = 60  ## prediction length
    num_trials = 1


    ### Set up reconstructor class for the training set
    reconstructor_train = ONMF_timeseries_reconstructor(path=path_COVID_tracking_proj,
                                                  source=source,
                                                  data_source=data_source,
                                                  country_list=None,
                                                  state_list=state_list_1,
                                                  alpha=3,  # L1 sparsity regularizer for minibatch and online learning
                                                  beta=1,  # default learning exponent --
                                                  # customized in both trianing and online prediction functions
                                                  # learning rate exponent in online learning -- smaller weighs new data more
                                                  n_components=n_components,  # number of dictionary elements -- rank
                                                  ONMF_iterations=50,  # number of iterations for the ONTF algorithm
                                                  ONMF_sub_iterations=2,
                                                  # number of i.i.d. subsampling for each iteration of ONTF
                                                  ONMF_batch_size=50,  # number of patches used in i.i.d. subsampling
                                                  num_patches_perbatch=100,
                                                  # number of patches per ONMF iteration (size of mini batch)
                                                  # number of patches that ONTF algorithm learns from at each iteration
                                                  patch_size=6,
                                                  prediction_length=1,
                                                  learnevery=1,
                                                  subsample=False,
                                                  if_onlynewcases=True,
                                                  # take the derivate of the time-series of total to get new cases
                                                  if_moving_avg_data=False,
                                                  if_log_scale=True)

    ### Online dictionary learning and prediction
    A_recons, W1, At1, Bt1, H = reconstructor_train.ONMF_predictor(mode=3,
                                                             ini_dict=None,
                                                             foldername=foldername,
                                                             beta=4,
                                                             # no effect if "if_learn_online" is false
                                                             ini_A=None,
                                                             ini_B=None,
                                                             a1=0,
                                                             # regularizer for training
                                                             a2=0,
                                                             # regularizer for prediction
                                                             future_extraploation_length=L,
                                                             if_learn_online=True,
                                                             if_save=True,
                                                             if_recons=False,
                                                             minibatch_training_initialization=True,
                                                             minibatch_alpha=3,
                                                             minibatch_beta=1,
                                                             print_iter=True,
                                                             online_learning=True,
                                                             num_trials=num_trials)

    print('A_recons.shape', A_recons.shape)
    print('A_recons', A_recons)


    ### Set up reconstructor class for the test set
    reconstructor_test = ONMF_timeseries_reconstructor(path=path_COVID_tracking_proj,
                                                  source=source,
                                                  data_source=data_source,
                                                  country_list=None,
                                                  state_list=state_list_2,
                                                  alpha=3,  # L1 sparsity regularizer for minibatch and online learning
                                                  beta=1,  # default learning exponent --
                                                  # customized in both trianing and online prediction functions
                                                  # learning rate exponent in online learning -- smaller weighs new data more
                                                  n_components=n_components,  # number of dictionary elements -- rank
                                                  ONMF_iterations=50,  # number of iterations for the ONTF algorithm
                                                  ONMF_sub_iterations=2,
                                                  # number of i.i.d. subsampling for each iteration of ONTF
                                                  ONMF_batch_size=50,  # number of patches used in i.i.d. subsampling
                                                  num_patches_perbatch=100,
                                                  # number of patches per ONMF iteration (size of mini batch)
                                                  # number of patches that ONTF algorithm learns from at each iteration
                                                  patch_size=6,
                                                  prediction_length=1,
                                                  learnevery=1,
                                                  subsample=False,
                                                  if_onlynewcases=True,
                                                  # take the derivate of the time-series of total to get new cases
                                                  if_moving_avg_data=False,
                                                  if_log_scale=True)

    ### Run ONMF_prediction on the entire dataset for validation
    if if_ONMF_predictor_historic:

        A_full_predictions_trials, W, code = reconstructor_test.ONMF_predictor_historic(mode=3,
                                                                                   foldername=foldername,
                                                                                   ini_dict=W1,
                                                                                   ini_A=At1,
                                                                                   ini_B=Bt1,
                                                                                   beta=1,
                                                                                   a1=0,
                                                                                   # regularizer for the code in partial fitting
                                                                                   a2=0,
                                                                                   # regularizer for the code in recursive prediction
                                                                                   future_extraploation_length=7,
                                                                                   if_save=True,
                                                                                   if_recons=False,
                                                                                   minibatch_training_initialization=False,
                                                                                   minibatch_alpha=1,
                                                                                   minibatch_beta=1,
                                                                                   online_learning=True,
                                                                                   num_trials=num_trials)  # take a number of trials to generate empirical confidence interval

        print('A_full_predictions_trials.shape', A_full_predictions_trials.shape)
        print('A_full_predictions_trials', A_full_predictions_trials)

        ### plot online-trained dictionary (from last iteration)
        filename = "full_prediction_trials_" + state_list_2[0] + "_num_states_" + str(len(state_list_2))

        for state in state_list_2:
            reconstructor_test.display_dictionary_Hospital(W, state_name=state, if_show=True, if_save=True,
                                                      foldername=foldername,
                                                      filename='online_' + filename)

        ### plot original and prediction curves

        reconstructor_test.display_prediction_evaluation(A_full_predictions_trials[:, ], if_show=False, if_save=True,
                                                    foldername=foldername,
                                                    filename=filename, if_errorbar=True, if_evaluation=True)

    np.save("Time_series_dictionary/full_result_" + str(data_source), reconstructor_test.result_dict)


def main():
    main_train_joint()


if __name__ == '__main__':
    main()

