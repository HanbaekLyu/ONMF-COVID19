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

    # country_list = ['Korea, South', 'China', 'US', 'Italy', 'Germany', 'Spain']
    state_list = ['California', 'New York']
    # state_list = ['California', 'Florida', 'Texas', 'New York']
    # state_list = ['CA', 'FL', 'TX', 'NY']
    # country_list = ['Russia', 'Brazil']

    # input_variable_list = ['input_hospitalized_Currently']

    data_source_list = ['COVID_ACT_NOW', 'COVID_TRACKING_PROJECT', 'JHU']
    data_source = data_source_list[1]
    if_train_fresh = True
    if_display_dict = False
    if_recons = False
    if_ONMF_timeseris_predictor_historic = True
    L = 60  ## prediction length
    num_trials = 3

    reconstructor = ONMF_timeseries_reconstructor(path=path_COVID_tracking_proj,
                                                  source=source,
                                                  data_source=data_source,
                                                  country_list=None,
                                                  state_list=state_list,
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

    ### Minibatch + Online dictionary learning
    if if_recons:
        ### Online dictionary learning and prediction
        A_recons, W1, At1, Bt1, H = reconstructor.ONMF_predictor(mode=3,
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
                                                                 minibatch_training_initialization=True,
                                                                 minibatch_alpha=3,
                                                                 minibatch_beta=1,
                                                                 print_iter=True,
                                                                 num_trials=num_trials)

        print('A_recons.shape', A_recons.shape)
        print('A_recons', A_recons)

    ### Run ONMF_prediction on the entire dataset for validation
    if if_ONMF_timeseris_predictor_historic:

        A_full_predictions_trials, W, code = reconstructor.ONMF_predictor_historic(mode=3,
                                                                                   foldername=foldername,
                                                                                   ini_dict=None,
                                                                                   ini_A=None,
                                                                                   ini_B=None,
                                                                                   beta=1,
                                                                                   a1=0,
                                                                                   # regularizer for the code in partial fitting
                                                                                   a2=0,
                                                                                   # regularizer for the code in recursive prediction
                                                                                   future_extraploation_length=7,
                                                                                   if_save=True,
                                                                                   minibatch_training_initialization=True,
                                                                                   minibatch_alpha=1,
                                                                                   minibatch_beta=1,
                                                                                   num_trials=num_trials)  # take a number of trials to generate empirical confidence interval

        print('A_full_predictions_trials.shape', A_full_predictions_trials.shape)
        print('A_full_predictions_trials', A_full_predictions_trials)

        ### plot online-trained dictionary (from last iteration)
        filename = "full_prediction_trials_" + state_list[0] + "_num_states_" + str(len(state_list))

        for state in state_list:
            reconstructor.display_dictionary_Hospital(W, state_name=state, if_show=True, if_save=True,
                                                      foldername=foldername,
                                                      filename='online_' + filename)

        ### plot original and prediction curves

        reconstructor.display_prediction_evaluation(A_full_predictions_trials[:, ], if_show=False, if_save=True,
                                                    foldername=foldername,
                                                    filename=filename, if_errorbar=True, if_evaluation=True)

    np.save("Time_series_dictionary/full_result_" + str(data_source), reconstructor.result_dict)

    # np.save('Time_series_dictionary/' + str(foldername) + '/recons_nononline', A_recons)

    # print('change in dictionary after online learning', np.linalg.norm(W_old - W1))

    '''
    ### For loading saved checkpoints just for plotting
    W1 = np.load('Time_series_dictionary/' + str(foldername) + '/dict_learned_3_pretraining_Korea, South.npy')
    code = np.load('Time_series_dictionary/' + str(foldername) + '/code_learned_3_pretraining_Korea, South.npy')
    reconstructor.code = code
    A_recons = np.load("Time_series_dictionary/" + str(foldername) + "/recons_nononline.npy")
    '''

    if if_display_dict and (data_source == 'JHU'):
        ### plot minibatch-trained dictionary (from last iteration)
        reconstructor.display_dictionary(W, cases='confirmed', if_show=True, if_save=True, foldername=foldername,
                                         filename='minibatch')
        reconstructor.display_dictionary(W, cases='death', if_show=True, if_save=True, foldername=foldername,
                                         filename='minibatch')
        reconstructor.display_dictionary(W, cases='recovered', if_show=True, if_save=True, foldername=foldername,
                                         filename='minibatch')

        ### plot minibatch-then-online-trained dictionary (from last iteration)
        reconstructor.display_dictionary(W1, cases='confirmed', if_show=True, if_save=True, foldername=foldername,
                                         filename='online')
        reconstructor.display_dictionary(W1, cases='death', if_show=True, if_save=True, foldername=foldername,
                                         filename='online')
        reconstructor.display_dictionary(W1, cases='recovered', if_show=True, if_save=True, foldername=foldername,
                                         filename='online')

        ### plot original and prediction curves
        reconstructor.display_prediction(source, A_recons, cases='confirmed', if_show=True, if_save=True,
                                         foldername=foldername, if_errorbar=True)
        reconstructor.display_prediction(source, A_recons, cases='death', if_show=True, if_save=True,
                                         foldername=foldername, if_errorbar=True)
        reconstructor.display_prediction(source, A_recons, cases='recovered', if_show=True, if_save=True,
                                         foldername=foldername, if_errorbar=True)

    elif if_display_dict and (data_source != 'JHU'):
        filename = state_list[0] + "_num_states_" + str(len(state_list))

        ### plot minibatch-trained dictionary (from last iteration)
        for state in state_list:
            """
            reconstructor.display_dictionary_Hospital(W, state_name=state, if_show=True, if_save=True,
                                                      foldername=foldername,
                                                      filename='minibatch_' + filename)
            """

            ### plot minibatch-then-online-trained dictionary (from last iteration)
            reconstructor.display_dictionary_Hospital(W1, state_name=state, if_show=True, if_save=True,
                                                      foldername=foldername,
                                                      filename='online_' + filename)

        ### plot original and prediction curves
        # filename = 'single'
        reconstructor.display_prediction_single_Hospital(A_recons, if_show=True, if_save=True, foldername=foldername,
                                                         filename=filename, if_errorbar=True)


def main():
    main_train_joint()


if __name__ == '__main__':
    main()

