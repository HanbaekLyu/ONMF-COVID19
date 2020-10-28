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

    foldername = 'test_transfer'  ## for saving files
    # source = [path_confirmed, path_deaths, path_recovered]
    # source = [path_confirmed, path_deaths, path_recovered]
    source = [path_COVIDactnow]

    n_components = 16

    full_state_list_train = ['California']  ### List of states for learning dictionary from
    full_state_list_test = ['New York']  ### List of states for transfer-prediction

    # state_list = ['California', 'Florida', 'Texas', 'New York']

    data_source_list = ['COVID_ACT_NOW', 'COVID_TRACKING_PROJECT', 'JHU']
    data_source = data_source_list[1]
    onestep_prediction_length = 1
    moving_window_size = 14
    future_extrapolation_length = 7
    num_trials = 1


    ### Set up reconstructor class for the training state
    reconstructor_transfer = ONMF_timeseries_reconstructor(path=path_COVID_tracking_proj,
                                                           source=source,
                                                           data_source=data_source,
                                                           country_list=None,
                                                           state_list_test=full_state_list_test,
                                                           state_list_train=full_state_list_train,
                                                           alpha=1,
                                                           # L1 sparsity regularizer for minibatch and online learning
                                                           beta=1,  # default learning exponent --
                                                           # customized in both trianing and online prediction functions
                                                           # learning rate exponent in online learning -- smaller weighs new data more
                                                           n_components=n_components,
                                                           # number of dictionary elements -- rank
                                                           ONMF_iterations=10,
                                                           # number of iterations for the ONTF algorithm
                                                           ONMF_sub_iterations=2,
                                                           # number of i.i.d. subsampling for each iteration of ONTF
                                                           ONMF_batch_size=5,
                                                           # number of patches used in i.i.d. subsampling
                                                           num_patches_perbatch=100,
                                                           # number of patches per ONMF iteration (size of mini batch)
                                                           # number of patches that ONTF algorithm learns from at each iteration
                                                           patch_size=moving_window_size,
                                                           prediction_length=onestep_prediction_length,
                                                           learnevery=1,
                                                           subsample=False,
                                                           if_onlynewcases=True,
                                                           # take the derivate of the time-series of total to get new cases
                                                           if_moving_avg_data=False,
                                                           if_log_scale=False)

    ### Run ONMF_prediction on the entire dataset for validation
    # print('!!!! W1.shape', W1.shape)
    # W2 = np.concatenate(np.vsplit(W1, len(state_list_train)), axis=1) # concatenate state-wise dictionary to predict one state
    # print('!!!! W2.shape', W2.shape)

    A_full_predictions_trials, W_total_seq_trials, code = reconstructor_transfer.ONMF_predictor_historic(mode=3,
                                                                                                  foldername=foldername,
                                                                                                  learn_from_future2past=True,
                                                                                                  ini_dict=None,
                                                                                                  ini_A=None,
                                                                                                  ini_B=None,
                                                                                                  beta=1,
                                                                                                  a1=0,
                                                                                                  # regularizer for the code in partial fitting
                                                                                                  a2=0,
                                                                                                  # regularizer for the code in recursive prediction
                                                                                                  future_extrapolation_length=future_extrapolation_length,
                                                                                                  if_save=True,
                                                                                                  learning_window_cap=10,
                                                                                                  # learn from past 30 days for prediction
                                                                                                  minibatch_training_initialization=True,
                                                                                                  minibatch_alpha=1,
                                                                                                  minibatch_beta=1,
                                                                                                  online_learning=True,
                                                                                                  num_trials=num_trials)  # take a number of trials to generate empirical confidence interval

    """
    # One can input W_total_seq_trials to transfer-predict, given that the test and train time series have the same length     
    A_full_predictions_trials, W_total_seq1, code = reconstructor_transfer.ONMF_predictor_historic(mode=3,
                                                                                                  foldername=foldername,
                                                                                                  prelearned_dict_seq=W_total_seq_trials,
                                                                                                  a1=0,
                                                                                                  # regularizer for the code in partial fitting
                                                                                                  a2=0,
                                                                                                  # regularizer for the code in recursive prediction
                                                                                                  future_extrapolation_length=7,
                                                                                                  if_save=True,
                                                                                                  num_trials=num_trials)  # take a number of trials to generate empirical confidence interval
    """

    print('A_full_predictions_trials.shape', A_full_predictions_trials.shape)
    # print('A_full_predictions_trials', A_full_predictions_trials)

    ### plot original and prediction curves
    list_states_abb_train = [us_state_abbrev[state] for state in full_state_list_train]
    list_train = '-'.join(list_states_abb_train)

    list_states_abb_test = [us_state_abbrev[state] for state in full_state_list_test]
    list_test = '-'.join(list_states_abb_test)

    ### plot online-trained dictionary (from last iteration)
    filename = "final_learned_dictionary"
    reconstructor_transfer.display_dictionary_Hospital(W_total_seq_trials[-1, -1, :, :], state_name=full_state_list_train[0],
                                                       if_show=True, if_save=True,
                                                       foldername=foldername,
                                                       filename='online_' + filename)

    filename = "full_prediction_trials_" + str(num_trials) + "_" + "PL_" + str(onestep_prediction_length) + "_FEL_" + str(future_extrapolation_length)  + "_" + list_train + str(2) + list_test

    title = full_state_list_test[0] + " (" + "transfer prediction using dictionary learned from " + str(list_train) + ") \n" + "moving window size= " + str(moving_window_size) + ", 1-step prediction= " + str(onestep_prediction_length) + " days ahead" + ", " + "total prediction: " + str(future_extrapolation_length+onestep_prediction_length-1) +" days ahead"
    reconstructor_transfer.display_prediction_evaluation(A_full_predictions_trials[:, ], if_show=False,
                                                         if_save=True,
                                                         foldername=foldername,
                                                         filename=filename, if_errorbar=True,
                                                         if_evaluation=True,
                                                         title=title)

    np.save("Time_series_dictionary/full_result_" + str(data_source), reconstructor_transfer.result_dict)


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

abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))


def main():
    main_train_joint()


if __name__ == '__main__':
    main()
