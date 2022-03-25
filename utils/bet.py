# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 21:45
# @Author  : nieyuzhou
# @File    : bet.py
# @Software: PyCharm
import numpy as np
import pandas as pd

from utils.get_data import create_feables, get_bookkeeper_probs


def get_reward(choice, matches):
    """ Get the reward of a given bet. """

    # Identify bet
    match = matches[matches.match_api_id == choice.match_api_id]
    bet_data = match.loc[:, (match.columns.str.contains(choice.bookkeeper))]
    cols = bet_data.columns.values
    cols[:3] = ['win', 'draw', 'defeat']
    bet_data.columns = cols

    # Identfiy bet type and get quota
    if choice.bet == 'Win':
        bet_quota = bet_data.win.values
    elif choice.bet == 'Draw':
        bet_quota = bet_data.draw.values
    elif choice.bet == 'Defeat':
        bet_quota = bet_data.defeat.values
    else:
        raise TypeError

    # Check label and compute reward
    if choice.bet == choice.label:
        reward = bet_quota
    else:
        reward = 0

    # Return reward
    return reward


def execute_bets(bet_choices, matches):
    """ Get rewards for all bets. """

    total_reward = 0
    total_invested = 0
    # 进入循环
    loops = np.arange(0, bet_choices.shape[0])
    for i in loops:
        # Get rewards and accumulate profit
        reward = get_reward(bet_choices.iloc[i, :], matches)
        total_reward = total_reward + reward
        total_invested += 1

    # Compute investment return
    investment_return = float(total_reward / total_invested) - 1

    return investment_return


def compare_probabilities(clf, dim_reduce, bk, bookkeepers, matches, fifa_data):
    """ Map bookkeeper and model probabilities. """

    # Create features and labels for given matches
    feables = create_feables(matches, fifa_data, bk, get_overall = True)

    # Ensure consistency
    match_ids = list(feables['match_api_id'])
    matches = matches[matches['match_api_id'].isin(match_ids)]

    # Get bookkeeper probabilities

    print("Obtaining bookkeeper probabilities...")
    bookkeeper_probs = get_bookkeeper_probs(matches, bookkeepers)
    bookkeeper_probs.reset_index(inplace = True, drop = True)

    inputs = feables.drop('match_api_id', axis = 1)
    labels = inputs.loc[:, 'label']
    features = inputs.drop('label', axis = 1)

    # Get model probabilities
    print("Predicting probabilities based on model...")
    model_probs = pd.DataFrame()
    label_table = pd.Series()
    temp_probs = pd.DataFrame(clf.predict_proba(dim_reduce.transform(features)),
                              columns = ['win_prob', 'draw_prob', 'defeat_prob'])
    for _ in bookkeepers:
        model_probs = model_probs.append(temp_probs, ignore_index = True)
        label_table = label_table.append(labels)
    model_probs.reset_index(inplace = True, drop = True)
    label_table.reset_index(inplace = True, drop = True)
    bookkeeper_probs['win_prob'] = model_probs['win_prob']
    bookkeeper_probs['draw_prob'] = model_probs['draw_prob']
    bookkeeper_probs['defeat_prob'] = model_probs['defeat_prob']
    bookkeeper_probs['label'] = label_table

    # Aggregate win probabilities for each match
    wins = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Win', 'win_prob', 'label']]
    wins.loc[:, 'bet'] = 'Win'
    wins = wins.rename(columns = {'Win': 'bookkeeper_prob',
                                  'win_prob': 'model_prob'})

    # Aggregate draw probabilities for each match
    draws = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Draw', 'draw_prob', 'label']]
    draws.loc[:, 'bet'] = 'Draw'
    draws = draws.rename(columns = {'Draw': 'bookkeeper_prob',
                                    'draw_prob': 'model_prob'})

    # Aggregate defeat probabilities for each match
    defeats = bookkeeper_probs[['bookkeeper', 'match_api_id', 'Defeat', 'defeat_prob', 'label']]
    defeats.loc[:, 'bet'] = 'Defeat'
    defeats = defeats.rename(columns = {'Defeat': 'bookkeeper_prob',
                                        'defeat_prob': 'model_prob'})

    total = pd.concat([wins, draws, defeats])

    # Return total
    return total


def find_good_bets(clf, dim_reduce, bk, bookkeepers, matches, fifa_data, percentile, prob_cap, verbose = False):
    """ Find good bets for a given classifier and matches. """

    # Compare model and classifier probabilities
    probs = compare_probabilities(clf, dim_reduce, bk, bookkeepers, matches, fifa_data)
    probs.loc[:, 'prob_difference'] = probs.loc[:, "model_prob"] - probs.loc[:, "bookkeeper_prob"]

    # Sort by createst difference to identify most underestimated bets
    values = probs['prob_difference']
    values = values.sort_values(ascending = False)
    values.reset_index(inplace = True, drop = True)

    if verbose:
        print("Selecting attractive bets...")

    # Identify choices that fulfill requirements such as positive difference, minimum probability and match outcome
    relevant_choices = probs[(probs.prob_difference > 0) & (probs.model_prob > prob_cap) & (probs.bet != "Draw")]

    # Select given percentile of relevant choices
    top_percent = 1 - percentile
    choices = relevant_choices[
        relevant_choices.prob_difference >= relevant_choices.prob_difference.quantile(top_percent)]
    choices.reset_index(inplace = True, drop = True)

    # Return choices
    return choices


def optimize_betting(best_clf, best_dm_reduce, bk_cols_selected, bk_cols, match_data, fifa_data,
                     n_samples, sample_size, parameter_1_grid, parameter_2_grid):
    """ Tune parameters of bet selection algorithm. """

    # Generate data samples
    samples = []
    for i in range(0, n_samples):
        sample = match_data.sample(n = sample_size, random_state = 42)
        samples.append(sample)

    results = pd.DataFrame(columns = ["parameter_1", "parameter_2", "results"])
    row = 0

    # Iterate over all 1 parameter
    for i in parameter_1_grid:
        # Iterate over all 2 parameter
        for j in parameter_2_grid:

            # Compute average score over all samples
            profits = []
            for sample in samples:
                choices = find_good_bets(best_clf, best_dm_reduce, bk_cols_selected, bk_cols, sample, fifa_data, i, j)
                profit = execute_bets(choices, match_data)
                profits.append(profit)
            result = np.mean(np.array(profits))
            results.loc[row, "results"] = result
            results.loc[row, "parameter_1"] = i
            results.loc[row, "parameter_2"] = j
            row = row + 1
            print("Simulated parameter combination: {}".format(row))

    best_result = results.ix[results['results'].idxmax()]
    return best_result
