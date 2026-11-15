import os
import pandas as pd
import numpy as np

pack_dir = r'c:\Users\VICTUS\Downloads\klj\data_pack\Data Pack'
matches = pd.read_csv(os.path.join(pack_dir, 'matches.csv'))
comps = pd.read_csv(os.path.join(pack_dir, 'competitions.csv'))
bbb = pd.read_csv(os.path.join(pack_dir, 'ball_by_ball.csv'))

# Merge matches with competitions
m = matches.merge(comps[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')

# Merge match metadata into ball_by_ball
bbb = bbb.merge(m[['match_id', 'comp_name', 'stage', 'knock_out', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target', 'match_format', 'season']], on='match_id', how='left')

# Sort properly
bbb = bbb.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Delivery types and runs
bbb['is_legal'] = (~bbb['extras_type'].isin(['wide', 'Overstep'])).astype(int)
bbb['ball_runs'] = bbb['runs'] + bbb['extras1_runs'] + bbb['extras_2_runs']
bbb['bowler_runs'] = bbb['runs'] + bbb['extras1_runs'] + np.where(~bbb['extras_type'].isin(['byes', 'legbyes']), bbb['extras_2_runs'], 0)
bbb['is_bowler_wicket'] = bbb['dismissal_type'].isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']).astype(int)
bbb['is_wicket_num'] = bbb['is_wicket'].astype(int)

# Cumulative states before ball
bbb['cum_runs'] = bbb.groupby(['match_id', 'innings'])['ball_runs'].cumsum()
bbb['cum_runs_before'] = bbb['cum_runs'] - bbb['ball_runs']

bbb['cum_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket_num'].cumsum()
bbb['cum_wickets_before'] = bbb['cum_wickets'] - bbb['is_wicket_num']

bbb['cum_legal'] = bbb.groupby(['match_id', 'innings'])['is_legal'].cumsum()
bbb['cum_legal_before'] = bbb['cum_legal'] - bbb['is_legal']

# Innings 1 total runs per match
inn1_runs = bbb[bbb['innings'] == 1].groupby('match_id')['ball_runs'].sum().rename('inn1_total')
bbb = bbb.merge(inn1_runs, on='match_id', how='left')

# Max scheduled balls per innings
def calc_max_balls(row):
    if row['innings'] >= 3:
        return 6
    if row['innings'] == 1:
        if pd.notnull(row['first_inn_revised_overs']):
            return int(row['first_inn_revised_overs'])
        return 120
    else:
        if pd.notnull(row['second_inn_revised_overs']):
            return int(row['second_inn_revised_overs'])
        return 120

bbb['max_balls'] = bbb.apply(calc_max_balls, axis=1)
bbb['balls_remaining_before'] = np.maximum(0, bbb['max_balls'] - bbb['cum_legal_before'])

# Target for innings 2
bbb['target'] = np.where(bbb['innings'] == 2,
                         np.where(bbb['revised_target'].notnull(), bbb['revised_target'], bbb['inn1_total'] + 1),
                         np.nan)
bbb['runs_needed_before'] = np.maximum(0, bbb['target'] - bbb['cum_runs_before'])

# RRR and CRR
bbb['rrr_before'] = np.where((bbb['innings'] == 2) & (bbb['balls_remaining_before'] > 0),
                             (bbb['runs_needed_before'] / bbb['balls_remaining_before']) * 6.0,
                             np.nan)
bbb['crr_before'] = np.where(bbb['cum_legal_before'] > 0,
                             (bbb['cum_runs_before'] / bbb['cum_legal_before']) * 6.0,
                             0.0)

# Vectorized Pressure Index Computation
# Innings 2:
rrr = bbb['rrr_before'].fillna(0)
p_rrr = np.clip((rrr - 6.0) / 8.0, 0.0, 1.0)
p_w = np.power(np.clip(bbb['cum_wickets_before'] / 9.0, 0.0, 1.0), 1.2)

b_rem = bbb['balls_remaining_before']
p_phase2 = np.where(b_rem <= 30, 0.70 + 0.30 * (30 - b_rem) / 30.0,
           np.where(b_rem <= 60, 0.40 + 0.30 * (60 - b_rem) / 30.0,
                                 0.15 + 0.25 * (120 - b_rem) / 60.0))

pi_inn2 = 0.40 * p_rrr + 0.35 * p_w + 0.25 * p_phase2
# Close death chase boost
close_chase = (b_rem <= 24) & (bbb['runs_needed_before'] > 0) & (bbb['runs_needed_before'] <= 2.2 * b_rem) & (bbb['cum_wickets_before'] <= 8)
pi_inn2 = np.where(close_chase, np.minimum(1.0, pi_inn2 + 0.15 * (24 - b_rem) / 24.0), pi_inn2)
# Dampen if already won or lost/impossible
won_or_dead = (bbb['runs_needed_before'] <= 0) | (bbb['cum_wickets_before'] >= 10) | ((b_rem <= 6) & (bbb['runs_needed_before'] > 36))
pi_inn2 = np.where(won_or_dead, 0.0, pi_inn2)

# Innings 1:
w_lost1 = bbb['cum_wickets_before']
over_num = bbb['overid']
# Collapse indicators
pp_collapse = (over_num <= 6) & (w_lost1 >= 3)
mid_collapse = (over_num <= 12) & (w_lost1 >= 4)
general_w = np.power(np.clip(w_lost1 / 9.0, 0.0, 1.0), 1.3)
p_crisis = np.maximum(general_w, np.where(pp_collapse, 0.75 + 0.08 * (w_lost1 - 3),
                                 np.where(mid_collapse, 0.70 + 0.08 * (w_lost1 - 4), 0.0)))

p_phase1 = np.where(over_num >= 16, 0.70 + 0.30 * (over_num - 16) / 4.0,
           np.where(over_num >= 7, 0.30 + 0.30 * (over_num - 7) / 8.0,
                                   0.25))
crr = bbb['crr_before']
p_rate1 = np.where(over_num >= 6, np.clip((7.2 - crr) / 3.0, 0.0, 0.4), 0.0)

pi_inn1 = 0.45 * p_crisis + 0.40 * p_phase1 + 0.15 * p_rate1
pi_inn1 = np.where(w_lost1 >= 10, 0.0, pi_inn1)

# Super overs (innings >= 3)
pi_base = np.where(bbb['innings'] >= 3, 1.0, np.where(bbb['innings'] == 2, pi_inn2, pi_inn1))

# Match stakes multiplier
knockout_stages = ['Final', 'Qualifier', 'Eliminator', 'Semi Final', 'Super 4']
m_stakes = np.where(bbb['knock_out'] == 1, 1.15,
           np.where(bbb['stage'].isin(knockout_stages), 1.15,
           np.where(bbb['match_format'] == 'T20', 1.05, 1.00)))

bbb['pressure_index'] = np.clip(pi_base * m_stakes, 0.0, 1.0)

print('Pressure Index distribution:')
print(bbb['pressure_index'].describe(percentiles=[0.25, 0.5, 0.7, 0.75, 0.8, 0.9]))
print('\nPercentage of balls with PI >= 0.60: {:.2f}%'.format((bbb['pressure_index'] >= 0.60).mean() * 100))
print('Percentage of balls with PI >= 0.65: {:.2f}%'.format((bbb['pressure_index'] >= 0.65).mean() * 100))

# Save processed dataframe for fast iterative analysis
bbb.to_feather(r'c:\Users\VICTUS\Downloads\klj\bbb_processed.feather')
print('Processed data saved to feather format successfully.')
