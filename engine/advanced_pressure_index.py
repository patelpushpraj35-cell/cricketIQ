import os
import sys

# Suppress joblib wmic warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:\\Temp')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

def build_advanced_dataset(pack_dir=r'c:\Users\VICTUS\Downloads\klj\data_pack\Data Pack', output_feather=r'c:\Users\VICTUS\Downloads\klj\bbb_advanced.feather'):
    print("Loading data files...")
    matches = pd.read_csv(os.path.join(pack_dir, 'matches.csv'))
    comps = pd.read_csv(os.path.join(pack_dir, 'competitions.csv'))
    bbb = pd.read_csv(os.path.join(pack_dir, 'ball_by_ball.csv'))

    matches['match_date_dt'] = pd.to_datetime(matches['match_date'])
    m = matches.merge(comps[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')
    m = m.sort_values(['comp_id', 'match_date_dt', 'match_number']).reset_index(drop=True)

    # 1. Knockout, Final, Semi-Final, and Virtual Knockout Detection
    m['is_final'] = (m['stage'].str.lower() == 'final').astype(int)
    m['is_semi'] = (m['stage'].str.lower().isin(['semi final', 'semi-final'])).astype(int)
    m['is_elim_qual'] = (m['stage'].str.lower().isin(['eliminator', 'qualifier', 'super 4'])).astype(int)

    def detect_virtual_knockouts(group):
        n_matches = len(group)
        v_ko = pd.Series(0, index=group.index)
        if n_matches >= 30:
            league_mask = group['stage'].str.lower() == 'league'
            league_matches = group[league_mask]
            if len(league_matches) >= 30:
                crunch_start = int(len(league_matches) * 0.82)
                crunch_match_ids = league_matches.iloc[crunch_start:].index
                v_ko.loc[crunch_match_ids] = 1
        elif n_matches in [3, 5, 7]:
            last_match_idx = group.index[-1]
            v_ko.loc[last_match_idx] = 1
        return v_ko

    m['is_virtual_ko'] = m.groupby('comp_id', group_keys=False).apply(detect_virtual_knockouts)
    m['is_virtual_ko'] = np.where(m['is_final'] | m['is_semi'] | m['is_elim_qual'], 0, m['is_virtual_ko'])
    m['is_any_knockout'] = ((m['is_final'] == 1) | (m['is_semi'] == 1) | (m['is_elim_qual'] == 1) | (m['knock_out'] == 1) | (m['is_virtual_ko'] == 1)).astype(int)

    meta_cols = ['match_id', 'comp_name', 'stage', 'knock_out', 'is_final', 'is_semi', 'is_elim_qual', 'is_virtual_ko', 'is_any_knockout',
                 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target', 'match_format', 'season']
    bbb = bbb.merge(m[meta_cols], on='match_id', how='left')

    # Sort strictly by delivery sequence
    bbb = bbb.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

    # Cricket delivery rules
    bbb['is_legal'] = (~bbb['extras_type'].isin(['wide', 'Overstep'])).astype(int)
    bbb['ball_runs'] = bbb['runs'] + bbb['extras1_runs'] + bbb['extras_2_runs']
    bbb['bowler_runs'] = bbb['runs'] + bbb['extras1_runs'] + np.where(~bbb['extras_type'].isin(['byes', 'legbyes']), bbb['extras_2_runs'], 0)
    bbb['is_bowler_wicket'] = bbb['dismissal_type'].isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']).astype(int)
    bbb['is_wicket_num'] = bbb['is_wicket'].astype(int)
    bbb['striker_dismissed'] = ((bbb['is_wicket'] == True) & (bbb['striker_id'] == bbb['dismissed_player_id'])).astype(int)
    bbb['is_boundary'] = (bbb['is_four'] | bbb['is_six']).astype(int)
    bbb['is_dot_bat'] = ((bbb['runs'] == 0) & (bbb['is_legal'] == 1)).astype(int)
    bbb['is_dot_bowl'] = ((bbb['bowler_runs'] == 0) & (bbb['is_legal'] == 1)).astype(int)

    # Cumulative match states
    bbb['cum_runs'] = bbb.groupby(['match_id', 'innings'])['ball_runs'].cumsum()
    bbb['cum_runs_before'] = bbb['cum_runs'] - bbb['ball_runs']

    bbb['cum_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket_num'].cumsum()
    bbb['cum_wickets_before'] = bbb['cum_wickets'] - bbb['is_wicket_num']
    bbb['wickets_in_hand_before'] = np.maximum(0, 10 - bbb['cum_wickets_before'])

    bbb['cum_legal'] = bbb.groupby(['match_id', 'innings'])['is_legal'].cumsum()
    bbb['cum_legal_before'] = bbb['cum_legal'] - bbb['is_legal']

    # Total innings 1 runs
    inn1_runs = bbb[bbb['innings'] == 1].groupby('match_id')['ball_runs'].sum().rename('inn1_total')
    bbb = bbb.merge(inn1_runs, on='match_id', how='left')

    # Max scheduled balls
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

    # Innings 2 Target & Runs needed
    bbb['target'] = np.where(bbb['innings'] == 2,
                             np.where(bbb['revised_target'].notnull(), bbb['revised_target'], bbb['inn1_total'] + 1),
                             np.nan)
    bbb['runs_needed_before'] = np.maximum(0, bbb['target'] - bbb['cum_runs_before'])

    # Required Run Rate and Current Run Rate
    bbb['rrr_before'] = np.where((bbb['innings'] == 2) & (bbb['balls_remaining_before'] > 0),
                                 (bbb['runs_needed_before'] / bbb['balls_remaining_before']) * 6.0,
                                 0.0)
    bbb['crr_before'] = np.where(bbb['cum_legal_before'] > 0,
                                 (bbb['cum_runs_before'] / bbb['cum_legal_before']) * 6.0,
                                 0.0)

    # Momentum: wickets lost in preceding 12 balls (pre-delivery)
    bbb['recent_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket_num'].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).sum()
    ).fillna(0)

    print("Computing Pre-Delivery Pressure Index...")
    b_rem = bbb['balls_remaining_before']
    r_need = bbb['runs_needed_before']
    w_hand = bbb['wickets_in_hand_before']
    rrr = bbb['rrr_before']

    # --- Innings 2 ---
    z = 0.35 * (8.5 - rrr) + 0.38 * (w_hand - 5.0) + 0.08 * (b_rem / 12.0)
    wp = 1.0 / (1.0 + np.exp(-np.clip(z, -6.0, 6.0)))
    wp_leverage = 4.0 * wp * (1.0 - wp) * (0.45 + 0.55 * (120.0 - b_rem) / 120.0)
    p_rrr = np.clip((rrr - 6.0) / 10.0, 0.0, 1.0)
    p_wickets = 1.0 - np.power(np.clip(w_hand / 10.0, 0.0, 1.0), 1.25)
    p_stage = np.where(b_rem <= 30, 0.70 + 0.30 * (30.0 - b_rem) / 30.0,
              np.where(b_rem <= 60, 0.40 + 0.30 * (60.0 - b_rem) / 30.0,
                                    0.15 + 0.25 * (120.0 - b_rem) / 60.0))
    p_momentum = np.clip(bbb['recent_wickets'] / 2.5, 0.0, 1.0)

    raw_pi_inn2 = (
        0.30 * wp_leverage +
        0.25 * p_rrr +
        0.20 * p_wickets +
        0.15 * p_stage +
        0.10 * p_momentum
    ) * 100.0

    is_decided = (r_need <= 0) | (w_hand <= 0) | ((b_rem <= 6) & (r_need > 36))
    raw_pi_inn2 = np.where(is_decided, 0.0, raw_pi_inn2)

    # --- Innings 1 ---
    exp_rr_by_w = 7.0 + 0.35 * w_hand
    proj_total = bbb['cum_runs_before'] + (b_rem / 6.0) * exp_rr_by_w
    p_deficit = np.clip((175.0 - proj_total) / 45.0, 0.0, 1.0)

    over_num = bbb['overid']
    is_pp_collapse = (over_num <= 6) & (w_hand <= 7)
    is_mid_collapse = (over_num <= 10) & (w_hand <= 6)
    p_crisis = np.where(is_pp_collapse, 0.75 + 0.08 * (7 - w_hand),
               np.where(is_mid_collapse, 0.65 + 0.08 * (6 - w_hand), p_wickets))

    p_stage1 = np.where(over_num >= 16, 0.70 + 0.30 * (over_num - 16) / 4.0,
               np.where(over_num >= 7, 0.30 + 0.30 * (over_num - 7) / 8.0, 0.25))

    raw_pi_inn1 = (
        0.30 * p_deficit +
        0.25 * p_crisis +
        0.20 * p_wickets +
        0.15 * p_stage1 +
        0.10 * p_momentum
    ) * 100.0

    raw_pi = np.where(bbb['innings'] >= 3, 100.0,
             np.where(bbb['innings'] == 2, raw_pi_inn2, raw_pi_inn1))

    stakes_mult = np.where(bbb['is_final'] == 1, 1.15,
                  np.where(bbb['is_semi'] == 1, 1.12,
                  np.where(bbb['is_elim_qual'] == 1, 1.10,
                  np.where(bbb['is_virtual_ko'] == 1, 1.08,
                  np.where(bbb['match_format'] == 'T20', 1.04, 1.00)))))

    bbb['pressure_index'] = np.clip(raw_pi * stakes_mult, 0.0, 100.0)

    # Categorization: 0-39 Low, 40-59 Moderate, 60-79 High, 80-100 Extreme
    bbb['pressure_category'] = pd.cut(
        bbb['pressure_index'],
        bins=[-1, 39.99, 59.99, 79.99, 100.1],
        labels=['Low', 'Moderate', 'High', 'Extreme']
    )
    bbb['is_hp'] = (bbb['pressure_index'] >= 60.0).astype(int)

    # Partnership calculation
    bbb['wicket_group'] = bbb.groupby(['match_id', 'innings'])['is_wicket_num'].cumsum()
    bbb['partnership_runs'] = bbb.groupby(['match_id', 'innings', 'wicket_group'])['ball_runs'].cumsum() - bbb['ball_runs']

    # --- Section 5 Special Event Detectors ---
    print("Classifying Special High-Stakes Events...")
    # Over 20 runs needed at start of over 20
    o20_start_needed = bbb[(bbb['innings'] == 2) & (bbb['overid'] == 20)].groupby(['match_id'])['runs_needed_before'].transform('first')
    bbb['event_chase_20plus'] = ((bbb['innings'] == 2) & (bbb['overid'] == 20) & (o20_start_needed >= 20.0)).astype(int)
    bbb['event_defend_lt10'] = ((bbb['innings'] == 2) & (bbb['overid'] == 20) & (o20_start_needed <= 10.0)).astype(int)
    bbb['event_defend_10_15'] = ((bbb['innings'] == 2) & (bbb['overid'] == 20) & (o20_start_needed > 10.0) & (o20_start_needed <= 15.0)).astype(int)

    bbb['event_collapse_ge2'] = ((bbb['overid'] <= 6) & (bbb['cum_wickets_before'] >= 2)).astype(int)
    bbb['event_collapse_ge3'] = ((bbb['overid'] <= 6) & (bbb['cum_wickets_before'] >= 3)).astype(int)
    bbb['event_high_rrr'] = ((bbb['innings'] == 2) & (bbb['rrr_before'] >= 11.0)).astype(int)
    bbb['event_death_bat'] = (bbb['overid'] >= 17).astype(int)
    bbb['event_death_bowl'] = (bbb['overid'] >= 17).astype(int)

    bbb['event_partnership_rescue'] = (
        (bbb['cum_wickets_before'] >= 2) & (bbb['overid'] <= 14) & (bbb['partnership_runs'] >= 30)
    ).astype(int)

    bbb['event_dangerous_stand'] = (
        (bbb['partnership_runs'] >= 45) | ((bbb['cum_runs_before'] >= 75) & (bbb['cum_wickets_before'] <= 2))
    ).astype(int)
    bbb['event_break_dangerous_stand'] = ((bbb['is_bowler_wicket'] == 1) & (bbb['event_dangerous_stand'] == 1)).astype(int)

    # --- Section 6 Expected-Performance Prediction Models (GroupKFold Out-Of-Fold) ---
    print("Training Machine Learning Expected-Performance Models (HistGradientBoosting GroupKFold)...")
    feature_cols = [
        'overid', 'innings', 'balls_remaining_before', 'wickets_in_hand_before',
        'cum_runs_before', 'rrr_before', 'crr_before', 'batting_order', 'pressure_index', 'is_any_knockout'
    ]
    X = bbb[feature_cols].copy()
    X['batting_order'] = X['batting_order'].fillna(6)

    gkf = GroupKFold(n_splits=5)
    groups = bbb['match_id']

    bbb['exp_runs'] = 0.0
    bbb['exp_boundary'] = 0.0
    bbb['exp_dot_bat'] = 0.0
    bbb['exp_bowler_runs'] = 0.0
    bbb['exp_wicket'] = 0.0
    bbb['exp_dot_bowl'] = 0.0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

        # 1. Batter Runs
        m_r = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        m_r.fit(X_tr, bbb['runs'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_runs'] = m_r.predict(X_val)

        # 2. Batter Boundary Probability
        m_b = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        m_b.fit(X_tr, bbb['is_boundary'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_boundary'] = m_b.predict_proba(X_val)[:, 1]

        # 3. Batter Dot Ball Probability
        m_d = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        m_d.fit(X_tr, bbb['is_dot_bat'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_dot_bat'] = m_d.predict_proba(X_val)[:, 1]

        # 4. Bowler Runs Conceded
        m_br = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        m_br.fit(X_tr, bbb['bowler_runs'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_bowler_runs'] = m_br.predict(X_val)

        # 5. Bowler Wicket Probability
        m_w = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        m_w.fit(X_tr, bbb['is_bowler_wicket'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_wicket'] = m_w.predict_proba(X_val)[:, 1]

        # 6. Bowler Dot Ball Probability
        m_db = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        m_db.fit(X_tr, bbb['is_dot_bowl'].iloc[train_idx])
        bbb.loc[val_idx, 'exp_dot_bowl'] = m_db.predict_proba(X_val)[:, 1]

    # Post-process bounds
    bbb['exp_runs'] = np.clip(bbb['exp_runs'], 0.0, 6.0)
    bbb['exp_bowler_runs'] = np.clip(bbb['exp_bowler_runs'], 0.0, 6.0)
    bbb['exp_boundary'] = np.clip(bbb['exp_boundary'], 0.0, 1.0)
    bbb['exp_dot_bat'] = np.clip(bbb['exp_dot_bat'], 0.0, 1.0)
    bbb['exp_wicket'] = np.clip(bbb['exp_wicket'], 0.0, 1.0)
    bbb['exp_dot_bowl'] = np.clip(bbb['exp_dot_bowl'], 0.0, 1.0)

    # Calculate Value Added Metrics
    bbb['runs_above_exp'] = bbb['runs'] - bbb['exp_runs']
    bbb['boundary_above_exp'] = bbb['is_boundary'] - bbb['exp_boundary']
    bbb['dot_below_exp'] = bbb['exp_dot_bat'] - bbb['is_dot_bat']

    bbb['runs_prevented_exp'] = bbb['exp_bowler_runs'] - bbb['bowler_runs']
    bbb['wickets_above_exp'] = bbb['is_bowler_wicket'] - bbb['exp_wicket']
    bbb['dot_above_exp'] = bbb['is_dot_bowl'] - bbb['exp_dot_bowl']

    # Win Probability Added
    bbb['wpa_bat'] = np.where(bbb['innings'] == 2,
                              (bbb['ball_runs'] - bbb['exp_runs']) * 0.035 - (bbb['striker_dismissed'] * 0.12),
                              (bbb['ball_runs'] - bbb['exp_runs']) * 0.020 - (bbb['striker_dismissed'] * 0.08))
    bbb['wpa_bowl'] = -bbb['wpa_bat']

    print(f"Saving enriched dataset to {output_feather}...")
    bbb.to_feather(output_feather)
    print("Advanced dataset built successfully!")
    return bbb


def compute_advanced_pressure_index(pack_dir=r'c:\Users\VICTUS\Downloads\klj\data_pack\Data Pack',
                                  output_feather=r'c:\Users\VICTUS\Downloads\klj\bbb_advanced.feather'):
    """Backward-compatible wrapper used by the evaluation tests and scripts."""
    return build_advanced_dataset(pack_dir=pack_dir, output_feather=output_feather)


if __name__ == '__main__':
    build_advanced_dataset()
