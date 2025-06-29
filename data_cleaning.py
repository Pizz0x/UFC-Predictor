import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 1. CLEAN AND PROCESS THE DATA
def data_merging(df_event, df_fight, df_fight_stat, df_fighter):
    # 1.1 FIGHTERS:
    df_fighter['fighter_nc_dq'] = df_fighter['fighter_nc_dq'].fillna(0)
    df_fighter['fighter_stance'] = df_fighter['fighter_stance'].fillna('Unknown')
    df_fighter.drop(["fighter_nickname", "fighter_url", "fighter_f_name", "fighter_l_name"], axis=1, inplace=True)
    print(df_fighter.isna().values.any())
    #print(df_fighter.head(10))
    print("Shape fighter:", df_fighter.shape)

    
    # 1.2 FIGHT
    df_fight["gender"] = df_fight["gender"].map({'M': 0, 'F': 1})
    df_fight["title_fight"] = df_fight["title_fight"].map({'F': 0, 'T': 1})
    df_fight["winner"] = df_fight.apply(lambda row: 1 if row['winner'] == row['f_1'] else 0, axis = 1)
    def time_to_seconds(t):
        if isinstance(t, str) and ':' in t:
            minutes, seconds = map(int, t.split(':'))
            return minutes * 60 + seconds
        return np.nan

    df_fight["finish_time"] = df_fight["finish_time"].apply(time_to_seconds)
    df_fight["fight_duration"] = 300 * (df_fight["finish_round"] - 1) + df_fight["finish_time"]

    #df_fight["result_details"].fillna(df_fight["result"], inplace=True)

    def detailed_decision(row):
        if row['result'] == 'Decision':
            return row['result_details'] + ' ' + row['result']
        return row['result']
        
    df_fight['result'] = df_fight.apply(detailed_decision, axis=1)


    df_fight["weight_class"].fillna('Catch Weight', inplace=True)

    df_fight.drop(["fight_url", "finish_time", "finish_round", "referee", "result_details"], axis=1, inplace=True)
    df_fight.dropna(subset=['f_1', 'f_2'], inplace=True)
    print(df_fight.isna().values.any())
    print("Shape fight:", df_fight.shape)

    # 1.3 FIGHT STATISTICS
    df_fight_stat["ctrl_time"] = df_fight_stat["ctrl_time"].apply(time_to_seconds)
    df_fight_stat.dropna(inplace=True)
    df_fight_stat.drop(["fight_stat_id", "fight_url"], axis=1, inplace=True)
    print(df_fight_stat.isna().values.any())
    print("Shape fight stats:", df_fight_stat.shape)

    # 1.4 MERGE ALL THE DATA TOGETHER
    # fight with event
    df = df_fight.merge(df_event, on='event_id', how='left')
    # fight with fighter
    df = df.merge(df_fighter.add_suffix('_1'), left_on='f_1', right_on='fighter_id_1', how='left')
    df = df.merge(df_fighter.add_suffix('_2'), left_on='f_2', right_on='fighter_id_2', how='left')
    # fight with stats
    df = df.merge(df_fight_stat.add_suffix('_f1stat'), left_on=['fight_id', 'f_1'], right_on=['fight_id_f1stat', 'fighter_id_f1stat'], how='left')
    df = df.merge(df_fight_stat.add_suffix('_f2stat'), left_on=['fight_id', 'f_2'], right_on=['fight_id_f2stat', 'fighter_id_f2stat'], how='left')
    print("Shape finale:", df.shape)
    # Now we have a single dataset with a lot of features, let's see who they are and how they can be useful for the prediction task:
    # first of all we can delete all the ids since they are not useful for the prediction:
    df.drop(["fight_id", "event_url", "event_name", "event_city", "event_state", "event_url", "fighter_id_1", "fighter_id_2", "fight_id_f1stat", "fight_id_f2stat", "fighter_id_f1stat", "fighter_id_f2stat"], axis=1, inplace=True)
    #df.dropna(inplace=True)
    return df


# 2. FEATURE ENGINEERING
def feature_engineering(df):
    # we can see that these are all fight of the first era of UFc where probably data weren't that precise so we remove them
    # now what we want to do is keep the difference between the stats and characteristics of the 2 fighter and not the single characteristic:
    # event_date, fighter_dob_1 & fighter_dob_2 can be used to obtain the difference in age between fighter 1 and 2
    df["fighter_age_1"] = (pd.to_datetime(df['event_date'], errors='coerce') - pd.to_datetime(df['fighter_dob_1'], errors='coerce')).dt.days // 365
    df["fighter_age_2"] = (pd.to_datetime(df['event_date'], errors='coerce') - pd.to_datetime(df['fighter_dob_2'], errors='coerce')).dt.days // 365
    # event_country could be useful if we have the country of the fighter to create a field ishomecountry
    # fighter_height, fighter_weight, fighter_reach can be used for the difference between the 2 fighter

    # we can also calculate the win ratio and experience of the 2 fighter and calculate the difference bewtween them
    df["fighter_winratio_1"] = df["fighter_w_1"] / (df["fighter_w_1"] + df["fighter_l_1"] + df["fighter_d_1"])
    df["fighter_winratio_2"] = df["fighter_w_2"] / (df["fighter_w_2"] + df["fighter_l_2"] + df["fighter_d_2"])
    df["fighter_winratio_diff"] = df["fighter_winratio_1"] - df["fighter_winratio_2"]

    # Before calculating the differences between the 2 fighter, we have to make sure the values are not NaN, otherwise we will impute the value based on the median on fighters of the same category, and then compute the differences. if the column has no value for that category, then i will replace everyone with 0 since even the differences are going to be 0
    imputed_col = ['fighter_age_1', 'fighter_height_cm_1', 'fighter_weight_lbs_1', 'fighter_reach_cm_1','fighter_age_2', 'fighter_height_cm_2', 'fighter_weight_lbs_2', 'fighter_reach_cm_2']
    imputer = SimpleImputer(strategy='median')
    for weight_class in df["weight_class"].dropna().unique():
        idx = df[df["weight_class"] == weight_class].index
        subset = df.loc[idx, imputed_col]
        valid_cols = subset.columns[subset.notna().any()]
        empty_cols = subset.columns[~subset.notna().any()]
        imputed_values = imputer.fit_transform(subset[valid_cols])
        df.loc[idx, valid_cols] = imputed_values
        df.loc[idx, empty_cols] = 0

    df["fighter_height_diff"] = df["fighter_height_cm_1"] - df["fighter_height_cm_2"]
    df["fighter_weight_diff"] = df["fighter_weight_lbs_1"] - df["fighter_weight_lbs_2"]
    df["fighter_age_diff"] = df["fighter_age_1"] - df["fighter_age_2"]
    df["fighter_reach_diff"] = df["fighter_reach_cm_1"] - df["fighter_reach_cm_2"]
    df["fighter_experience_diff"] = (df["fighter_w_1"] + df["fighter_l_1"] + df["fighter_d_1"]) - (df["fighter_w_2"] + df["fighter_l_2"] + df["fighter_d_2"])

    # we can also calculate the difference in the stats of the 2 fighter for the fight: differences in # of strikes and ratio of successfull and significant strikes; same things with takedown, knockouts control_time and submission attempted.
    # this calculated features are useful only for the case where the fight already happened and just by watching the stats we want to predict the winner, in the case of a real prediction, we cannot use this fields so we will comment this rows, they are just for learning purpose
    #print(df.columns)
    # df["knockdown_diff"] = df['knockdowns_f1stat'] - df['knockdowns_f2stat']
    # df["strikes_diff"] = df['total_strikes_succ_f1stat'] - df['total_strikes_succ_f2stat']
    # df["strikes_ratio_diff"] = (df['total_strikes_succ_f1stat']/(df['total_strikes_att_f1stat']+1)) - (df['total_strikes_succ_f2stat']/(df['total_strikes_att_f2stat']+1))
    # df["sign_strikes_diff"] = df['sig_strikes_succ_f1stat'] - df['sig_strikes_succ_f2stat']
    # df["takedown_diff"] = df['takedown_succ_f1stat'] - df['takedown_succ_f2stat']
    # df["takedown_ratio_diff"] = (df['takedown_succ_f1stat']/(df['takedown_att_f1stat']+1)) - (df['takedown_succ_f2stat']/(df['takedown_att_f2stat']+1))
    # df["submission_diff"] = df["submission_att_f1stat"] - df["submission_att_f2stat"]
    # df["ctrl_time_diff"] = df["ctrl_time_f1stat"] - df["ctrl_time_f2stat"]

    # we can calculate stats for the fighter like the finish ratio and other ratio like the submission one and the ko one
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["is_finish"] = df.apply(lambda row: 1 if (row['result'] == 'KO/TKO' or row['result'] == 'Submission') else 0, axis=1)
    df["is_sub"] = df.apply(lambda row: 1 if (row['result'] == 'Submission') else 0, axis=1)
    df["is_ko"] = df.apply(lambda row: 1 if (row['result'] == 'KO/TKO') else 0, axis=1)
    df_f1 = df[df["winner"] == 1][["f_1", "event_date", "is_finish", "is_sub", "is_ko"]].copy()
    df_f1 = df_f1.rename(columns={"f_1": "fighter"})
    df_f2 = df[df["winner"] == 0][["f_2", "event_date", "is_finish", "is_sub", "is_ko"]].copy()
    df_f2 = df_f2.rename(columns={"f_2": "fighter"})
    fighter_wins = pd.concat([df_f1, df_f2], ignore_index=True)
    fighter_wins = fighter_wins.sort_values(by="event_date")
    fighter_wins["cum_finish"] = fighter_wins.groupby("fighter")["is_finish"].cumsum().shift(1)
    fighter_wins["cum_ko"] = fighter_wins.groupby("fighter")["is_ko"].cumsum().shift(1)
    fighter_wins["cum_sub"] = fighter_wins.groupby("fighter")["is_sub"].cumsum().shift(1)
    fighter_wins["cum_wins"] = fighter_wins.groupby("fighter").cumcount()
    fighter_wins['finish_ratio'] = fighter_wins.apply(lambda row: row['cum_finish'] / row['cum_wins'] if row['cum_wins'] > 0 else 0,axis=1)
    fighter_wins['ko_ratio'] = fighter_wins.apply(lambda row: row['cum_ko'] / row['cum_wins'] if row['cum_wins'] > 0 else 0, axis=1)
    fighter_wins['sub_ratio'] = fighter_wins.apply(lambda row: row['cum_sub'] / row['cum_wins'] if row['cum_wins'] > 0 else 0, axis=1)
    #print(fighter_wins.head(20))

    # we can also calculate other properties for the fighter like strikes and significant strikes per min or average control time and takedown per match to check if the guy is a striker or a grappler
    df_f1 = df[["f_1", "event_date", 'fight_duration', 'total_strikes_succ_f1stat', 'sig_strikes_succ_f1stat', 'ctrl_time_f1stat', 'takedown_succ_f1stat']].copy()
    df_f1 = df_f1.rename(columns={"f_1": "fighter", 'total_strikes_succ_f1stat':"succ_strikes", 'sig_strikes_succ_f1stat':"sign_strikes", 'ctrl_time_f1stat':"ctrl_time", 'takedown_succ_f1stat': "takedown_succ"})
    df_f2 = df[["f_2", "event_date", 'fight_duration', 'total_strikes_succ_f2stat', 'sig_strikes_succ_f2stat', 'ctrl_time_f2stat', 'takedown_succ_f2stat']].copy()
    df_f2 = df_f2.rename(columns={"f_2": "fighter", 'total_strikes_succ_f2stat':"succ_strikes", 'sig_strikes_succ_f2stat':"sign_strikes", 'ctrl_time_f2stat':"ctrl_time", 'takedown_succ_f2stat': "takedown_succ"})
    fighter_stats = pd.concat([df_f1, df_f2], ignore_index=True)
    fighter_stats = fighter_stats.sort_values(by="event_date")
    fighter_stats["cum_strikes_succ"] = fighter_stats.groupby("fighter")["succ_strikes"].cumsum().shift(1)
    fighter_stats["cum_sign_strikes"] = fighter_stats.groupby("fighter")["sign_strikes"].cumsum().shift(1)
    fighter_stats["cum_ctrl_time"] = fighter_stats.groupby("fighter")["ctrl_time"].cumsum().shift(1)
    fighter_stats["cum_takedown_succ"] = fighter_stats.groupby("fighter")["takedown_succ"].cumsum().shift(1)
    fighter_stats["cum_duration"] = fighter_stats.groupby("fighter")["fight_duration"].cumsum().shift(1)
    fighter_stats["cum_match"] = fighter_stats.groupby("fighter").cumcount()
    fighter_stats["cum_duration"] = fighter_stats["cum_duration"] / 60
    fighter_stats["strikes_per_min"] = fighter_stats["cum_strikes_succ"] / fighter_stats["cum_duration"]
    fighter_stats["sign_strikes_per_min"] = fighter_stats["cum_sign_strikes"] / fighter_stats["cum_duration"]
    fighter_stats["avg_ctrl_time"] = fighter_stats.apply(lambda row: row['cum_ctrl_time'] / row['cum_match'] if row['cum_match'] > 0 else 0, axis=1)
    fighter_stats["avg_takedowns"] = fighter_stats.apply(lambda row: row['cum_takedown_succ'] / row['cum_match'] if row['cum_match'] > 0 else 0, axis=1)
    #print(fighter_stats.head(20))


    # at this point i can merge these 2 dataframes with the original one:
    # fighter 1
    df = df.merge(
        fighter_wins.rename(columns=lambda col: f"{col}_f1" if col not in ['fighter', 'event_date'] else col),
        how='left',
        left_on=['f_1', 'event_date'],
        right_on=['fighter', 'event_date']
    ).drop(columns=["fighter"])
    df = df.merge(
        fighter_stats.rename(columns=lambda col: f"{col}_f1" if col not in ['fighter', 'event_date'] else col),
        how='left',
        left_on=['f_1', 'event_date'],
        right_on=['fighter', 'event_date']
    ).drop(columns=["fighter"])
    # fighter 2
    df = df.merge(
        fighter_wins.rename(columns=lambda col: f"{col}_f2" if col not in ['fighter', 'event_date'] else col),
        how='left',
        left_on=['f_2', 'event_date'],
        right_on=['fighter', 'event_date']
    ).drop(columns=["fighter"])
    df = df.merge(
        fighter_stats.rename(columns=lambda col: f"{col}_f2" if col not in ['fighter', 'event_date'] else col),
        how='left',
        left_on=['f_2', 'event_date'],
        right_on=['fighter', 'event_date']
    ).drop(columns=["fighter"])


    df.drop(["result", "event_country", "fighter_height_cm_1", "fighter_height_cm_2", "fighter_weight_lbs_1", "fighter_weight_lbs_2", "fighter_reach_cm_1", "fighter_reach_cm_2", "fighter_w_1", "fighter_w_2", "fighter_l_1", "fighter_l_2", "fighter_d_1", "fighter_d_2", "fighter_nc_dq_1", "fighter_nc_dq_2","event_date", "fighter_dob_1", "fighter_dob_2", "fighter_age_1", "fighter_age_2",'knockdowns_f1stat', 'total_strikes_att_f1stat','total_strikes_succ_f1stat', 'sig_strikes_att_f1stat','sig_strikes_succ_f1stat', 'takedown_att_f1stat','takedown_succ_f1stat', 'submission_att_f1stat', 'reversals_f1stat','ctrl_time_f1stat', 'knockdowns_f2stat', 'total_strikes_att_f2stat','total_strikes_succ_f2stat', 'sig_strikes_att_f2stat','sig_strikes_succ_f2stat', 'takedown_att_f2stat','takedown_succ_f2stat', 'submission_att_f2stat', 'reversals_f2stat','ctrl_time_f2stat', 'fight_duration', 'fight_duration_f1', 'succ_strikes_f1', 'sign_strikes_f1','ctrl_time_f1', 'takedown_succ_f1', 'cum_strikes_succ_f1','cum_sign_strikes_f1', 'cum_ctrl_time_f1', 'cum_takedown_succ_f1','cum_duration_f1', 'cum_match_f1','fight_duration_f2', 'succ_strikes_f2', 'sign_strikes_f2','ctrl_time_f2', 'takedown_succ_f2', 'cum_strikes_succ_f2','cum_sign_strikes_f2', 'cum_ctrl_time_f2', 'cum_takedown_succ_f2','cum_duration_f2', 'cum_match_f2','is_finish','is_finish_f1','cum_finish_f1', 'cum_ko_f1', 'cum_sub_f1', 'cum_wins_f1', 'is_finish_f2', 'cum_finish_f2','cum_ko_f2', 'cum_sub_f2', 'cum_wins_f2','is_sub', 'is_ko', 'is_sub_f1','is_ko_f1', 'is_sub_f2', 'is_ko_f2', "fighter_winratio_1", "fighter_winratio_2"], axis=1, inplace=True)

    # Now we want to handle missing values for what concerns the statistics -> NaN means that it's the first experience in UFC for the fighter
    # In this case a NaN value is significative since it means the fighter has no experience, so we will set the nan values at 0 and with another feature flag when one of the 2 fighter is at his first fight
    stat_cols = ['finish_ratio', 'ko_ratio', 'sub_ratio', 'strikes_per_min', 'avg_ctrl_time', 'avg_takedowns', 'sign_strikes_per_min']
    for col in stat_cols:
        f1 = f"{col}_f1"
        f2 = f"{col}_f2"
        diff = f"{col}_diff"
        flag1 = f"{col}_flag_f1"
        flag2 = f"{col}_flag_f2"

        df[flag1] = df[f1].isna()
        df[flag2] = df[f2].isna()
        df[f1] = df[f1].fillna(0)
        df[f2] = df[f2].fillna(0)
        df[diff] = np.where((~df[flag1]) & (~df[flag2]), df[f1]-df[f2], 0) # if at least one of them is nan then we set 0
        df.drop(columns=[flag1, flag2], inplace=True)

    #print(df.columns)
    nan_summary = df.isna().sum()
    print(nan_summary[nan_summary > 0])


    # Now that we have all the useful data and we dealt with the missing values what we should do is encode the string values using One-Hot encoding, required for the prediciton task
    oh = OneHotEncoder(sparse_output=False)
    #print(df.info())

    is_object = np.array([pd.api.types.is_object_dtype(dtype) for dtype in df.dtypes])
    categorical_idx = np.flatnonzero(is_object)
    categorical_df = df.iloc[:, categorical_idx]
    oh.fit(categorical_df)
    encoded = oh.transform(categorical_df)
    df.drop(["fighter_stance_2", "fighter_stance_1", "weight_class", "num_rounds"], axis=1, inplace=True) # we remove the categorical feature for adding them later as encoded
    for i, col in enumerate(oh.get_feature_names_out()):
        df = df.copy()
        df[col] = encoded[:, i]
    return df