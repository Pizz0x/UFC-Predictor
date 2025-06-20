import numpy as np
import pandas as pd

df_event = pd.read_csv("data/ufc_event_data.csv")
df_fight = pd.read_csv("data/ufc_fight_data.csv")
df_fight_stat = pd.read_csv("data/ufc_fight_stat_data.csv")
df_fighter = pd.read_csv("data/ufc_fighter_data.csv")

# 1. CLEAN AND PROCESS THE DATA

# 1.1 FIGHTERS:
df_fighter['fighter_nc_dq'] = df_fighter['fighter_nc_dq'].fillna(0)
df_fighter['fighter_stance'] = df_fighter['fighter_stance'].fillna('Unknown')
#df_fighter['fighter_byear'] = pd.to_datetime(df_fighter['fighter_dob'], errors='coerce').dt.year
# delete useless features
df_fighter.drop(["fighter_nickname", "fighter_url", "fighter_f_name", "fighter_l_name"], axis=1, inplace=True)
# split features in numerical and categorical:
is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in df_fighter.dtypes])  
numerical_idx = np.flatnonzero(is_numerical)
categorical_idx = np.flatnonzero(is_numerical==False) 
categorical_idx = np.delete(categorical_idx, 1)
categorical_fighter = df_fighter.iloc[:, categorical_idx]
numerical_fighter = df_fighter.iloc[:, numerical_idx]
# since it's not useful to impute the nan values of the table, we will set them at 0 and sign them as NaN values using another feature
numerical_fighter = numerical_fighter.fillna(0)
# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
oh_fighter = OneHotEncoder(sparse_output=False)
oh_fighter.fit(categorical_fighter)
encoded = oh_fighter.transform(categorical_fighter)
encoded_df = pd.DataFrame(encoded, columns=oh_fighter.get_feature_names_out(), index=numerical_fighter.index)
df_fighter = pd.concat([numerical_fighter, df_fighter[['fighter_dob']], encoded_df], axis=1)
print(df_fighter.isna().values.any())
print(df_fighter.head(10))
print("Shape fighter:", df_fighter.shape)

# 1.2 FIGHT
df_fight["gender"] = df_fight["gender"].map({'M': 0, 'F': 1})
df_fight["title_fight"] = df_fight["title_fight"].map({'F': 0, 'T': 1})
df_fight["winner"] = df_fight.apply(lambda row: 1 if row['winner'] == row['f_1'] else 0, axis = 1)
def time_to_seconds(t):
    if isinstance(t, str) and ':' in t:
        minutes, seconds = map(int, t.split(':'))
        return minutes * 60 + seconds
    return 0

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

# One-Hot Encoding
is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in df_fight.dtypes])  
categorical_idx = np.flatnonzero(is_numerical==False)
categorical_fight = df_fight.iloc[:, categorical_idx]
oh_fight = OneHotEncoder(sparse_output=False)
oh_fight.fit(categorical_fight)
encoded = oh_fight.transform(categorical_fight)
numerical_idx = np.flatnonzero(is_numerical)
numerical_fight = df_fight.iloc[:, numerical_idx]
encoded_df = pd.DataFrame(encoded, columns=oh_fight.get_feature_names_out(), index=numerical_fight.index)
df_fight = pd.concat([numerical_fight, encoded_df], axis=1)
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
df.drop(["fight_id", "event_id", "event_url", "event_name", "event_city", "event_state", "event_url", "fighter_id_1", "fighter_id_2", "fight_id_f1stat", "fight_id_f2stat", "fighter_id_f1stat", "fighter_id_f2stat"], axis=1, inplace=True)
df.dropna(inplace=True)


# 2. FEATURE ENGINEERING
# we can see that these are all fight of the first era of UFc where probably data weren't that precise so we remove them
# now what we want to do is keep the difference between the stats and characteristics of the 2 fighter and not the single characteristic:
# event_date, fighter_dob_1 & fighter_dob_2 can be used to obtain the difference in age between fighter 1 and 2
df["fighter_age_1"] = (pd.to_datetime(df['event_date'], errors='coerce') - pd.to_datetime(df['fighter_dob_1'], errors='coerce')).dt.days // 365
df["fighter_age_2"] = (pd.to_datetime(df['event_date'], errors='coerce') - pd.to_datetime(df['fighter_dob_2'], errors='coerce')).dt.days // 365
df["fighter_age_diff"] = df["fighter_age_1"] - df["fighter_age_2"]
# event_country could be useful if we have the country of the fighter to create a field ishomecountry
# fighter_height, fighter_weight, fighter_reach can be used for the difference between the 2 fighter
df["fighter_height_diff"] = df["fighter_height_cm_1"] - df["fighter_height_cm_2"]
df["fighter_weight_diff"] = df["fighter_weight_lbs_1"] - df["fighter_weight_lbs_2"]
df["fighter_reach_diff"] = df["fighter_reach_cm_1"] - df["fighter_reach_cm_2"]
# we can also calculate the win ratio and experience of the 2 fighter and calculate the difference bewtween them
df["fighter_winratio_1"] = df["fighter_w_1"] / (df["fighter_w_1"] + df["fighter_l_1"] + df["fighter_d_1"])
df["fighter_winratio_2"] = df["fighter_w_2"] / (df["fighter_w_2"] + df["fighter_l_2"] + df["fighter_d_2"])
df["fighter_experience_diff"] = (df["fighter_w_1"] + df["fighter_l_1"] + df["fighter_d_1"]) - (df["fighter_w_2"] + df["fighter_l_2"] + df["fighter_d_2"])
# we can also calculate the difference in the stats of the 2 fighter for the fight: differences in # of strikes and ratio of successfull and significant strikes; same things with takedown, knockouts control_time and submission attempted
#print(df.columns)
df["knockdown_diff"] = df['knockdowns_f1stat'] - df['knockdowns_f2stat']
df["strikes_diff"] = df['total_strikes_succ_f1stat'] - df['total_strikes_succ_f2stat']
df["strikes_ratio_diff"] = (df['total_strikes_succ_f1stat']/(df['total_strikes_att_f1stat']+1)) - (df['total_strikes_succ_f2stat']/(df['total_strikes_att_f2stat']+1))
df["sign_strikes_diff"] = df['sig_strikes_succ_f1stat'] - df['sig_strikes_succ_f2stat']
df["takedown_diff"] = df['takedown_succ_f1stat'] - df['takedown_succ_f2stat']
df["takedown_ratio_diff"] = (df['takedown_succ_f1stat']/(df['takedown_att_f1stat']+1)) - (df['takedown_succ_f2stat']/(df['takedown_att_f2stat']+1))
df["submission_diff"] = df["submission_att_f1stat"] - df["submission_att_f2stat"]
df["ctrl_time_diff"] = df["ctrl_time_f1stat"] - df["ctrl_time_f2stat"]

# we can also calculate the submission rate and other properties like strikes per min or control time to check if the guy is a striker or a grappler, to do this we have to use past event since we cannot use the present one or future ones
df["event_date"] = pd.to_datetime(df["event_date"])
finish_cols = ['result_KO/TKO', 'result_Submission']
df["is_finish"] = df[finish_cols].fillna(0).astype(int).sum(axis=1) > 0
df_f1 = df[df["winner"] == 1][["f_1", "event_date", "is_finish", "result_KO/TKO", "result_Submission"]].copy()
df_f1 = df_f1.rename(columns={"f_1": "fighter"})
df_f2 = df[df["winner"] == 0][["f_2", "event_date", "is_finish", "result_KO/TKO", "result_Submission"]].copy()
df_f2 = df_f2.rename(columns={"f_2": "fighter"})
fighter_wins = pd.concat([df_f1, df_f2], ignore_index=True)
fighter_wins = fighter_wins.sort_values(by="event_date")
fighter_wins["cum_finish"] = fighter_wins.groupby("fighter")["is_finish"].cumsum().shift(1)
fighter_wins["cum_ko"] = fighter_wins.groupby("fighter")["result_KO/TKO"].cumsum().shift(1)
fighter_wins["cum_sub"] = fighter_wins.groupby("fighter")["result_Submission"].cumsum().shift(1)
fighter_wins["cum_wins"] = fighter_wins.groupby("fighter").cumcount()
fighter_wins['finish_ratio'] = fighter_wins.apply(lambda row: row['cum_finish'] / row['cum_wins'] if row['cum_wins'] > 0 else 0,axis=1)
fighter_wins['ko_ratio'] = fighter_wins.apply(lambda row: row['cum_ko'] / row['cum_wins'] if row['cum_wins'] > 0 else 0, axis=1)
fighter_wins['sub_ratio'] = fighter_wins.apply(lambda row: row['cum_sub'] / row['cum_wins'] if row['cum_wins'] > 0 else 0, axis=1)
print(fighter_wins.head(20))


# here i can also calculate: avg_ctrl_time, strikes_per_match, significant_strikes_per_match, takedown_rate_per_match of the past match


df.drop(["fighter_height_cm_1", "fighter_height_cm_2", "fighter_weight_lbs_1", "fighter_weight_lbs_2", "fighter_reach_cm_1", "fighter_reach_cm_2", "fighter_w_1", "fighter_w_2", "fighter_l_1", "fighter_l_2", "fighter_d_1", "fighter_d_2", "fighter_nc_dq_1", "fighter_nc_dq_2","event_date", "fighter_dob_1", "fighter_dob_2", "fighter_age_1", "fighter_age_2",'knockdowns_f1stat', 'total_strikes_att_f1stat','total_strikes_succ_f1stat', 'sig_strikes_att_f1stat','sig_strikes_succ_f1stat', 'takedown_att_f1stat','takedown_succ_f1stat', 'submission_att_f1stat', 'reversals_f1stat','ctrl_time_f1stat', 'knockdowns_f2stat', 'total_strikes_att_f2stat','total_strikes_succ_f2stat', 'sig_strikes_att_f2stat','sig_strikes_succ_f2stat', 'takedown_att_f2stat','takedown_succ_f2stat', 'submission_att_f2stat', 'reversals_f2stat','ctrl_time_f2stat'], axis=1, inplace=True)
print(df.columns)
