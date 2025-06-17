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
df_fighter['fighter_byear'] = pd.to_datetime(df_fighter['fighter_dob'], errors='coerce').dt.year
# delete useless features
df_fighter.drop(["fighter_nickname", "fighter_url", "fighter_dob", "fighter_f_name", "fighter_l_name"], axis=1, inplace=True)
# split features in numerical and categorical:
is_numerical = np.array([np.issubdtype(dtype, np.number) for dtype in df_fighter.dtypes])  
numerical_idx = np.flatnonzero(is_numerical)
categorical_idx = np.flatnonzero(is_numerical==False)
categorical_fighter = df_fighter.iloc[:, categorical_idx]
numerical_fighter = df_fighter.iloc[:, numerical_idx]
# since it's not useful to impute the nan values of the table, we will set them at 0 and sign them as NaN values using another feature
numerical_fighter = pd.concat([numerical_fighter, numerical_fighter.isna().astype(int)], axis=1)
numerical_fighter = numerical_fighter.fillna(0)
# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse_output=False)
oh.fit(categorical_fighter)
encoded = oh.transform(categorical_fighter)
encoded_df = pd.DataFrame(encoded, columns=oh.get_feature_names_out(), index=numerical_fighter.index)
numerical_fighter = pd.concat([numerical_fighter, encoded_df], axis=1)
print(numerical_fighter.head(10))
