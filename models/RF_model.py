import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_cleaning import data_merging, feature_engineering

df_event = pd.read_csv("data/ufc_event_data.csv")
df_fight = pd.read_csv("data/ufc_fight_data.csv")
df_fight_stat = pd.read_csv("data/ufc_fight_stat_data.csv")
df_fighter = pd.read_csv("data/ufc_fighter_data.csv")

# dataframe to save the names of the fighter:
fighters_name = pd.read_csv("data/ufc_fighter_data.csv")
fighters_name.drop(["fighter_nickname", "fighter_height_cm", "fighter_weight_lbs", "fighter_reach_cm", "fighter_stance", "fighter_dob", "fighter_w", "fighter_l", "fighter_d", "fighter_nc_dq", "fighter_url"], axis=1, inplace=True)

df = data_merging(df_event, df_fight, df_fight_stat, df_fighter)
df = feature_engineering(df)
# Now we can pass to the prediction task training the model

# The prediction task is to predict the winner of the events -> UFC 293; UFC FN Grasso vs Shevchenko 2; UFC FN Fiziev vs Gamrot; UFC FN Dawson vs Green
# we will do that based on all the previous events of UFC:
target_events = [662, 663, 664, 665]

df_train = df[~df["event_id"].isin(target_events)]
df_pred = df[df["event_id"].isin(target_events)]
#we then have to separate the response and the predictor variables
X_train = df_train.drop(columns=["winner"])
y_train = df_train["winner"]
X_pred = df_pred.drop(columns=["winner"])
y_pred = df_pred["winner"]

#print(X.head())
# In this file we will use the Random Forest as classifier since it's an easy to implement and robust model
base_model = RandomForestClassifier()
parameters = { 'n_estimators': [50, 100],
    'max_leaf_nodes': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [5, 10],
    'bootstrap': [True, False]
    }
tuned_model = GridSearchCV(base_model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
tuned_model.fit(X_train, y_train)
print ("Best Score: {:.3f}".format(tuned_model.best_score_) )
#print("Best Params: ", tuned_model.best_params_)
test_acc = accuracy_score(y_true = y_pred, y_pred = tuned_model.predict(X_pred) )
print("Test Accuracy: {:.3f}".format(test_acc) )

for feature, importance in zip(X_train.columns, tuned_model.best_estimator_.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# now i have a dataframe fighters_name that is composed by the id and the names, our dataframe for the prediction and the response for each row. We are curious to see which are the prediction for each fight
predictions = X_pred[["f_1", "f_2"]].copy()
predictions["predicted_winner"] = tuned_model.predict(X_pred)
predictions["real_winner"] = y_pred.values

predictions = predictions.merge(fighters_name.add_suffix('_1'), left_on=["f_1"], right_on=["fighter_id_1"])
predictions = predictions.merge(fighters_name.add_suffix('_2'), left_on=["f_2"], right_on=["fighter_id_2"])
predictions = predictions[["fighter_f_name_1", "fighter_l_name_1", "fighter_f_name_2", "fighter_l_name_2","predicted_winner","real_winner"]]
print(predictions)
