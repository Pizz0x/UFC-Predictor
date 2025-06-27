import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data_cleaning import data_merging, feature_engineering
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

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

# In this file we will use an Artificial Neural Network for the classification task
# Neural Networks need data to be standardized, in this way they all start with the same impact
standardizer = StandardScaler()
X_train = standardizer.fit_transform(X_train)
X_pred_stand = standardizer.transform(X_pred)

model =  models.Sequential()

model.add( Input(shape=(X_train.shape[1],)) )
model.add( layers.Dense(24, activation='relu'))
model.add( layers.Dense(8, activation='relu'))
model.add( layers.Dense(1, activation='sigmoid'))

model.compile( optimizer=optimizers.Adam(learning_rate=.01),
              loss='binary_crossentropy', metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)

hist = model.fit(x=X_train, y=y_train, epochs=100, validation_split=0.2, verbose=0, callbacks=[early_stop])

test_loss, test_acc = model.evaluate(X_pred_stand, y_pred)
print(f"Test Accuracy: {test_acc:.3f}")

# Plot the accuracy and the loss function values for the training set and the validation set -> to see if there is overfit
# import matplotlib.pyplot as plt
# print(hist.history.keys())

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,6))
# axes[0].plot(hist.history['acc'], label='Train')
# axes[0].plot(hist.history['val_acc'], label='Validation')
# axes[0].set_title("Accuracy History", fontsize=18)
# axes[0].set_xlabel("Epochs", fontsize=18)
# axes[0].legend(fontsize=20)

# axes[1].plot(hist.history['loss'], label='Train')
# axes[1].plot(hist.history['val_loss'], label='Validation')
# axes[1].set_title("Loss History", fontsize=18)
# axes[1].set_xlabel("Epochs", fontsize=18)
# axes[1].legend(fontsize=20)
# plt.tight_layout()
# plt.show()

# features impact on the prediction task:
first_layer_weights = model.layers[0].get_weights()[0] # shape: (n_features, n_hidden_units) -> we need the number of features
feature_importance = np.sum(np.abs(first_layer_weights), axis=1) # compute the importance by summing the values given from each synapsis
feature_importance = feature_importance / np.sum(feature_importance) # normalization
feature_names = X_pred.columns
importance_df = pd.DataFrame({'feauture': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print(importance_df)


# The results:
predictions = X_pred[["f_1", "f_2"]].copy()
pred_probs = model.predict(X_pred_stand)
predictions["predicted_winner"] = (pred_probs > 0.5).astype(int)
predictions["real_winner"] = y_pred.values

predictions = predictions.merge(fighters_name.add_suffix('_1'), left_on=["f_1"], right_on=["fighter_id_1"])
predictions = predictions.merge(fighters_name.add_suffix('_2'), left_on=["f_2"], right_on=["fighter_id_2"])
predictions = predictions[["fighter_f_name_1", "fighter_l_name_1", "fighter_f_name_2", "fighter_l_name_2","predicted_winner","real_winner"]]
print(predictions)

# correct_pred = (predictions["predicted_winner"] == predictions["real_winner"]).sum()
# total_pred = len(predictions)
# print("Accuracy: ", correct_pred / total_pred)

