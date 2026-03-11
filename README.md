# UFC Match Predictor

*UFC Match Predictor* is a ML project that aims to predict the outcomes of UFC (Ultimate Fighting Championship) matches based on hystorical data and fighter's stats.

> **⚠️ Disclaimer:** This project was ideated only with educational purposes. It's not designed for professional use or as an assistant for gambling suggestions.

## Index
- [Description](#descrizione)
- [Structure of the Project](#struttura)
- [Models](#modelli-utilizzati)

## 💡 Description <a name="descrizione"></a>
The project obtains data from previous matches using a web scraper (based on http://ufcstats.com/statistics). \\
Then thanks to a script it cleans them and with some feature engineering extract the most useful data. \\
Subsequently, different ML algorithms are trained on the obtained data to predict the winner of future matches.

## 📁 Structure of the Project <a name="struttura"></a>
* `web_scraper/` - Scripts to get raw data from the web, *they must be runned to get the data in the data folder*.
* `data_cleaning.py` - Script for data cleaning and feature engineering.
* `*_model.py` - Scripts containing the training and evalutation of the different models.

## 🤖 Models <a name="modelli-utilizzati"></a>
To evaluate which ML algorithm fits better this supervised learning tast, the following models have been implemented and compared:
* **ANN (Artificial Neural Network)** - Neural networks that catch complex non-linear relations.
* **RF (Random Forest)** - Ensamble method rubust against overfitting.
* **SV (Support Vector Machine)** - Excellent in case of the defined kernel functions works heavenly with the given data.
* **XGBoost** - Model based on decision tree optimized to gain a better accuracy.