# Titanic — From Data to Deck

This repository contains the code behind my submission to Kaggle’s iconic [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The goal is to predict who survived the sinking of the RMS Titanic, using only the features available at boarding time.

The solution leans on thorough feature engineering, paired with a LightGBM model, Bayesian hyperparameter tuning, and threshold optimization. Model interpretation is provided by a set of SHAP visualizations.

Alongside the full notebook, you’ll find a streamlined, well-commented Python script for reproducibility.

For the full story, including the modeling process, design decisions, and the human context behind the dataset, check out the accompanying writeup: : [From Data to Deck — How I Hit Above 0.8 on the Titanic Challenge, and How You Can Too](http://stacktracesofalife.com/posts/titanic-competition-submission-writeup/).

🛠️ Tools used:
- Python & pandas
- LightGBM
- KMeans (sklearn)
- Bayesian optimization (skopt)
- SHAP
- Stubbornnes
