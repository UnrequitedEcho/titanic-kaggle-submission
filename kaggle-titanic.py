import time
import re

import numpy as np
import pandas as pd

# ML Models
import lightgbm as lgb
from sklearn.cluster import KMeans

# Metrics
from sklearn.metrics import mean_absolute_error, r2_score, precision_recall_curve, confusion_matrix, classification_report
import shap
from dython.nominal import associations

# Hyperparameter Tuning
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import KFold, StratifiedKFold

# Set random seed for reproducibility
SEED = 1

# Load and combine train/test datasets for unified preprocessing
df = pd.concat([
    pd.read_csv('DataSet/train.csv'),
    pd.read_csv('DataSet/test.csv')
], sort=False, ignore_index=True)

print(f"Train Dataset Size: {len(df.loc[df['Survived'].notna()])} rows")
print(f"Test Dataset Size:  {len(df.loc[df['Survived'].isna()])} rows")

df['Survived'] = df['Survived'].astype('boolean')

# Initialize feature tracking lists
numeric_features = []
categorical_features = ['Pclass', 'Sex', 'Embarked']

# --- Feature Engineering: Extract Title from Name ---
# ----------------------------------------------------

# Map raw titles to broader social categories
title_mappings = {
    'Noble': ['Lady', 'Countess', 'Dona', 'Jonkheer', 'Don', 'Sir'],
    'Military': ['Major', 'Col', 'Capt'],
    'Professional': ['Dr', 'Rev'],
    'Miss': ['Miss', 'Ms', 'Mlle'],
    'Mrs': ['Mrs', 'Mme'],
    'Master': ['Master'],
    'Mr': ['Mr']
}
reverse_title_mappings = {
    title: group for group, titles in title_mappings.items() for title in titles
}

# Extract title from name and map to category
df['Title'] = (
    df['Name']
    .str.extract(r',\s*([A-Za-z]+)\.', expand=False)  # Capture title after comma
    .map(reverse_title_mappings)
)

# Count number of words in the name (excluding passenger's Title)
df['NbNames'] = df['Name'].str.split().str.len().sub(1)

# Register new features
categorical_features += ['Title']
numeric_features += ['NbNames']

# --- Feature Engineering: Family Relationships ---
# -------------------------------------------------

# Total number of family members onboard (including the passenger)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Flags based on who the passenger is traveling with
# True if the passenger has no family onboard
df['IsAlone'] = (df['FamilySize'] == 1)
# True if likely traveling with a spouse (adults only, and at least one sibling/spouse onboard)
df['WithSpouse'] = (~df['Title'].isin(['Miss', 'Master'])) & (df['SibSp'] > 0)
# True if likely traveling with children (adults only, and at least one parent/child onboard)
df['WithChildren'] = (~df['Title'].isin(['Miss', 'Master'])) & (df['Parch'] > 0)
# True if likely traveling with parents (minors only, and at least one parent/child onboard)
df['WithParents'] = (df['Title'].isin(['Miss', 'Master'])) & (df['Parch'] > 0)

# Register new features
categorical_features += ['IsAlone', 'WithSpouse', 'WithChildren', 'WithParents']
numeric_features += ['FamilySize']

# --- Feature Engineering: Cabin & Deck Information ---
# -----------------------------------------------------

# Extract the deck letter (first character of the cabin string)
deck_letters = df['Cabin'].str[0]

# Map deck letters to broader categories
deck_mappings = {
    'A': 'A', 'B': 'B', 'C': 'C',  # Upper decks (mostly 1st class)
    'D': 'D', 'E': 'E',            # Mid-level decks
    'F': 'Lower', 'G': 'Lower', 'T': 'Lower'  # Lower/deep decks
}
df['DeckLevel'] = deck_letters.map(deck_mappings)

# Extract numeric portion of the cabin (if present) and compute median in case of multiple cabins
df['CabinNumber'] = (
    df['Cabin']
    .fillna('')
    .str.findall(r'(\d+)')  # Find all numeric groups in cabin string
    .apply(lambda x: np.median([int(n) for n in x]) if x else pd.NA)
    .astype('Int64')
)

# Bin cabin numbers into rough horizontal locations (fore/midship/aft)
df['CabinLocation'] = pd.cut(
    df['CabinNumber'],
    bins=[0, 50, 100, float('inf')],
    labels=['Forward', 'Midship', 'Aft'],
    right=False
)

# Register new features
categorical_features += ['DeckLevel', 'CabinLocation']


# --- Feature Engineering: Ticket Prefix & Clustering ---
# --------------------------------------------------------

# Detect whether a ticket has a non-numeric prefix (e.g., "PC 17599")
df['TicketHasPrefix'] = ~df['Ticket'].str.isdigit()

# Extract the numeric part of the ticket for clustering
df['Ticket'] = (
    df['Ticket']
    .str.extract(r'^.*?(\d*)$')  # Extract last number group (may be empty)
    .replace('', '0')            # Replace empty strings with zero
    .astype('Int64')             # Convert to integer (nullable)
)

# Cluster tickets into groups using the square root to reduce variance
cluster = KMeans(n_clusters=6, random_state=SEED).fit(
    pd.DataFrame(df['Ticket'].pow(1/2))
)
df['TicketCluster'] = cluster.labels_

# Register new features
categorical_features += ['TicketHasPrefix']
numeric_features += ['TicketCluster']

# --- Feature Engineering: Fare Sharing ---
# -----------------------------------------

# Estimate group size by counting how many passengers share the same ticket
df['GroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

# Label tickets as shared or solo
df['TicketType'] = np.where(df['GroupSize'] > 1, 'Shared', 'Solo')

# Calculate fare paid per person (assumes total fare is split evenly)
df['PerPersonFare'] = df['Fare'] / df['GroupSize']

# Register new features
numeric_features += ['PerPersonFare']

# --- Feature Engineering: Group Survival Rate ---
# ------------------------------------------------

# Compute global survival rate (used as fallback for unknown outcomes)
overall_survival = df.loc[df['Survived'].notna(), 'Survived'].mean()

# For each passenger, calculate average survival rate among their ticket group
def calculate_group_survival(p_id, ticket):
    survival = []
    for p in df.itertuples():
        if p.Ticket == ticket:
            if pd.notna(p.Survived) and p.PassengerId != p_id:
                survival.append(p.Survived)
            else:
                # Use global average when survival status is unknown or it's the current passenger
                survival.append(overall_survival)
    return np.average(survival)

# Apply group survival rate function to each row
df['GroupSurvivalRate'] = df.apply(
    lambda row: calculate_group_survival(row['PassengerId'], row['Ticket']),
    axis=1
).astype('Float64')

# Register new features
numeric_features += ['GroupSurvivalRate']


# --- Feature Engineering: Age Imputation with LightGBM ---
# ---------------------------------------------------------

# Features used to predict missing ages
age_features = [
    'Pclass', 'Sex', 'Fare', 'Embarked', 'Title', 'NbNames',
    'FamilySize', 'IsAlone', 'WithSpouse', 'WithChildren', 'WithParents',
    'DeckLevel', 'CabinLocation', 'TicketHasPrefix',
    'TicketCluster', 'GroupSize', 'PerPersonFare', 'GroupSurvivalRate'
]

# Prepare feature dataset
age_df = df[age_features].copy()

# Convert categorical features to proper dtype
for col in [f for f in age_features if f in categorical_features]:
    age_df[col] = age_df[col].astype('category')

# Split dataset into known/missing age subsets
age_train = age_df[df['Age'].notna()]
age_missing = age_df[df['Age'].isna()]
y_age = df.loc[df['Age'].notna(), 'Age']

# LightGBM parameters for regression
params = {
    'objective': 'regression',
    'metric': 'mae', # Mean Absolute Error
    'num_leaves': 96,
    'learning_rate': 0.1,
    'min_child_samples': 3,
    'seed': SEED,
    'verbosity': -1
}

# Use cross-validation to find the optimal number of boosting rounds
cv_results = lgb.cv(
    params | {'num_iterations': 50000, 'early_stopping_rounds': 50},
    lgb.Dataset(age_train, label=y_age),
    nfold=5,
    stratified=False,
    seed=SEED
)

# Train final model using best iteration from CV
age_model = lgb.train(
    params | {'num_iterations': len(cv_results['valid l1-mean'])},
    lgb.Dataset(age_train, label=y_age)
)

# Predict missing ages and fill them in
predicted_ages = age_model.predict(age_missing)
df.loc[df['Age'].isna(), 'Age'] = predicted_ages

# Flag children using both title and age (to disambiguate adult "Miss")
df['IsChild'] = (
    (df['Title'] == 'Master') |
    ((df['Title'] == 'Miss') & (df['Age'] < 18))
)

# Register new features
categorical_features += ['IsChild']
numeric_features += ['Age']


# --- Preparation for Modeling ---
# --------------------------------

# Retain only relevant features + target + ID
df_train = df[categorical_features + numeric_features + ['Survived', 'PassengerId']].copy()

# Ensure categorical features are properly typed for LightGBM
df_train[categorical_features] = df_train[categorical_features].astype('category')

# Create masks for train/test splits
train_mask = df_train['Survived'].notna()
test_mask = df_train['Survived'].isna()

# Save test passenger IDs for final submission
test_ids = df_train.loc[test_mask, 'PassengerId']

# Split into training features/labels and test set
X_train = df_train.loc[train_mask].drop(columns=['Survived', 'PassengerId'])
y_train = df_train.loc[train_mask, 'Survived']
X_test = df_train.loc[test_mask].drop(columns=['Survived', 'PassengerId'])

# --- Hyperparameter Tuning with Bayesian Optimization ---
# --------------------------------------------------------

# Logger class to track timing, score, and progress of tuning
class LoggerCallback:
    def __init__(self):
        self.start_time = time.time()
        self.last_iter_time = self.start_time
        self.best_loss = float('inf')
        self.iteration = 0

    def __call__(self, res):
        current_time = time.time()
        elapsed_last = current_time - self.last_iter_time
        elapsed_total = current_time - self.start_time
        self.last_iter_time = current_time
        self.iteration += 1

        current_loss = res.func_vals[-1]
        improvement_sign = "v" if current_loss < self.best_loss else "-"
        if current_loss < self.best_loss:
            self.best_loss = current_loss

        last_time_str = f"{elapsed_last:4.1f}s".replace(' ', ' ')
        total_time_str = f"{elapsed_total:4.0f}s".replace(' ', ' ')
        param_str = " ".join([
            f"{name[:3]}:{value:>1.3f}" if isinstance(value, float) 
            else f"{name[:3]}:{value:>2}"
            for name, value in zip(search_param_names, res.x_iters[-1])
        ])

        # Display progress info
        print(
            f"Iter {self.iteration:02d} "
            f"- {last_time_str}/{total_time_str} "
            f"| Loss: {current_loss:.5f} {improvement_sign} "
            f"| Best: {self.best_loss:.5f} "
            f"| Params: {param_str}"
            f" iter:{nb_iters[tuple(res.x_iters[-1])]}"
        )

# Function used to evaluate each hyperparameter set
def train_evaluate(search_param_list):
    search_params = {
        k: v for k, v in zip(search_param_names, search_param_list)
    }

    # Perform 10-fold cross-validation to evaluate current config
    cv_results = lgb.cv(
        params=fixed_params | search_params | {
            'num_iterations': 50000,
            'early_stopping_rounds': 50
        },
        train_set=lgb.Dataset(X_train, label=y_train),
        nfold=10,
        stratified=True,
        shuffle=True,
        return_cvbooster=True,
        seed=SEED
    )

    # Store optimal iteration count for this configuration
    nb_iters[tuple(search_param_list)] = len(cv_results['valid binary_logloss-mean'])

    # Return the final validation log loss
    return cv_results['valid binary_logloss-mean'][-1]

# Parameters we will tune
search_param_names = ['num_leaves', 'min_data_in_leaf', 'colsample_bynode']

# Fixed parameters used for all configurations
fixed_params = {
    'metric': ['binary_logloss'],    # Evaluation metric
    'objective': 'binary',           # Binary classification
    'boosting': 'gbrt',              # Gradient boosting trees
    'learning_rate': 0.01,           # Slow learning for precision
    'is_unbalanced': True,           # Adjust for class imbalance
    'verbose': -1,                   # Suppress logs
    'seed': SEED                     # Reproducibility
}

# Store best iteration count for each trial
nb_iters = {}

# Perform Bayesian optimization (200 calls, 3 params)
result = gp_minimize(
    train_evaluate,
    [
        Integer(16, 64),        # num_leaves
        Integer(10, 50),        # min_data_in_leaf
        Real(0.01, 0.8),        # colsample_bynode
    ],
    callback=[LoggerCallback()],
    n_initial_points=20,
    n_calls=200,
    n_jobs=-1,
    random_state=SEED
)

# Extract best parameters and optimal iteration count
best_params = {
    k: v for k, v in zip(search_param_names, result.x)
}
best_params['num_iterations'] = nb_iters[tuple(result.x)]

# Display summary of results
print(f'Total evaluations: {len(result.func_vals)}')
print(f'Lowest Logloss: {result.fun}')
print('Best parameters:')
for k, v in best_params.items():
    print(f'{k} : {v}')


# --- Threshold Optimization for Classification ---
# -------------------------------------------------

# Instead of using the default 0.5 threshold to convert probabilities into binary predictions,
# we optimize the threshold based on validation set performance.

nfolds = 5
folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=SEED)

# Placeholder for out-of-fold predictions
oof_pred = np.zeros(len(X_train))

# Generate out-of-fold predicted probabilities using 5-fold CV
for train_idx, val_idx in tqdm(folds.split(X_train, y_train), total=nfolds):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.train(
        fixed_params | best_params,
        lgb.Dataset(X_tr, label=y_tr)
    )
    oof_pred[val_idx] = model.predict(X_val)

# Evaluate default threshold (0.5)
print('Standard threshold: 0.5')
print(classification_report(y_train, (oof_pred >= 0.5).astype(int)))

# Search for a threshold that improves accuracy
best_thresh = 0.5
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_pred)

# Filter thresholds that meet a minimum recall (e.g., ≥ 0.6 for survivors)
valid_indices = np.where(recalls[:-1] >= 0.6)[0]
if len(valid_indices) > 0:
    # Among those, choose the one with the highest precision
    best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    best_thresh = thresholds[best_idx]
    print(f"Optimal threshold: {best_thresh:.4f}")
else:
    print("No threshold meets recall constraint. Using 0.5")

# Final performance report with the optimized threshold
print(classification_report(y_train, (oof_pred >= best_thresh).astype(int)))


# --- Final Model Training & Prediction Export ---
# ------------------------------------------------

# Train final model using the entire training dataset
model = lgb.train(
    fixed_params | best_params,
    lgb.Dataset(X_train, label=y_train)
)

# Predict survival probabilities on the test set
y_pred = model.predict(X_test)

# Apply optimized threshold to convert probabilities to binary predictions
df_pred = pd.DataFrame({
    'PassengerId': test_ids.reset_index(drop=True),
    'Survived': (y_pred >= best_thresh).astype(int)
})

# Export final predictions to CSV for Kaggle submission
df_pred.to_csv(f'predictions-{SEED}.csv', index=False)
