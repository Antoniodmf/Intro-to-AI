import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from time import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import Counter
from xgboost import XGBRegressor

if __name__ == "__main__":

    # Creates a constant seed so the results can be reproductible
    SEED = 42

    np.random.seed(SEED)

    print("Initiating preprocessing of the dataset... \n")
    preproc_time = time()

    base_dir = os.path.dirname(os.path.abspath(__file__))       # Path for this python file
    dataset_path = os.path.join(base_dir ,'the-movies-dataset/versions/7/')      # Path for the dataset

    metadata = pd.read_csv(os.path.join(dataset_path, 'movies_metadata.csv'), low_memory=False).drop_duplicates(subset="id")
    credits = pd.read_csv(os.path.join(dataset_path, 'credits.csv')).drop_duplicates(subset="id")
    keywords = pd.read_csv(os.path.join(dataset_path, 'keywords.csv')).drop_duplicates(subset="id")

    """
    # print(metadata.shape)
    # print(credits.shape)
    # print(keywords.shape)

    # print(metadata.dtypes)
    # print(credits.dtypes)
    # print(keywords.dtypes)

    # To merge the info from these three files we need to use the id
    # The id in the metadata is an object while in the others it is an int
    # So the it needs to be converted into an int in order for the join to work
    """

    # Converts ids in metadata to int, errors='coerce' tranforms non-numbers to NaN in order to avoid errors
    metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce')
    metadata = metadata.dropna(subset=['id'])
    metadata['id'] = metadata['id'].astype(int)

    df = metadata.merge(credits, on='id').merge(keywords, on='id')
    print(f"Original Shape: {df.shape}")

    def get_director(crew_str):
        try:
            crew = ast.literal_eval(crew_str)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name']
            return None
        except (ValueError, SyntaxError):
            return None

    def get_cast(cast_str, n=5):
        '''
            This method saves the first 5 names on the cast of a movie
            This is to make sure the movies all have a similar number of
            actors, and we keep the first 5 because the actors are in 
            order of importance in the movie.
        '''
        try:
            cast = ast.literal_eval(cast_str)
            return [member['name'] for member in cast[:n]]
        except (ValueError, SyntaxError):
            return []
        
    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(get_cast)
    df = df[df['cast'].apply(lambda x: len(x) > 0)]

    features = ['budget', 'genres', 'keywords', 'original_language', 
                'production_companies', 'production_countries', 'release_date', 
                'runtime', 'spoken_languages', 'cast', 'director', 'revenue']

    # print(df[features].isnull().sum())
    df = df.dropna(subset=features)

    """
        This shows that the set is actually very incomplete,
        ~35000 movies have no value for revenue and ~37000
        don't have value for budget.
        Movies without a revenue have to be removed since it
        would be impossible to train a model without the target value.
        In the case of the budgetless movies, two approaches will be taken
        one is to train the model without these movies, the other is to
        remove the budget from the model training features.
        The one with the best results will be chosen.

        # print(df[features].isnull().sum())

        # print(f"Budget = 0: {(df['budget'] == '0').sum()}")
        # print(f"Revenue = 0: {(df['revenue'] == 0).sum()}")
    """

    # Remove revenue = 0
    # df = df[df['revenue'] != 0]
    # print(f"After removing revenue=0: {df.shape}")

    # Convert budget from string to numeric
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

    df = df.dropna(subset=['budget', 'revenue'])
    df = df[df['revenue'] > 0].copy()

    # Check overlapping
    # print(f"Budget = 0: {(df['budget'] == 0).sum()}")
    # print(f"Shape final: {df.shape}")

    # Version 1 - no movies with budget=0
    df_full = df[df['budget'] != 0].copy()
    # print(f"df_full: {df_full.shape}")

    # Version 2 - imputing budget from the dataset - This version brought worse results
    # df_imputed = df.copy()
    # median_budget = df_imputed[df_imputed['budget'] != 0]['budget'].median()
    # df_imputed.loc[df_imputed['budget'] == 0, 'budget'] = median_budget
    # df_full = df_imputed.copy() 
    # print(f"df_imputed: {df_imputed.shape}")
    # print(f"Median budget: ${median_budget:,.0f}")

    # The revenue and budget values used will go through a logarithmic transformation
    # This is because the difference in values needs to be looked at with perspective
    # The difference between a movie with 1M and 10M budget is completely different
    # from a movie with 501M and another with 510M budgets, although the absolute 
    # value difference is the same - 9M.
    # We use log1p (log(1+x)) to avoid log(0)

    df_full['revenue_log'] = np.log1p(df_full['revenue'])
    df_full['budget_log'] = np.log1p(df_full['budget'])

    def preprocessing_vis():

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].hist(df_full['revenue_log'], bins=50)
        axes[0].set_xlabel('log(Revenue)')

        axes[1].hist(df_full['budget_log'], bins=50)
        axes[1].set_xlabel('log(Budget)')

        axes[2].scatter(df_full['budget_log'], df_full['revenue_log'], alpha=0.3)
        axes[2].set_xlabel('log(Budget)')
        axes[2].set_ylabel('log(Revenue)')
        fig.suptitle('Budget vs Revenue')
        plt.savefig('budget_vs_revenue.png')
        plt.show()

    # Uncomment to visualise the preprocessing data relation between budget and revenue
    # preprocessing_vis()

    # Extract the release_date as an actual date
    df_full['release_date'] = pd.to_datetime(df_full['release_date'], errors='coerce')
    df_full = df_full.dropna(subset=['release_date'])
    df_full['release_month'] = df_full['release_date'].dt.month
    df_full['release_year'] = df_full['release_date'].dt.year

    df_full['genres_list'] = df_full['genres'].apply(
        lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notna(x) else []
    )

    # Check most frequent genres
    all_genres = [genre for sublist in df_full['genres_list'] for genre in sublist]

    df_full['director_list'] = df_full['director'].apply(
        lambda x: [x] if pd.notna(x) else []
    )

    df_full['cast_list'] = df_full['cast'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # print(Counter(all_genres).most_common(10))
    # print(f"Total unique genres: {len(set(all_genres))}") # There are 20 unique genres

    # all_directors = df_full['director'].unique()
    # all_actors = [a for sublist in df_full['cast'] for a in sublist]
    # print(f"Unique directors: {len(all_directors)}") # 2404 unique directors
    # print(f"Unique Actors: {len(set(all_actors))}") # 10327 unique directors

    mlb_genres = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb_genres.fit_transform(df_full['genres_list']),
        columns=[f'genre_{g}' for g in mlb_genres.classes_],
        index=df_full.index
    )

    df_full = pd.concat([df_full, genres_encoded], axis=1)
    # print(f"Added columns: {genres_encoded.shape[1]}")

    for col, key in [('keywords', 'name'), ('production_companies', 'name'), 
                    ('production_countries', 'iso_3166_1'), ('spoken_languages', 'iso_639_1')]:
        
        df_full[f'{col}_list'] = df_full[col].apply(
            lambda x: [g[key] for g in ast.literal_eval(x)] if pd.notna(x) else []
        )
        all_vals = [v for sublist in df_full[f'{col}_list'] for v in sublist]
        unique_vals = len(set(all_vals))
        # print(f"{col}: {unique_vals} unique values, top 5: {Counter(all_vals).most_common(5)}")

    def get_top_n(series_of_lists, n):
        ''' Saves only N most frequent values '''
        all_vals = [v for sublist in series_of_lists for v in sublist]
        top_n = {val for val, _ in Counter(all_vals).most_common(n)}
        return series_of_lists.apply(lambda lst: [v for v in lst if v in top_n])

    configs = [
        # ('genres_list', None), # All genres
        ('director_list', 50), # Top 50 directors
        ('cast_list', 150), # Top 150 actors
        ('production_countries_list', None), # All production countries
        ('spoken_languages_list', None), # All spoken languages
        ('production_companies_list', 50), # Top 50 production companies
        ('keywords_list', 100) # Top 100 keywords
    ]

    for col, top_n in configs:
        filtered = get_top_n(df_full[col], top_n) if top_n else df_full[col]
        mlb = MultiLabelBinarizer()
        encoded = pd.DataFrame(
            mlb.fit_transform(filtered),
            columns=[f'{col.replace("_list", "")}_{v}' for v in mlb.classes_],
            index=df_full.index
        )
        df_full = pd.concat([df_full, encoded], axis=1)
        # print(f"{col}: {encoded.shape[1]} added columns")

    lang_dummies = pd.get_dummies(df_full['original_language'], prefix='lang')
    df_full = pd.concat([df_full, lang_dummies], axis=1)
    # print(f"original_language: {lang_dummies.shape[1]} added columns")

    # Columns to keep
    cols_to_keep = (
        ['budget_log', 'runtime', 'release_month', 'release_year','revenue_log'] +
        [c for c in df_full.columns if c.startswith('genre_')] +
        [c for c in df_full.columns if c.startswith('director_') and not c.endswith('_list')] + 
        [c for c in df_full.columns if c.startswith('cast_') and not c.endswith('_list')] +  
        [c for c in df_full.columns if c.startswith('production_countries_') and not c.endswith('_list')] +
        [c for c in df_full.columns if c.startswith('spoken_languages_') and not c.endswith('_list')] +
        [c for c in df_full.columns if c.startswith('production_companies_') and not c.endswith('_list')] +
        [c for c in df_full.columns if c.startswith('keywords_') and not c.endswith('_list')] +
        [c for c in df_full.columns if c.startswith('lang_')]
    )


    df_model = df_full[cols_to_keep].copy()
    # print(f"Final shape for training: {df_model.shape}")
    # print(f"Nulls: {df_model.isnull().sum().sum()}")

    # Remove nulls
    df_model = df_model.dropna()
    print(f"Final Shape: {df_model.shape}")
    print(f"Preprocessing duration: {time()-preproc_time:.3f}s")
    print("\nFinished preprocessing.\nInitiating training...\n")

    # =========================
    # This marks the end of the preprocessing of the dataset
    # ========================

    # ==========
    # Data split
    # ==========

    X = df_model.drop('revenue_log', axis=1)
    y = df_model['revenue_log']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    X_train = X_train.copy()
    X_test = X_test.copy()

    # print(X_train.dtypes[X_train.dtypes == 'object']) 

    list_cols = [c for c in X_train.columns if c.endswith('_list')]
    X_train = X_train.drop(columns=list_cols)
    X_test = X_test.drop(columns=list_cols)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)


    # Convert back from log to real values
    # We use exmp1 because it's the inverse of log1p
    y_test_real = np.expm1(y_test)

    ### MODEL TRAINING

    # =================
    # Linear Regression
    # =================

    def lr():
        lr_timestamp = time()

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)

        # Convert back from log to real values
        y_pred_lr_real = np.expm1(y_pred_lr)

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_lr_real))
        r2 = r2_score(y_test_real, y_pred_lr_real)

        print(f"Linear Regression - RMSE: ${rmse:,.0f}, R²: {r2:.4f}")
        print(f"Linear Regression model duration: {time()-lr_timestamp:.3f}s\n")
        return r2, rmse, y_pred_lr

    r2_lr, rmse_lr, y_pred_lr = lr()

    # =============
    # Random Forest
    # =============

    def rf(n_estimators, max_depth, min_samples):
        rf_timestamp = time()

        rf_model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, max_depth=max_depth, min_samples_split=min_samples, random_state=SEED)
        rf_model.fit(X_train, y_train)

        # Evaluate training
        y_pred_train = rf_model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)

        y_pred_rf = rf_model.predict(X_test)
        
        y_pred_rf_real = np.expm1(y_pred_rf) # Convert back from log to real values
        rmse_rf = np.sqrt(mean_squared_error(y_test_real, y_pred_rf_real))
        r2_rf = r2_score(y_test_real, y_pred_rf_real)

        print(f"Random Forest - Train R²: {r2_train:.4f} | Test R²: {r2_rf:.4f} | RMSE: ${rmse_rf:,.0f}")
        print(f"Random Forest model duration: {time()-rf_timestamp:.3f}s\n")
        return r2_rf, y_pred_rf

    def get_best_rf():
        n_est = [225, 250, 275, 300, 325, 350]
        max_depth = [15, 20, 25, 30, 35, None]
        min_samples = [2, 3, 4, 5, 6, 7]
        cfgs = []
        from itertools import product
        for n, depth, samples in product(n_est, max_depth, min_samples):
            cfgs.append({
                "N_EST": n,
                "MAX_DEPTH": depth,
                "MIN_SAMPLES": samples
            })
        result = []
        max_r2 = 0
        for cfg in cfgs:
            print(cfg)
            r2, _ = rf(n_estimators=cfg["N_EST"], max_depth=cfg["MAX_DEPTH"], min_samples=cfg["MIN_SAMPLES"])
            if r2 > max_r2:
                max_r2 = r2
                result = (cfg , max_r2)
        return result
        
    # print(f"Best hyperparameters for Random Forest \n{get_best_rf()}\n") # Uncomment to find best hyperparameters for Random Forest

    r2_rf, y_pred_rf = rf(n_estimators=225, max_depth=30, min_samples=2)

    # =====================
    # Multilayer Perceptron
    # =====================

    # MLP requires normalization
    # The random forest takes decisions based on thresholds, the scale doesn't matter, just the relative order
    # The MLP on the ther hand calculates weighted sums, scaling values makes the model learn in a 
    # a more even manner, not giving huge advantages to bigger features

    class MLP(nn.Module):
        def __init__(self, input_size, device):
            super().__init__()
            self.to(device)
            self.device = device
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
    
        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
        def train_loop(self, dataloader, loss_fn, optimizer):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                # Compute prediction and loss
                y = y.unsqueeze(1)
                pred = self(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()

        def test_loop(self, dataloader, loss_fn):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            num_batches = len(dataloader)
            test_loss = 0

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)

                    pred = self(X)
                    test_loss += loss_fn(pred, y.unsqueeze(1)).item()

            test_loss /= num_batches
            print(f"Test loss: {test_loss:.4f}")

    def mlp(epochs, batch_size):
        mlp_timestamp = time()

        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {device} device")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        scaler_X = StandardScaler()
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

        X_train_tensor = torch.tensor(X_train_sc, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_sc, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_sc, dtype=torch.float32)

        # Create dataset and loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        y_test_tensor = torch.tensor(
            scaler_y.transform(y_test.values.reshape(-1,1)).ravel(),
            dtype=torch.float32
        )

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = MLP(X_train.shape[1], device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            if (epoch+1) % 250 == 0:
                print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
            model.train_loop(train_dataloader, loss_fn, optimizer)
        model.test_loop(test_dataloader, loss_fn)
        print("Done!")

        with torch.no_grad():
            y_pred_mlp = model(X_test_tensor.to(device)).squeeze()

        y_pred_mlp = y_pred_mlp.cpu().numpy().ravel()

        y_pred_mlp_log = scaler_y.inverse_transform(
            y_pred_mlp.reshape(-1,1)
        ).ravel()
        y_pred_mlp_real = np.expm1(y_pred_mlp_log)

        with torch.no_grad():
            y_pred_train_mlp = model(X_train_tensor.to(device)).squeeze()

        y_pred_train_mlp = y_pred_train_mlp.cpu().numpy().ravel()
        y_pred_train_log = scaler_y.inverse_transform(
            y_pred_train_mlp.reshape(-1, 1)
        ).ravel()

        r2_train = r2_score(y_train, y_pred_train_log)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_mlp_real))
        r2 = r2_score(y_test_real, y_pred_mlp_real)

        print(f"MLP - Train R²: {r2_train:.4f} | Test R²: {r2:.4f} | RMSE: ${rmse:,.0f}")
        print(f"MLP model duration: {time()-mlp_timestamp:.3f}s\n")
        return r2, rmse, y_pred_mlp_log

    r2_mlp, rmse_mlp, y_pred_log_mlp = mlp(epochs=1000, batch_size=128)

    # =================
    # Boosting Gradient - XGBoost
    # =================

    def gb(n_estimators=2000, max_depth=6, learning_rate=0.1, subsample=0.85, colsample_bytree=0.8):
        gb_timestamp = time()
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=SEED
        )
        gb_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=SEED,
            n_jobs=1,
            tree_method='exact',
            eval_metric='rmse',
            early_stopping_rounds=50,
        )
        gb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        # Evaluate training
        y_pred_train = gb_model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)

        y_pred_gb = gb_model.predict(X_test)
        y_pred_gb_real = np.expm1(y_pred_gb)
        rmse_gb = np.sqrt(mean_squared_error(y_test_real, y_pred_gb_real))
        r2_gb = r2_score(y_test_real, y_pred_gb_real)

        print(f"XGBoost - Train R²: {r2_train:.4f} | Test R²: {r2_gb:.4f} | RMSE: ${rmse_gb:,.0f}")
        print(f"XGBoost model duration: {time()-gb_timestamp:.3f}s\n")
        return r2_gb, y_pred_gb, r2_train

    r2_gb, y_pred_gb, _ = gb(n_estimators=2000, max_depth=9, learning_rate=0.12, subsample=0.9, colsample_bytree=0.8)

    def get_best_gb():
        from itertools import product
        
        # Change if in need of testing
        max_depths = [5, 6, 7, 8, 9, 10, 11, 12]
        learning_rates = [0.08, 0.1, 0.12]
        subsamples = [0.8, 0.85, 0.9]
        
        max_r2 = 0
        best = None
        
        for depth, lr, sub in product(max_depths, learning_rates, subsamples):
            cfg = {"max_depth": depth, "lr": lr, "subsample": sub}
            print(cfg)
            r2, _, r2_train = gb(n_estimators=2000, max_depth=depth, learning_rate=lr, 
                    subsample=sub, colsample_bytree=0.8)
            if r2 > max_r2 and r2_train < 0.85:
                max_r2 = r2
                best = (cfg, max_r2)
        
        return best

    # print(f"Best hyperparameters for XGBoost\n{get_best_gb()}\n") # Uncomment to find best parameters

    # =================
    def evaluate(name, y_true_log, y_pred_log):
        """Compute metrics in both log and original (USD) scales."""
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(np.clip(y_pred_log, a_min=None, a_max=30))
        pct_errors = np.abs((y_true - y_pred) / y_true) * 100
        return {
            "model": name,
            "rmse_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),4),
            "mae_log": round(float(mean_absolute_error(y_true_log, y_pred_log)),4),
            "r2_log": round(float(r2_score(y_true_log, y_pred_log)),4),
            "rmse_usd": format(float(np.sqrt(mean_squared_error(y_true, y_pred))), '.3g'),
            "mae_usd": format(float(mean_absolute_error(y_true, y_pred)), '.3g'),
            "mdape": round(float(np.median(pct_errors)), 2)

        }

    print(evaluate('Linear Regression', y_test, y_pred_lr))
    print(evaluate('Random Forest', y_test, y_pred_rf))
    print(evaluate('MLP', y_test, y_pred_log_mlp))
    print(evaluate('Gradient Boosting', y_test, y_pred_gb))
