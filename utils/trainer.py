from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
from joblib import dump
from sklearn.metrics import classification_report


def train_model(X: np.array,
                y: np.array,
                show_results: bool = True,
                test_size: float = 0.25,
                n_jobs: int = 4,
                epochs: int = 1,
                max_depth: int = 15,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                max_features: Union[str, int, float] = 'auto',
                model_save_path: Optional[Union[str, Path]] = None,
                max_elements_in_batch: int = 500000,
                n_estimators: int = 100
                ) -> RandomForestClassifier:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    N_splits = max(1, len(X_train) // max_elements_in_batch) + 1
    batch_size = len(X_train) // N_splits

    if len(X_train) >= max_elements_in_batch:
        model = RandomForestClassifier(warm_start=True,
                                       n_estimators=1,
                                       n_jobs=n_jobs,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       )
        for _ in tqdm(range(epochs)):
            for i in tqdm(range(N_splits - 1)):
                model.fit(X_train[i * batch_size: (i + 1) * batch_size, :], y_train[i * batch_size: (i + 1) * batch_size])
                model.n_estimators += 1
            shuffler = np.random.permutation(len(X_train))
            X_train = X_train[shuffler]
            y_train = y_train[shuffler]
    else:
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       n_jobs=n_jobs,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=max_features,
                                       )
        model.fit(X_train, y_train)

    if show_results:
        y_pred_test = model.predict(X_test)
        print('Results on test data:')
        print(classification_report(y_test, y_pred_test))

    if model_save_path:
        if model_save_path.endswith('.joblib'):
            dump(model, model_save_path)
        else:
            print('Model file extension must be .joblib!')

    return model