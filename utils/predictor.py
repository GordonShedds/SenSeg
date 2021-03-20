from pathlib import Path

import numpy as np
from typing import Union, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from .featurizer import SentinelFeaturizer
from scipy.ndimage.morphology import binary_opening
from scipy.ndimage.morphology import binary_closing
import rasterio as rio


def predict(image: Union[np.ndarray, str],
            model: Union[Path, str, RandomForestClassifier],
            chosen_channels: Optional[List[str]] = None,
            window_size: int = 1,
            norm_dict: Optional[Union[Path, str, Dict[str, List[float]]]] = None,
            opening_size: int = 0,
            save_path: Optional[Union[str, Path]] = None
            ) -> np.ndarray:

    featurizer = SentinelFeaturizer(image=image,
                                    chosen_channels=chosen_channels
                                    )
    if norm_dict is not None:
        featurizer.normalize(norm_dict=norm_dict)

    X = featurizer.get_features(window_size=window_size)
    y_pred = model.predict(X)

    height, width, transform, crs = featurizer.height, featurizer.width, featurizer.transform, featurizer.crs
    pad = (window_size // 2)
    y_pred = y_pred.reshape(height - 2 * pad, width - 2 * pad)

    if opening_size > 0:
        y_pred = binary_opening(y_pred, np.ones((opening_size, opening_size))).astype(np.uint8)
        y_pred = binary_closing(y_pred).astype(np.uint8)

    if save_path:
        if save_path.endswith('.tif'):
            with rio.open(
                    save_path,
                    'w',
                    driver='GTiff',
                    width=y_pred.shape[1],
                    height=y_pred.shape[0],
                    count=1,
                    dtype=np.uint8,
                    transform=transform,
                    crs=crs,
            ) as output:
                output.write(y_pred[np.newaxis, ...].astype(np.uint8))
        else:
            print('Please, use .tif extension for saved file!')

    return y_pred
