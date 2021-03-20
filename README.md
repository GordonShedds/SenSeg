# SenSeg
Semantic segmentation of Sentinel-2 images with classic ML

Simple interface to quickly train baseline models on Sentinel-2 data.
It supports multichannel data in .tif and .jp2 format. Even if your bands are saved in different files - the script can merge them automatically.

Class labels mask can be both in vector format (.shp, .geojson) or raster (.tif). Vector data will be rasterized accordingly with crs of initial image.

By default, Random forest algorithm with batch learning is used.

**Code example:**
```python
from utils.featurizer import SentinelFeaturizer
from utils.trainer import train_model
from utils.predictor import predict

# Read initial image and mask
featurizer = SentinelFeaturizer(image='image.tif',
                                mask='mask.geojson')
                                
# Get feature vectors and labels with sliding window
x, y = featurizer.get_features(window_size=3)

# Train model
trained_model = train_model(x, y)

# Predict on new image
predict(image='new_image_for_prediction.tif', model=model, window_size=3)

```

