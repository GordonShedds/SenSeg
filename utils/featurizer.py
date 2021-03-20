import errno
import json
import os
import warnings
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from tqdm import tqdm

warnings.filterwarnings("ignore")

from .sentinel_info import CHANNELS_INFO
from .utils import get_window_features, get_window_labels, rasterize_mask, get_mask_roi


class SentinelFeaturizer():

    def __init__(self,
                 image: Union[np.ndarray, str],
                 mask: Optional[Union[gpd.GeoDataFrame, str]] = None,
                 chosen_channels: Optional[List[str]] = None,
                 mask_label_column: Optional[str] = None,
                 crop_mask: bool = False):

        if chosen_channels is not None:
            assert all([c in CHANNELS_INFO for c in chosen_channels]),\
                f'Please, choose channels from list {list(CHANNELS_INFO.keys())}'

        self.chosen_channels = chosen_channels
        self.mask_label_column = mask_label_column

        self.prepare_image(image)
        self.norm_dict = None

        if mask is not None:
            self.prepare_mask(mask)

            if crop_mask:
                ymin, ymax, xmin, xmax = get_mask_roi(self.mask)
                self.mask = self.mask[ymin: ymax, xmin: xmax]
                self.image = self.image[ymin: ymax, xmin: xmax, :]
            else:
                self.mask = self.mask

            if np.all(self.mask == 0):
                print('WARNING: your raster mask is full of zeros!')
        else:
            self.mask = None

    def prepare_image(self,
                      image: Union[np.ndarray, str]):

        if isinstance(image, str):
            if os.path.isfile(image):
                with rio.open(image) as src:
                    self.width = src.meta['width']
                    self.height = src.meta['height']
                    self.crs = src.meta['crs']
                    self.transform = src.meta['transform']
                    image_raster = src.read()
                    image_raster = np.squeeze(image_raster)
                    if image_raster.shape[0] <= len(CHANNELS_INFO):
                        image_raster = image_raster.transpose((1, 2, 0))
                    self.image = image_raster
            elif os.path.isdir(image):
                image_path = image + '/' if not image.endswith('/') else image
                img_channels_paths = list(Path(image_path).rglob('*B*.jp2')) + list(Path(image_path).rglob('*B*.tif'))
                assert img_channels_paths, 'No .jp2 or .tif images in directory! ' \
                                           'Please check that image files have a channel listed in their name!' \
                                           '\nExample: B01.tif'
                if self.chosen_channels is None:
                    self.chosen_channels = [c for c in CHANNELS_INFO.keys() if c in img_channels_paths]
                    assert self.chosen_channels, 'There are no valid channels in directory!'

                channels_resolutions = [CHANNELS_INFO[c]['resolution'] for c in self.chosen_channels]
                channels_resolutions = np.array(channels_resolutions)

                # Choosing channel with best resolution for meta
                highest_res_channel = self.chosen_channels[np.argmin(channels_resolutions)]
                for c in img_channels_paths:
                    if highest_res_channel in c:
                        with rio.open(c, "r") as src:
                            self.width = src.meta['width']
                            self.height = src.meta['height']
                            self.crs = src.meta['crs']
                            self.transform = src.meta['transform']
                        break
                mc_image = np.zeros((self.height, self.width, len(self.chosen_channels)))
                for c_ind, chosen_band in enumerate(self.chosen_channels):
                    for c in img_channels_paths:
                        if chosen_band in c:
                            with rio.open(c) as src:
                                img = src.read()
                                img = np.squeeze(img)
                                if src.meta['height'] != self.height:
                                    img = cv2.resize(img, dsize=(self.width, self.height))
                                mc_image[..., c_ind] = img
                                break
                self.image = mc_image

            else:
                raise Exception('Invalid path! Please, use a path to image file or to a directory with channels')

    def prepare_mask(self,
                     mask):

        if isinstance(mask, str):
            if mask.endswith('.shp') or mask.endswith('.geojson'):
                mask_df = gpd.read_file(mask)
                mask_raster = rasterize_mask(mask=mask_df,
                                             crs=self.crs,
                                             height=self.height,
                                             width=self.width,
                                             transform=self.transform,
                                             label_column_name=self.mask_label_column)
            elif mask.endswith('.tif'):
                with rio.open(mask) as src:
                    assert src.meta['crs'] == self.crs, 'Please use the same crs for image and raster mask!'
                    mask_raster = src.read()
                    mask_raster = np.squeeze(mask_raster)

                    if mask_raster.shape[0] <= len(CHANNELS_INFO):
                        mask_raster = mask_raster.transpose((1, 2, 0))

            else:
                raise Exception('Unknown file type! Please, use one the following: .shp, .geojson, .tif')

        elif isinstance(mask, gpd.GeoDataFrame):
            mask_raster = rasterize_mask(mask=mask,
                                         crs=self.crs,
                                         height=self.height,
                                         width=self.width,
                                         transform=self.transform,
                                         label_column_name=self.mask_label_column)
        else:
            raise Exception('Invalid mask type!')

        if (mask_raster.shape[0] != self.height) or (mask_raster.shape[1] != self.width):
            print(f'WARNING: image shape ({self.height}, {self.width}) '
                  f'doesnt match with raster shape of ({mask_raster.shape[0]}, {mask_raster.shape[1]})!\n'
                  f'The resizing is performed!')
            mask_raster = cv2.resize(mask_raster, dsize=(self.width, self.height))

        no_data_mask = np.any(self.image != 0, axis=-1)  # Mask of "black zones" on image
        mask_raster = mask_raster * no_data_mask

        self.mask = mask_raster

    def plot(self,
             channels_inds_to_draw: Union[int, List[int]] = None,
             tile_size: int = 512,
             n_tiles: int = 4,
             mask_alpha: float = 0.5,
             show_axis: bool = False,
             max_iter: int = 2000):

        if channels_inds_to_draw is not None:
            if isinstance(channels_inds_to_draw, list):
                assert (len(channels_inds_to_draw) == 1) or (len(channels_inds_to_draw) == 3),\
                    'Please use 1 or 3 channels inds to draw!'
                assert all([c < self.image.shape[2] for c in channels_inds_to_draw]),\
                    f'There are only {self.image.shape[2]} in your image!'
            else:
                assert channels_inds_to_draw < self.image.shape[2],\
                    f'There are only {self.image.shape[2]} in your image!'
            img_to_draw = self.image[..., channels_inds_to_draw]
        else:
            if (self.chosen_channels is not None) and ({'B04', 'B03', 'B02'} <= set(self.chosen_channels)):
                red_index = self.chosen_channels.index('B04')
                green_index = self.chosen_channels.index('B03')
                blue_index = self.chosen_channels.index('B02')
                img_to_draw = self.image[..., [red_index, green_index, blue_index]]
            else:
                img_to_draw = self.image[..., :min(3, self.image.shape[2])]

        fig, axs = plt.subplots(nrows=n_tiles // 2, ncols=2, figsize=(10, 10))
        for i in range(n_tiles):
            tile_mask = np.zeros((tile_size, tile_size))
            n_iters = 0
            while (not np.any(tile_mask)) and n_iters < max_iter:
                # Get random tile with non-zero pixels
                x_min = np.random.randint(0, self.width - tile_size)
                y_min = np.random.randint(0, self.height - tile_size)
                x_max = x_min + tile_size
                y_max = y_min + tile_size
                tile_mask = self.mask[y_min:y_max, x_min:x_max] if self.mask is not None else None
                n_iters += 1
            if (n_iters == max_iter) and (not (self.mask is None)):
                print('Failed to find non-zero tile!')
                return

            if not show_axis:
                axs[i // 2, i % 2].axis('off')

            axs[i // 2, i % 2].imshow(img_to_draw[y_min:y_max, x_min:x_max, :])
            if tile_mask is not None:
                axs[i // 2, i % 2].imshow(tile_mask, alpha=mask_alpha)

        plt.show()

    def normalize(self,
                  norm_dict: Optional[Dict[str, List[float]]] = None,
                  save_path: Optional[Union[str, Path]] = None) -> Dict[str, List[float]]:
        """
        Normalize image data and return dict with stds and means per each channel
        :param norm_dict: if mentioned - normalize will use values from this dict for normalization
        :param save_path: JSON. If mentioned - normalize will save the dict with means and stds as json file
        :return: dict with stds and means per each channel
        """

        if norm_dict:
            assert ('std' in norm_dict) and ('mean' in norm_dict), 'norm_dict must contain std and mean keys!'
            assert len(norm_dict['std']) == self.image.shape[-1],\
                f'norm_dict must contain {self.image.shape[-1]} std values!'
            assert len(norm_dict['mean']) == self.image.shape[-1], \
                f'norm_dict must contain {self.image.shape[-1]} mean values!'

            means_per_channel = np.array(norm_dict['mean'])
            stds_per_channel = np.array(norm_dict['std'])
        else:
            means_per_channel = np.mean(self.image, axis=(0, 1))
            stds_per_channel = np.std(self.image, axis=(0, 1))

        normalized_img = (self.image - means_per_channel) / stds_per_channel
        normalization_dict = {'mean': means_per_channel.tolist(),
                              'std': stds_per_channel.tolist()}

        if save_path:
            assert save_path.endswith('.json'), 'Please, use the .json extension for save_path!'
            with open(save_path, 'w') as fp:
                json.dump(normalization_dict, fp)

        self.image = normalized_img
        self.norm_dict = normalization_dict

        return normalization_dict

    def get_features(self,
                     window_size: int = 1,
                     save_path_dir: Optional[str] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """

        :param window_size: window size to collect features around each pixel
        :param save_path_dir: if mentioned - directory to save image features and labels as .npy
        :return: array with image features and array with corresponding labels
        """

        X = get_window_features(self.image, window_size)

        if self.mask is not None:
            y = get_window_labels(self.mask, window_size)
            assert X.shape[0] == len(y), 'Oops, number of X point doesnt match with the number of labels! ' \
                                         'Please, validate your data.'

        if save_path_dir is not None:
            save_path = save_path_dir + '/' if not save_path_dir.endswith('/') else save_path_dir
            x_path = save_path + 'X.npy'
            y_path = save_path + 'y.npy'

            if not os.path.exists(os.path.dirname(x_path)):
                try:
                    os.makedirs(os.path.dirname(x_path))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            np.save(x_path, X)
            if self.mask is not None:
                np.save(y_path, y)

        if self.mask is not None:
            return X, y
        else:
            return X


def featurize(images: List[Union[np.ndarray, str]],
              masks: Optional[List[Union[gpd.GeoDataFrame, str]]] = None,
              window_size: int = 1,
              normalize_data: bool = False,
              normalization_dict: Optional[Union[Path, str, Dict[str, List[float]]]] = None,
              norm_dict_save_path: Optional[Union[str, Path]] = None,
              chosen_channels: Optional[List[str]] = None,
              mask_label_column: Optional[str] = None,
              crop_mask: bool = False
              ) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    norm_dicts = []

    if masks is not None:
        assert len(images) == len(masks), 'Please, pass the same amount of images and masks!'

        for img, msk in tqdm(zip(images, masks)):
            feat = SentinelFeaturizer(img, msk, chosen_channels, mask_label_column, crop_mask)
            if normalize_data:
                norm_dict = feat.normalize()
                norm_dicts.append(norm_dict)
            X_i, y_i = feat.get_features(window_size)
            X.append(X_i)
            y.append(y_i)

        X = np.vstack(X)
        y = np.hstack(y)
    else:
        for img in tqdm(images):
            feat = SentinelFeaturizer(image=img,
                                      chosen_channels=chosen_channels,
                                      mask_label_column=mask_label_column,
                                      crop_mask=crop_mask)
            if normalize_data:
                norm_dict = feat.normalize(norm_dict=normalization_dict)
                norm_dicts.append(norm_dict)
            X_i = feat.get_features(window_size)
            X.append(X_i)

        X = np.vstack(X)

    if norm_dict_save_path is not None:
        mean_values = (np.sum(np.array([np.array(d['mean']) for d in norm_dicts]), axis=0) / len(images)).tolist()
        std_values = (np.sum(np.array([np.array(d['std']) for d in norm_dicts]), axis=0) / len(images)).tolist()

        norm_dict = {'mean': mean_values,
                     'std': std_values}

        if not norm_dict_save_path.endswith('.json'):
            print('Please, use the .json extension for norm_dict_save_path!')
        else:
            with open(norm_dict_save_path, 'w') as fp:
                json.dump(norm_dict, fp)

    if masks is not None:
        return X, y
    else:
        return X
