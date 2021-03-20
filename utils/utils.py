from typing import Tuple, Optional

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon
from shapely.ops import unary_union


def get_window_features(image_np: np.ndarray,
                        window_size: int = 1) -> np.ndarray:
    """
    :param image_np: исходная картинка в виде np.array,  shape = (h, w, d)
    :param window_size: размер окна (его высота и ширина), в пределах которого будут собираться признаки
    :return: список признаков всех пикселей (для которых корректно определено окно)
    количество признаков = window_size ** 2 * d
    """

    assert len(image_np.shape) == 3, 'Изображение должно иметь форму (height, width, depth)'

    n = window_size
    h, w, d = image_np.shape

    assert h > n, 'Высота изображения должна превосходить размер окна!'
    assert w > n, 'Ширина изображения должна превосходить размер окна!'

    if n == 1:
        return image_np.reshape(h * w, d)
    else:
        s = image_np.strides
        tmp = np.lib.stride_tricks.as_strided(image_np, strides=s[:2] + s, shape=(h - n + 1, w - n + 1, n, n, d))
        X = tmp.reshape(-1, n ** 2 * d)
        return X


def get_window_labels(mask: np.ndarray,
                      window_size: int = 1) -> np.ndarray:
    """
    :param mask: маска для изображения, форма (height, width)
    :param window_size: размер окна (его высота и ширина), в пределах которого собирались признаки
    :return: вектор меток с формой (n,)
    """

    if window_size == 1:
        return mask.ravel()
    n = window_size // 2
    labels = np.ravel(mask[n:-n, n:-n])
    return labels


def poly_from_utm(polygon: Polygon,
                  transform: Affine) -> Polygon:
    """
    :param polygon: исходный полигон
    :param transform: трансформация (берется из исходного снимка) для полигона
    :return: трансформированный полигон
    """
    poly_pts = []
    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))
    new_poly = Polygon(poly_pts)
    return new_poly


def rasterize_mask(mask: gpd.GeoDataFrame,
                   crs: str,
                   height: int,
                   width: int,
                   transform: Affine,
                   label_column_name: Optional[str] = None,
                   one_hot: bool = False) -> np.ndarray:
    """
    Растеризация маски
    :param mask: исходная маска в формате GeoDataFrame
    :param crs: целевая система координат (как у снимка)
    :param height: итоговая высота растра
    :param width: итоговая ширина растра
    :param transform: афинное преобразование для полигонов маски
    :param label_column_name: название столбца с меткой класса
    :param one_hot: возвращать ли лейблы в формате one-hot encoding
    :return: np.ndarray с маской, форма = (height, width)
    """
    mask = mask.to_crs(crs)
    poly_shp = []
    if label_column_name is None:
        for num, row in mask.iterrows():
            if row['geometry'].geom_type == 'Polygon' or (row['geometry'].geom_type == 'MultiPolygon'):
                poly = poly_from_utm(row['geometry'], transform)
                poly_shp.append(poly)

        return rasterize(shapes=poly_shp,
                         out_shape=(height, width))
    else:
        labels = mask[label_column_name].unique()
        rasters = []
        for label in labels:
            poly_shp = []
            for _, row in mask[mask[label_column_name] == label].iterrows():
                if row['geometry'].geom_type == 'Polygon' or (row['geometry'].geom_type == 'MultiPolygon'):
                    poly = poly_from_utm(row['geometry'], transform)
                    poly_shp.append(poly)
            rasters.append(rasterize(shapes=poly_shp, out_shape=(height, width)))

        result = np.stack(rasters, axis=-1)
        zero_class_mask = np.any(result != 0, axis=-1)

        if one_hot:
            # Возвращаем растр с формой (height, width, n_classes + 1), с единицей на соответствующем слое
            return result
        else:
            # Возвращаем растр с формой (height, width) - каждый класс представлен своим индексом
            return (np.argmax(result, axis=-1) + 1) * zero_class_mask


def get_mask_roi(mask: np.ndarray) -> Tuple[int, int, int, int]:

    """
    По бинарной маске находит координаты наименьшего прямоугольника, содержащего размеченные пиксели
    :param mask: растеризованная маска
    :return: координаты минимального ограничивающго прямоугольника, содержащего размеченные пиксели
    """
    positions = np.nonzero(mask)

    top = positions[0].min()
    bottom = positions[0].max()
    left = positions[1].min()
    right = positions[1].max()

    return top, bottom, left, right
