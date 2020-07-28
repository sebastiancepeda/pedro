import cv2
import numpy as np
import pandas as pd

from cv.image_processing import get_xs
from io_utils.read_polygons_json import get_labels_plates_text


def load_image(im_data, folder, dsize, in_channels, alphabet):
    dsize_cv2 = (dsize[1], dsize[0])
    image = im_data.image.values[0]
    # Image load
    im_file = f"{folder}/{image}"
    im = cv2.imread(im_file)
    assert im is not None, f"Error while reading image: {im_file}"
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if in_channels == 3:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = cv2.resize(im, dsize=dsize_cv2, interpolation=cv2.INTER_CUBIC)
    # Setting labels
    gt = np.zeros((dsize[0], dsize[1], len(alphabet)))
    gt = gt.astype('uint8')
    intensity = 1
    for row in im_data.itertuples():
        label = row.label
        label_idx = alphabet[label]
        p0 = row.x0, row.y0
        p1 = row.x1, row.y1
        p2 = row.x2, row.y2
        p3 = row.x3, row.y3
        pts = np.array([p0, p1, p2, p3], np.int32)
        pts = list(get_xs(pts))
        pts = np.array(pts, np.int32)
        pts = [pts.reshape((-1, 1, 2))]
        aux = np.zeros((im.shape[0], im.shape[1]))
        cv2.fillPoly(aux, pts, color=intensity)
        aux = cv2.resize(aux, dsize=dsize_cv2, interpolation=cv2.INTER_CUBIC)
        gt[:, :, label_idx] = aux
    # Filling the zero class (non in the alphabet)
    gt_zero = gt.max(axis=2)
    gt_zero = (gt_zero == 0.0).astype(int)
    gt[:, :, 0] = gt_zero
    return im, gt


def get_image_label_gen(folder, metadata, dsize, in_channels, out_channels, params):
    alphabet = params['alphabet']
    image_name_list = metadata.image_name.unique()
    set_size = len(image_name_list)
    x = np.zeros((set_size, dsize[0], dsize[1], in_channels))
    y = np.zeros((set_size, dsize[0], dsize[1], out_channels))
    for image_name in image_name_list:
        image_data = metadata.loc[metadata.image_name == image_name]
        idx = image_data.idx.values[0]
        im, gt = load_image(image_data, folder, dsize, in_channels, alphabet)
        if in_channels == 3:
            x[idx, :, :, :] = im[:, :, 0:in_channels]
        else:
            x[idx, :, :, 0] = im[:, :]
        y[idx, :, :, :] = gt
    return x, y


def load_label_data(labels):
    labels = labels.groupby(['filename']).apply(
        lambda x: x.assign(point_idx=range(len(x)))).reset_index(drop=True)
    labels_x = labels.pivot(
        index='filename',
        columns='point_idx',
        values='x'
    )
    labels_x.columns = [f"x{c}" for c in labels_x.columns]
    labels_x = labels_x.reset_index(drop=False)
    labels_y = labels.pivot(
        index='filename',
        columns='point_idx',
        values='y'
    )
    labels_y.columns = [f"y{c}" for c in labels_y.columns]
    labels_y = labels_y.reset_index(drop=False)
    labels_wh = labels.drop_duplicates(['filename'])[['filename', 'w', 'h']]
    labels2 = labels_wh.merge(labels_x, on=['filename'], how='left')
    labels2 = labels2.merge(labels_y, on=['filename'], how='left')
    return labels2


def get_plates_bounding_metadata(params):
    metadata = params['metadata']
    labels = params['labels']
    metadata = pd.read_csv(metadata)
    labels = pd.read_csv(labels, sep=',')
    labels = load_label_data(labels)
    labels = labels.rename(columns={'filename': 'image'})
    metadata = metadata.merge(labels, on=['image'], how='left')
    return metadata


def get_plates_text_metadata(params):
    metadata = params['metadata']
    labels = params['labels']
    metadata = pd.read_csv(metadata)
    labels = get_labels_plates_text(labels)
    labels = labels.assign(image_name=labels.filename)
    metadata = metadata.assign(image_name=metadata.image)
    labels.image_name = labels.image_name.str.split('.').str[0]
    labels.image_name = labels.image_name.str.split('_').str[-1]
    metadata.image_name = metadata.image_name.str.split('.').str[0]
    metadata.image_name = metadata.image_name.str.split('_').str[-1]
    metadata = metadata.merge(labels, on=['image_name'], how='left')
    return metadata
