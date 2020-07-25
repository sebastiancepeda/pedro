import cv2
import numpy as np
import pandas as pd

from cv.image_processing import get_xs
from io_utils.read_polygons_json import get_labels_plates_text


def load_image(row, folder, dsize):
    image = row.image
    x0, x1, x2, x3 = row.x_0, row.x_1, row.x_2, row.x_3
    y0, y1, y2, y3 = row.y_0, row.y_1, row.y_2, row.y_3
    color = (255, 255, 255)
    im_file = f"{folder}/{image}"
    im = cv2.imread(im_file)
    assert im is not None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    gt = np.zeros((im.shape[0], im.shape[1]))
    gt = gt.astype('uint8')
    pts = np.array([
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [x3, y3]], np.int32)
    p0, p1, p2, p3 = get_xs(pts)
    pts = np.array([p0, p1, p2, p3], np.int32)
    pts = [pts.reshape((-1, 1, 2))]
    cv2.fillPoly(gt, pts, color=color)
    threshold = gt.mean()
    filt = gt >= threshold
    gt[filt] = 1
    gt[~filt] = 0
    im = cv2.resize(im, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    gt = cv2.resize(gt, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    return im, gt


def get_image_label_gen(folder, metadata, dsize):
    set_size = metadata.shape[0]
    x = np.zeros((set_size, dsize[0], dsize[1], 3))
    y = np.zeros((set_size, dsize[0], dsize[1], 2))
    for row in metadata.itertuples():
        idx = row.idx
        im, gt = load_image(row, folder, dsize)
        x[idx, :, :, :] = im[:, :, 0:3]
        y[idx, :, :, 0] = gt
        y[idx, :, :, 1] = gt * -1.0 + 1.0
    return x, y


def load_label_data(labels):
    labels = labels.groupby(['filename']).apply(
        lambda x: x.assign(point_idx=range(len(x)))).reset_index(drop=True)
    labels_x = labels.pivot(
        index='filename',
        columns='point_idx',
        values='x'
    )
    labels_x.columns = [f"x_{c}" for c in labels_x.columns]
    labels_x = labels_x.reset_index(drop=False)
    labels_y = labels.pivot(
        index='filename',
        columns='point_idx',
        values='y'
    )
    labels_y.columns = [f"y_{c}" for c in labels_y.columns]
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
    labels.image_name = labels.image_name.str.split('_').str[1]
    labels.image_name = labels.image_name.str.split('.').str[0]
    metadata.image_name = metadata.image_name.str.split('.').str[0]
    metadata = metadata.merge(labels, on=['image_name'], how='left')
    return metadata
