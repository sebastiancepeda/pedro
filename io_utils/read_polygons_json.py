"""
@author: sebastian
"""
import json
import pandas as pd


def get_image_polygons(row):
    regions = row['regions']
    vals = []
    for region in regions.values():
        label = region['region_attributes']['label']
        x = region['shape_attributes']['all_points_x'][0:-1]
        y = region['shape_attributes']['all_points_y'][0:-1]
        new_val = {
            'filename': row['filename'],
            'label': label,
            'x0': x[0],
            'y0': y[0],
            'x1': x[1],
            'y1': y[1],
            'x2': x[2],
            'y2': y[2],
            'x3': x[3],
            'y3': y[3],
        }
        new_val = pd.DataFrame.from_dict(new_val, orient='index')
        new_val = new_val.T
        vals.append(new_val)
    vals = pd.concat(vals, axis=0)
    return vals


def get_labels_plates_text(file):
    with open(file) as json_file:
        data = json.load(json_file)
    data = list(data.values())
    data = [get_image_polygons(row) for row in data]
    data = pd.concat(data, axis=0).reset_index(drop=True)
    int_cols = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']
    for c in int_cols:
        data[c] = data[c].astype(int)
    return data
