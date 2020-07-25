"""
@author: sebastian
"""
import json
import pandas as pd


def get_labels_plates_text(file):
    with open(file) as json_file:
        data = json.load(json_file)
        data = list(data.values())
    data2 = []
    for row in data:
        label = row['regions']['0']['region_attributes']['label']
        x = row['regions']['0']['shape_attributes']['all_points_x'][0:-1]
        y = row['regions']['0']['shape_attributes']['all_points_y'][0:-1]
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
        data2.append(new_val)
    data = pd.concat(data2, axis=0).reset_index(drop=True)
    data.x0 = data.x0.astype(int)
    data.y0 = data.y0.astype(int)
    data.x1 = data.x1.astype(int)
    data.y1 = data.y1.astype(int)
    data.x2 = data.x2.astype(int)
    data.y2 = data.y2.astype(int)
    data.x3 = data.x3.astype(int)
    data.y3 = data.y3.astype(int)
    return data
