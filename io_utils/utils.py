"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import pandas as pd


def set_index(meta):
    index = meta.image_name.unique()
    index = pd.DataFrame(data={
        'image_name': index,
        'idx': range(len(index)),
    })
    meta = meta.merge(index, on=['image_name'], how='left')
    return meta
