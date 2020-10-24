"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""
import pandas as pd


def set_index(meta):
    index = meta.file_name.unique()
    index = pd.DataFrame(data={
        'file_name': index,
        'idx': range(len(index)),
    })
    meta = meta.merge(index, on=['file_name'], how='left')
    return meta


class CustomLogger:

    def __init__(self, prefix, base_logger):
        self.info = lambda msg: base_logger.info(f"[{prefix}] {msg}")
        self.debug = lambda msg: base_logger.debug(f"[{prefix}] {msg}")
