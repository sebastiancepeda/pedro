"""
@author: Sebastian Cepeda
@email: sebastian.cepeda.fuentealba@gmail.com
"""

import pathlib

from io.data_source import get_plates_text_metadata


def get_params():
    path = str(pathlib.Path().absolute().parent.parent)
    folder = f'{path}/data/plates'
    params = {
        'folder': folder,
        'labels': f"{folder}/output_plate_segmentation/labels_plates_text-name_20200724050222.json",
        'metadata': f"{folder}/files.csv",
    }
    return params


def test_get_plates_text_metadata():
    params = get_params()
    plates_text_metadata = get_plates_text_metadata(params)
    a = 0


if __name__ == "__main__":
    test_get_plates_text_metadata()
