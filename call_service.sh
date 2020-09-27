export PYTHONPATH=/home/sebastian/projects/pedro
echo PYTHONPATH: $PYTHONPATH | rm data/plates/alpr/*.png
conda run -n seg_models_cpu python services/alpr/inference.py
