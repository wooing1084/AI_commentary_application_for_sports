git clone 'https://github.com/facebookresearch/detectron2'
dist=$(python -c "from distutils.core import run_setup; dist = run_setup('./detectron2/setup.py'); print(' '.join([f\"'{x}'\" for x in dist.install_requires]))")
python -m pip install $dist
python -c "import os, sys; sys.path.insert(0, os.path.abspath('./detectron2'))"