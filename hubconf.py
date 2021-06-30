dependencies = ['torch', 'unet']


import sys
from pathlib import Path
code_dir = Path(__file__).parent / 'resseg'
sys.path.insert(0, str(code_dir))
from model import ressegnet
