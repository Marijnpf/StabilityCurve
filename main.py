from stability_curve import Vessel
from pathlib import Path
import numpy as np

vessel = Vessel(Path('data/hull_mesh.stl'), 120, 40, 1.5, np.array([60, 0, 15]))
vessel.stability_curve()
vessel.animate_cross_section()

print('test')