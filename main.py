from stability_curve import Vessel
from pathlib import Path
import numpy as np

vessel = Vessel(Path('data/example_hull.stl'), 160, 20, 6, np.array([60, 0, 9]))
vessel.stability_curve()
vessel.animate_cross_section()
