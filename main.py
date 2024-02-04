from pathlib import Path
import numpy as np
from stability_curve import Vessel


if __name__ == '__main__':
    vessel = Vessel(Path('data/example_hull.stl'), length=160, breadth=20, draft=6, center_of_gravity=np.array([60, 0, 10]))
    vessel.stability_curve(heeling_range=[-5, 100], increment=2)
    vessel.animate_cross_section()
