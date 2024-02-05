import numpy as np
import trimesh
from trimesh import Trimesh
from trimesh.path import Path2D
from pathlib import Path
from trimesh import transformations
from typing import Union, List
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

# Color codes for plotting
COLORS = {'light blue': '#B7D5D4',
          'tropical indigo': '#7D83FF',
          'field drab': '#65532F',
          'licorice': '#120309',
          'pacific cyan': '#23B5D3',
          'fire red': '#C81D25'}


def change_sign(
        lst: list[Union[float, int]]
) -> list:
    """
    Simple function that changes the signs of floats/ints in a list
    :param lst: containing floats or ints
    """
    return [-i for i in lst]


class Vessel:

    def __init__(self, mesh_path: Path, length: float, breadth: float, draft: float, center_of_gravity: np.array
                 ) -> None:

        self.mesh = Mesh(mesh_path)
        self.length = length
        self.breadth = breadth
        self.draft = draft
        self.waterline, self.underwater_vessel = self.mesh.clip_mesh([0, 0, self.draft * 1e3], [0, 0, -1])
        _, self.abovewater_vessel = self.mesh.clip_mesh([0, 0, self.draft * 1e3], [0, 0, 1])
        self.center_of_gravity = center_of_gravity
        self.heel_counter = 0
        self.center_of_floatation = self.waterline.centroid * 1e-3  # [m]
        self.center_of_buoyancy = self.underwater_vessel.center_mass * 1e-3  # [m]
        self.GZt = self.center_of_buoyancy[1] - self.center_of_gravity[1]  # Horizontal distance between B and G

        self.mesh_history=[]
        self.cross_section_history = []
        self.righting_arm_history = []

    def heel(self, angle: float, center: np.array):

        # Subtract the angle counter, to heel with the actual input angle
        angle = angle - self.heel_counter

        # Keep track of current heeling angle
        self.heel_counter += angle

        # Convert angle to radians
        angle = float(np.radians(angle))
        # Rotate around the x-axis
        direction = [1, 0, 0]
        # Specify the center of the rotation
        self.center_of_gravity = rotate_roll(self.center_of_gravity, angle, center / 1e3)

        # Apply the rotation the vessel mesh and centerline
        self.mesh.rotate_mesh(angle, direction, center)

        # Update the vessel with the new COGs
        self.update()

    def update(self):

        self.waterline, self.underwater_vessel = self.mesh.clip_mesh([0, 0, self.draft * 1e3], [0, 0, -1])
        _, self.abovewater_vessel = self.mesh.clip_mesh([0, 0, self.draft * 1e3], [0, 0, 1])
        self.center_of_floatation = self.waterline.centroid * 1e-3  # [m]
        self.center_of_buoyancy = self.underwater_vessel.center_mass * 1e-3  # [m]
        self.GZt = self.center_of_buoyancy[1] - self.center_of_gravity[1]  # Horizontal distance between B and G

        self.cross_section_history.append(self.cross_section([self.length/2 * 1e3, 0, 0]))
        self.mesh_history.append([self.abovewater_vessel, self.underwater_vessel])

    def animate_cross_section(self, save_as: str = None):

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        margin = 10
        plt.style.use('data/style_light.mplstyle')

        def update(frame):
            cross_section = self.cross_section_history[frame][0]
            center_of_gravity = self.cross_section_history[frame][1]
            center_of_buoyancy = self.cross_section_history[frame][2]

            ax[0].clear()
            ax[0].plot(self.righting_arm_history[0], self.righting_arm_history[1])
            ax[0].scatter(self.righting_arm_history[0][frame], self.righting_arm_history[1][frame], color=COLORS['pacific cyan'])
            ax[0].axhline(0, color=COLORS['licorice'], linestyle='dashed')
            ax[0].set_xlabel('Heeling angle [deg]')
            ax[0].set_ylabel('Righting arm [m]')
            ax[0].grid(True)

            ax[1].clear()  # clearing the axes
            ax[1].plot(cross_section.discrete[0][:, 1] / 1e3, cross_section.discrete[0][:, 0] / 1e3, color=COLORS['licorice'])
            ax[1].axhline(self.draft, color=COLORS['pacific cyan'])
            ax[1].scatter(center_of_gravity[1], center_of_gravity[2], label='G')
            ax[1].text(center_of_gravity[1] - 0.6, center_of_gravity[2] + 0.6, 'G')
            ax[1].scatter(center_of_buoyancy[1], center_of_buoyancy[2], label='B')
            ax[1].text(center_of_buoyancy[1] - 0.6, center_of_buoyancy[2] + 0.6, 'B')
            ax[1].set_xlim(-self.breadth / 2 - margin, self.breadth / 2 + margin)
            ax[1].set_ylim(-self.breadth / 2 - margin, self.breadth / 2 + margin + self.draft)
            ax[1].grid(False)
            ax[1].set_aspect('equal', adjustable="datalim")
            plt.suptitle('righting arm curve and vessel cross-section'.title(), size=16 ,fontweight='bold')
            fig.canvas.draw()

        animation = FuncAnimation(fig, update, frames=range(len(self.cross_section_history)), interval=50)
        if save_as is not None:
            animation.save(save_as, writer='pillow')
        plt.show()

    def stability_curve(self, heeling_range: list, increment: float) -> List[np.array]:
        """
        Method that calculates the GZ-curve for vessel
        """

        # Specify the heeling increment and range to iterate over
        heeling_increment = increment
        heeling_angles = np.arange(heeling_range[0], heeling_range[1] + heeling_increment, heeling_increment)

        # Create progress bar
        progress_bar = tqdm(total=len(heeling_angles), desc='Calculating GZ-curve', unit='angle')

        # Create empty array to store the righting arms
        gz_array = np.zeros(len(heeling_angles))

        # Center of rotation
        center = np.array(
            [self.center_of_floatation[0], self.center_of_floatation[1], self.draft]) * 1e3  # Center of rotation

        # Calculate righting arm for all the heeling angles
        for i, heeling_angle in enumerate(heeling_angles):
            # Heel vessel
            self.heel(heeling_angle, center)

            # Save the righting arm to the array
            gz_array[i] = -self.GZt

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        self.righting_arm_history = [heeling_angles, gz_array]
        return self.righting_arm_history

    def cross_section(
            self,
            location: list[float],
    ) -> list:
        """
        Function that creates a cross-section of the vessel
        :param location: List containing the coordinate of the cross-section
        """

        # Unpack 'location'
        plane_origin = location

        # Select plane normal based on 'orientation' input
        plane_normal = [1, 0, 0]

        # Make cross-section on the specified plane origin and normal
        cross_section, _ = self.mesh.clip_mesh(plane_origin=plane_origin,
                                               plane_normal=plane_normal)

        return [cross_section, self.center_of_gravity, self.center_of_buoyancy]

    def animate_3d_mesh(self) -> None:
        plt.style.use('data/style_light.mplstyle')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = [COLORS['fire red'], COLORS['light blue']]

        def update(frame):
            ax.clear()
            # ax.view_init(elev=15, azim=215)  # Set the elevation and azimuthal angle
            for i, mesh_obj in enumerate(self.mesh_history[frame]):
                ax.plot_trisurf(mesh_obj.vertices[:, 0], mesh_obj.vertices[:, 1], triangles=mesh_obj.faces,
                                Z=mesh_obj.vertices[:, 2], color=colors[i])
            set_axes_equal(ax)

        animation = FuncAnimation(fig, update, frames=range(len(self.mesh_history)), interval=50)
        animation.save('data/3d animation.gif', writer='pillow')
        # plt.show()


class Mesh:

    def __init__(
            self,
            mesh_path: Path
    ) -> None:

        self.mesh = trimesh.load(mesh_path)

    def clip_mesh(
            self,
            plane_origin: list[float],
            plane_normal: list[float],
    ) -> Union[tuple, Path2D]:
        """
        Method to clip the mesh at a specific location and orientation
        :param plane_origin: coordinate containing the location of the slice
        :param plane_normal: direction of the slice
        :return: Path2D object containing the intersection and Trimesh object containing the sliced object
        """

        # Calls Trimesh method that returns the intersection outline at specified location
        intersection = self.mesh.section_multiplane(plane_origin=plane_origin,
                                                    plane_normal=change_sign(plane_normal),
                                                    heights=[0])

        # Calls Trimesh method that slices the mesh and caps it
        clipped_half_closed = trimesh.intersections.slice_mesh_plane(mesh=self.mesh,
                                                                    plane_normal=plane_normal,
                                                                    plane_origin=plane_origin,
                                                                    cap=True)

        return intersection[0], clipped_half_closed

    def rotate_mesh(
            self,
            angle: float,
            direction: Union[list[float], np.ndarray],
            center: Union[list[float], np.ndarray]
    ) -> None:
        """
        Method that rotates the mesh
        :param angle: angle in radians for the rotation
        :param direction: axis for the rotation
        :param center: coordinate for the center of the rotation
        """

        rot_matrix = transformations.rotation_matrix(angle, direction, center)
        self.mesh = self.mesh.apply_transform(rot_matrix)


def plot_3d_mesh(
        mesh_objects: List[Trimesh]
) -> None:
    """
    Simple function to plot a mesh in 3D
    :param mesh_objects: list of Trimesh objects to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for mesh_obj in mesh_objects:
        ax.plot_trisurf(mesh_obj.vertices[:, 0], mesh_obj.vertices[:,1], triangles=mesh_obj.faces, Z=mesh_obj.vertices[:,2])
    set_axes_equal(ax)
    plt.show()


def set_axes_equal(
        ax: plt.Axes
) -> None:
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    :param ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def rotate_roll(
        point: np.ndarray,
        angle: float,
        center: np.ndarray
) -> Union[np.ndarray, None]:
    """
    Rotate a coordinate around the x-axis

    :param point: Coordinate to rotate
    :param angle: Angle to rotate
    :param center: Coordinate to rotate around

    :return: The rotated coordinate
    """

    # Translate the point to the center
    translated_point = np.subtract(point, center)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])

    # Multiply the coordinate with the rotation matrix
    rotated_point = np.matmul(rotation_matrix, translated_point.T)
    # Translate to the correct position
    final_rotated_point = np.add(rotated_point, center)

    # Return rotated coordinate
    return final_rotated_point
