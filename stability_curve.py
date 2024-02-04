import numpy as np
import trimesh
from trimesh import Trimesh
from trimesh.path import Path2D
from pathlib import Path
from trimesh import transformations, intersections
from typing import Union, List
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


def change_sign(
        lst: list[Union[float, int]]
) -> list:
    """
    Simple function that changes the signs of floats/ints in a list
    :param lst: containing floats or ints
    """
    return [-i for i in lst]


class Vessel:

    def __init__(self, mesh_path, length, breadth, draft, center_of_gravity):

        self.mesh = Mesh(mesh_path)
        self.length = length
        self.breadth = breadth
        self.draft = draft
        self.waterline, self.underwater_vessel = self.mesh.slice_mesh([0, 0, self.draft * 1e3], [0, 0, -1])
        _, self.abovewater_vessel = self.mesh.slice_mesh([0, 0, self.draft * 1e3], [0, 0, 1])
        self.center_of_gravity = center_of_gravity
        self.heel_counter = 0

        self.center_of_floatation = self.waterline.centroid * 1e-3  # [m]
        self.center_of_buoyancy = self.underwater_vessel.center_mass * 1e-3  # [m]
        self.GZt = self.center_of_buoyancy[1] - self.center_of_gravity[1]  # Horizontal distance between B and G

        self.cross_section_history = []
        self.righting_arm_history = []

    def heel(self, angle, center):

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

        self.waterline, self.underwater_vessel = self.mesh.slice_mesh([0, 0, self.draft * 1e3], [0, 0, -1])
        _, self.abovewater_vessel = self.mesh.slice_mesh([0, 0, self.draft * 1e3], [0, 0, 1])
        # plot_3d_mesh([self.underwater_vessel, self.abovewater_vessel])
        self.center_of_floatation = self.waterline.centroid * 1e-3  # [m]
        self.center_of_buoyancy = self.underwater_vessel.center_mass * 1e-3  # [m]
        self.GZt = self.center_of_buoyancy[1] - self.center_of_gravity[1]  # Horizontal distance between B and G

        self.cross_section_history.append(self.cross_section([self.length/2 * 1e3, 0, 0]))

    def animate_cross_section(self):

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        margin = 10

        def update(frame):
            cross_section = self.cross_section_history[frame][0]
            center_of_gravity = self.cross_section_history[frame][1]
            center_of_buoyancy = self.cross_section_history[frame][2]

            ax[0].clear()
            ax[0].plot(self.righting_arm_history[0], self.righting_arm_history[1])
            ax[0].scatter(self.righting_arm_history[0][frame], self.righting_arm_history[1][frame])
            ax[0].set_xlabel('Heeling angle [deg]')
            ax[0].set_ylabel('Righting arm [m]')
            ax[0].grid(True)

            ax[1].clear()  # clearing the axes
            ax[1].plot(cross_section.discrete[0][:, 1] / 1e3, cross_section.discrete[0][:, 0] / 1e3, color='black')
            ax[1].axhline(self.draft)
            ax[1].scatter(center_of_gravity[1], center_of_gravity[2], label='G')
            ax[1].text(center_of_gravity[1] - 0.6, center_of_gravity[2] + 0.6, 'G')
            ax[1].scatter(center_of_buoyancy[1], center_of_buoyancy[2], label='B')
            ax[1].text(center_of_buoyancy[1] - 0.6, center_of_buoyancy[2] + 0.6, 'B')
            ax[1].set_xlim(-self.breadth / 2 - margin, self.breadth / 2 + margin)  # Set x-axis limits
            ax[1].set_ylim(-self.breadth / 2 - margin, self.breadth / 2 + margin + self.draft)  # Set y-axis limits
            ax[1].grid(False)
            ax[1].set_aspect('equal', adjustable="datalim")

            fig.canvas.draw()  # forcing the artist to redraw itself

        animation = FuncAnimation(fig, update, frames=range(len(self.cross_section_history)), interval=30)
        # To save the animation as a GIF, you can use the following line
        # animation.save('stability curve.gif', writer='pillow')
        plt.show()

    def stability_curve(self):
        """
        Method that calculates the GZ-curve for vessel
        """

        # Specify the heeling increment and range to iterate over
        heeling_increment = 1
        heeling_angles = np.arange(0, 90 + heeling_increment, heeling_increment)

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
        cross_section, _ = self.mesh.slice_mesh(plane_origin=plane_origin,
                                                plane_normal=plane_normal)

        return [cross_section, self.center_of_gravity, self.center_of_buoyancy]


class Mesh:

    def __init__(
            self,
            mesh_path: Path
    ) -> None:

        self.mesh = trimesh.load(mesh_path)

    def slice_mesh(
            self,
            plane_origin: list[float],
            plane_normal: list[float],
    ) -> Union[tuple, Path2D]:
        """
        Method to slice the mesh at a specific location and orientation
        :param plane_origin: coordinate containing the location of the slice
        :param plane_normal: direction of the slice
        :return: Path2D object containing the intersection and Trimesh object containing the sliced object
        """

        # Calls Trimesh method that returns the intersection outline at specified location
        intersection = self.mesh.section_multiplane(plane_origin=plane_origin,
                                                    plane_normal=change_sign(plane_normal),
                                                    heights=[0])

        # Calls Trimesh method that slices the mesh and caps it
        sliced_half_closed = trimesh.intersections.slice_mesh_plane(mesh=self.mesh,
                                                                    plane_normal=plane_normal,
                                                                    plane_origin=plane_origin,
                                                                    cap=True)

        return intersection[0], sliced_half_closed

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
    Roll a coordinate

    :param point: Coordinate to roll
    :param angle: Angle to roll
    :param center: Coordinate to roll around

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
