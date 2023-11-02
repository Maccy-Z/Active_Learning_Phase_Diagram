# Code to plot the results. Run directly for automatic evaluation

import numpy as np
from mayavi import mlab
import vtk
import time

from gaussian_sampler import suggest_point, suggest_two_points
from utils import ObsHolder, new_save_folder
from config import Config
from mayavi.core.scene import Scene
from tvtk.util.ctf import ColorTransferFunction


class Plotter:
    name: str

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # Assume this is your custom function that maps scalar values to RGB colors
    @staticmethod
    def custom_color_function(value, min_s, max_s):
        raise NotImplementedError

    # Similar to the custom_opacity_function, this is your function for opacity.
    @staticmethod
    def custom_opacity_function(value, min_s, max_s):
        raise NotImplementedError

    def plot3d(self, scene, scalar_field):
        scene = scene.scene.mayavi_scene
        self.plot(scene, scalar_field)

    def plot(self, scene, scalar_field):
        min_scalar = scalar_field.min()
        max_scalar = scalar_field.max()

        # First, define the points of the scalar field
        ex = np.array(self.cfg.extent)
        shape = scalar_field.shape
        xs, ys, zs = np.mgrid[ex[0, 0]:ex[0, 1]:shape[0] * 1j, ex[1, 0]:ex[1, 1]:shape[1] * 1j, ex[2, 0]:ex[2, 1]:shape[2] * 1j]

        # Plot the scalar field
        src = mlab.pipeline.scalar_field(xs, ys, zs, scalar_field, figure=scene)
        vol = mlab.pipeline.volume(src, figure=scene)

        # Create the color/opacity transfer functions
        color_tf = ColorTransferFunction()
        opacity_tf = vtk.vtkPiecewiseFunction()

        num_samples = 256  # Number of points to sample the function over the range of data
        for i in range(num_samples):
            scalar_value = min_scalar + (max_scalar - min_scalar) * float(i) / (num_samples - 1)
            # For the color transfer function
            R, G, B = self.custom_color_function(scalar_value, min_scalar, max_scalar)
            color_tf.add_rgb_point(scalar_value, R, G, B)

            # For the opacity transfer function
            opacity = self.custom_opacity_function(scalar_value, min_scalar, max_scalar)
            opacity_tf.AddPoint(scalar_value, opacity)

        # Apply the color and opacity transfer functions
        vol._volume_property.set_color(color_tf)
        vol._ctf = color_tf
        vol._volume_property.set_scalar_opacity(opacity_tf)

        # Make the axis visible
        axes = mlab.axes(vol, nb_labels=3, ranges=ex.flatten(), line_width=1.0, color=(0, 0, 0), figure=scene)
        label_text_property = axes.axes.axis_label_text_property
        label_text_property.color = (0, 0, 0)

        title = mlab.title(self.name, figure=scene)
        title.property.color = (0, 0, 0)
        #
        # if self.name == "Phase Diagram":
        #     mlab.points3d(0, 0, 0, scale_factor=1, color=(1, 1, 0), figure=scene)


class ProbPlotter(Plotter):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.name = "Error Probability"

    @staticmethod
    def custom_color_function(error_prob, min_s, max_s):
        error_prob = error_prob / max_s
        R = 1-error_prob
        G = 0.3*error_prob
        B = 0.7*error_prob + 0.3
        return R, G, B

    @staticmethod
    def custom_opacity_function(error_prob, min_s, max_s):
        # Clip the max probs to make it easier to see
        max_prob = np.clip(max_s, 0.5, 1)
        error_prob = error_prob / max_prob
        return error_prob


class AcquisitionPlotter(Plotter):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.name = "Acq. Function"

    @staticmethod
    def custom_color_function(acq, min_s, max_s):
        acq = (acq / max_s)
        R = acq
        G = acq ** 6
        B = 1-acq
        return R, G, B

    @staticmethod
    def custom_opacity_function(acq, min_s, max_s):
        #max_s = np.clip(max_s, 0.01, 1)
        acq = acq / max_s
        return acq


class PhasePlotter(Plotter):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.name = "Phase Diagram"

    @staticmethod
    def custom_color_function(value, min_s, max_s):
        if value < 0.5:
            R = 1
            G = 0
            B = 0
        elif value < 1.5:
            R = 0
            G = 1
            B = 0
        elif value < 2.5:
            R = 0
            G = 0
            B = 1
        elif value < 3.5:
            R = 1
            G = 1
            B = 0
        else:
            R = 0
            G = 1
            B = 1
        return R, G, B

    @staticmethod
    def custom_opacity_function(value, min_s, max_s):
        return 0.7

    @staticmethod
    def get_colormap_lut(phase_obs):
        s = np.arange(len(phase_obs))
        lut = []
        for phase in phase_obs:
            eps = 50
            if phase < 0.5:
                R = 255
                G = eps
                B = eps
            elif phase < 1.5:
                R = eps
                G = 255
                B = eps
            elif phase < 2.5:
                R = eps
                G = eps
                B = 255
            elif phase < 3.5:
                R = 255
                G = 255
                B = eps
            else:
                R = eps
                G = 255
                B = 255
            lut.append([R, G, B, 255])
        return s, lut

    def plot3d(self, scene, scalar_field, old_points, phase_obs, new_point):
        scene = scene.scene.mayavi_scene
        # Slightly noise observations to avoid poor quality plots (its a bug)
        scalar_field = scalar_field + 1e-5 * np.random.normal(*scalar_field.shape)

        self.plot(scene, scalar_field)

        xs, ys, zs = old_points
        s, lut = self.get_colormap_lut(phase_obs)

        points = mlab.points3d(xs, ys, zs, s, scale_factor=0.1, figure=scene, scale_mode='none')
        points.module_manager.scalar_lut_manager.lut.number_of_colors = len(phase_obs)
        points.module_manager.scalar_lut_manager.lut.table = lut

        xs, ys, zs = new_point
        mlab.points3d(xs, ys, zs, scale_factor=0.1, color=(1, 1, 1), figure=scene)


class DistanceSampler3d:
    point_holder: dict
    mayavi_scenes: list

    def __init__(self, init_phases, init_Xs, cfg: Config, save_dir="./saves"):
        save_path = new_save_folder(save_dir)
        print(f'{save_path = }')
        print()

        self.cfg = cfg
        self.obs_holder = ObsHolder(self.cfg, save_path=save_path)

        # This only works for 3d
        assert self.cfg.N_dim == 3

        # Check if initial inputs are within allowed area
        points = np.array(init_Xs)
        if points.shape[1] != self.cfg.N_dim:
            raise ValueError(f"Number of dimensions must be {self.cfg.N_dim}, got {points.shape[1]}")
        bounds = np.array(self.cfg.extent)
        mins, maxs = bounds[:, 0], bounds[:, 1]

        all_in = np.logical_and(points >= mins, points <= maxs).all(axis=1)
        if not all_in.all():
            print("\033[31mWarning: Observations are outside search area\033[0m")

        # Check phases are allowed
        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)

        self.prob_plotter = ProbPlotter(cfg)
        self.acq_plotter = AcquisitionPlotter(cfg)
        self.phase_plotter = PhasePlotter(cfg)

    # Load in axes for plotting
    def set_scenes(self, mavi_scenes: list):
        self.mayavi_scenes = []
        for scene in mavi_scenes:
            self.mayavi_scenes.append(scene)

    def add_obs(self, phase: int, X: np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)
        self.obs_holder.save()

    def single_obs(self, ):
        # self.obs_holder.save()

        new_point, (pd_old, acq_fn, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)

        # Reshape and process raw data for plotting
        pd = pd_old.reshape([self.cfg.N_display for _ in range(self.cfg.N_dim)])
        acq_fn = acq_fn.reshape([self.cfg.N_eval for _ in range(self.cfg.N_dim)])
        error_prob = 1 - np.amax(pd_probs, axis=1)
        error_prob = error_prob.reshape([self.cfg.N_eval for _ in range(self.cfg.N_dim)])

        self.point_holder = {'new_point': new_point, 'pd': pd, 'acq_fn': acq_fn, 'error_prob': error_prob, 'pd_probs': pd_probs}
        self.plot()

        return new_point, prob_at_point

    def plot(self):
        # Get existing ovservations
        X_obs, phase_obs = self.obs_holder.get_og_obs()
        xs, ys, zs = X_obs[:, 0], X_obs[:, 1], X_obs[:, 2]

        new_p = self.point_holder['new_point']

        # clear old plots
        [scene.clear_plot() for scene in self.mayavi_scenes]
        self.prob_plotter.plot3d(self.mayavi_scenes[0], self.point_holder['error_prob'])
        self.acq_plotter.plot3d(self.mayavi_scenes[1], self.point_holder['acq_fn'])
        self.phase_plotter.plot3d(self.mayavi_scenes[2], self.point_holder['pd'], (xs, ys, zs), phase_obs, new_p)

        # print(self.point_holder['pd'])


def main(save_dir):
    print(save_dir)
    cfg = Config()

    # Init observations to start off
    # X_init, _, _ = make_grid(cfg.N_init, cfg.extent)
    X_init = [[0.0, 1., 2], [0, 1.25, 2.59]]
    phase_init = [1, 2]
    print(phase_init)

    distance_sampler = DistanceSampler3d(phase_init, X_init, cfg, save_dir=save_dir)

    [mlab.figure(figure=i, bgcolor=(0.9, 0.9, 0.9)) for i in range(3)]
    for i in range(51):
        print()
        print("Step:", i)
        new_points, prob = distance_sampler.single_obs()
        print(f'{new_points =}, {prob = }')

        distance_sampler.obs_holder.add_obs(new_points, phase=0)


if __name__ == "__main__":
    save_dir = "./saves"

    main(save_dir)
