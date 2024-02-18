# Code to plot the results. Run directly for automatic evaluation
import numpy as np

from DistanceSampler import DistanceSampler
from gaussian_sampler import suggest_point
from config import Config
from utils import to_real_scale

from tvtk.util.ctf import ColorTransferFunction
from mayavi import mlab
import vtk


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
        ex = np.array(self.cfg.unit_extent)
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
        ex_true = np.array(self.cfg.extent)
        axes = mlab.axes(vol, nb_labels=3, ranges=ex_true.flatten(), line_width=1.0, color=(0, 0, 0), figure=scene)
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
        R = 0.7 * error_prob + 0.31 - error_prob
        G = 0.3 * error_prob
        B = 1 - error_prob
        return R, G, B

    @staticmethod
    def custom_opacity_function(error_prob, min_s, max_s):
        # Clip the max probs to make it easier to see
        max_prob = np.clip(max_s, 0.25, 1)
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
        B = 1 - acq
        return R, G, B

    @staticmethod
    def custom_opacity_function(acq, min_s, max_s):
        # max_s = np.clip(max_s, 0.01, 1)
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
        return 0.5

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
        point_size = 0.1

        # Slightly noise observations to avoid poor quality plots
        scalar_field = scalar_field + 1e-5 * np.random.normal(*scalar_field.shape)
        self.plot(scene, scalar_field)

        # Old points
        xs, ys, zs = old_points
        s, lut = self.get_colormap_lut(phase_obs)

        points = mlab.points3d(xs, ys, zs, s, scale_factor=point_size, figure=scene, scale_mode='none')
        # lut requires at least 2 points
        if len(phase_obs) > 2:
            points.module_manager.scalar_lut_manager.lut.number_of_colors = len(phase_obs)
            points.module_manager.scalar_lut_manager.lut.table = lut

        # Next point to sample
        xs, ys, zs = new_point
        mlab.points3d(xs, ys, zs, scale_factor=point_size, color=(1, 1, 1), figure=scene, scale_mode='none')
        # print(f'{new_point = }, {old_points = }')


class DistanceSampler3D(DistanceSampler):
    mayavi_scenes: list

    def __init__(self, init_phases, init_Xs, cfg: Config, init_probs=None, save_dir="./saves"):
        super().__init__(init_phases, init_Xs, init_probs=init_probs, cfg=cfg, save_dir=save_dir)
        assert cfg.N_dim == 3, "3D sampling and plotting only"

        self.prob_plotter = ProbPlotter(cfg)
        self.acq_plotter = AcquisitionPlotter(cfg)
        self.phase_plotter = PhasePlotter(cfg)

    # Load in axes for plotting
    def set_scenes(self, mavi_scenes: list):
        self.mayavi_scenes = []
        for scene in mavi_scenes:
            self.mayavi_scenes.append(scene)

    def single_obs(self):
        """
        Suggest a point and plot the results
        Note plotting is done in unit scale, but returns new_point as real scale
        """
        new_point, prob_at_point, (pd_old, acq_fn, pd_probs) = suggest_point(self.pool, self.obs_holder, self.cfg)
        # Reshape and process raw data for plotting
        pd = pd_old.reshape([self.cfg.N_display for _ in range(3)])
        acq_fn = acq_fn.reshape([self.cfg.N_eval for _ in range(3)])
        error_prob = 1 - np.amax(pd_probs, axis=1)
        error_prob = error_prob.reshape([self.cfg.N_eval for _ in range(3)])

        plot_holder = {'new_point': new_point, 'pd': pd, 'acq_fn': acq_fn, 'error_prob': error_prob, 'pd_probs': pd_probs}
        self.plot(plot_holder)

        return to_real_scale(new_point, self.cfg.extent), prob_at_point

    def plot(self, plot_holder):
        # Get existing ovservations
        X_obs, phase_obs, _ = self.obs_holder.get_obs()
        xs, ys, zs = X_obs[:, 0], X_obs[:, 1], X_obs[:, 2]

        # clear old plots
        [scene.clear_plot() for scene in self.mayavi_scenes]

        self.prob_plotter.plot3d(self.mayavi_scenes[0], plot_holder['error_prob'])
        self.acq_plotter.plot3d(self.mayavi_scenes[1], plot_holder['acq_fn'])
        self.phase_plotter.plot3d(self.mayavi_scenes[2], plot_holder['pd'], (xs, ys, zs), phase_obs, plot_holder['new_point'])


def main(save_dir):
    assert False, "Running 3D plots not currently supported. "
    print(save_dir)
    cfg = Config()

    # Init observations to start off
    # X_init, _, _ = make_grid(cfg.N_init, cfg.extent)
    X_init = [[0.0, 0.1, 0.2], [0, 0.25, 0.1]]
    phase_init = [1, 2]
    print(phase_init)

    distance_sampler = DistanceSampler3D(phase_init, X_init, cfg, save_dir=save_dir)

    scenes = [mlab.figure(figure=i, bgcolor=(0.9, 0.9, 0.9)) for i in range(3)]
    distance_sampler.set_scenes(scenes)
    for i in range(51):
        print()
        print("Step:", i)
        new_points, prob = distance_sampler.single_obs()
        print(f'{new_points =}, {prob = }')

        distance_sampler.add_obs(X=new_points, phase=0)


if __name__ == "__main__":
    save_dir = "./saves"
    main(save_dir)
