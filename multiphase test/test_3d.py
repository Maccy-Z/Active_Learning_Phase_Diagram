import sys
from PyQt5 import QtWidgets
from mayavi import mlab
from mayavi.core.api import Engine
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from tvtk.pyface.scene_model import SceneModel
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import numpy as np


def custom_plot(scene, scale):
    scalar_field = np.random.random((10, 10, 10))
    ex = np.array([[0, 1], [0, 1], [0, 1]])
    shape = scalar_field.shape
    xs, ys, zs = np.mgrid[ex[0, 0]:ex[0, 1]:shape[0] * 1j, ex[1, 0]:ex[1, 1]:shape[1] * 1j, ex[2, 0]:ex[2, 1]:shape[2] * 1j]

    # Plot the scalar field
    src = mlab.pipeline.scalar_field(xs, ys, zs, scalar_field)
    vol = mlab.pipeline.volume(src, figure=scene)

    return


def custom_plot2(scene, scale):
    points = np.array([[0, scale, 0], [0, 1, 1], [0, 1, 0]])
    plot = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=1, color=(0, 1, 0), figure=scene)
    plot = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], scale_factor=1, color=(0, 1, 1), figure=scene)
    return plot


def custom_plot3(scene, scale):
    scene.mlab.test_points3d()


class Visualization(HasTraits):
    engine = Instance(Engine, args=())
    scene = Instance(MlabSceneModel)
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)  # Make the view resizable.

    def _scene_default(self):
        self.engine.start()
        scene = MlabSceneModel(engine=self.engine)
        return scene

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        # mlab.clf(figure=self.scene.mayavi_scene)
        scalar_field = np.random.random((10, 10, 10))
        ex = np.array([[0, 1], [0, 1], [0, 1]])
        shape = scalar_field.shape
        xs, ys, zs = np.mgrid[ex[0, 0]:ex[0, 1]:shape[0] * 1j, ex[1, 0]:ex[1, 1]:shape[1] * 1j, ex[2, 0]:ex[2, 1]:shape[2] * 1j]

        # Plot the scalar field
        src = mlab.pipeline.scalar_field(xs, ys, zs, scalar_field, figure=self.scene.mayavi_scene)
        vol = mlab.pipeline.volume(src, figure=self.scene.mayavi_scene, color=(1, 0, 0))

class Visualization2(HasTraits):
    engine = Instance(Engine, args=())
    scene = Instance(MlabSceneModel)
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)  # Make the view resizable.

    def _scene_default(self):
        self.engine.start()
        return MlabSceneModel(engine=self.engine)

    def __init__(self):
        #super().__init__()
        self.update_plot(1.)

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        #mlab.clf(figure=self.scene.mayavi_scene)
        custom_plot2(self.scene.mayavi_scene, scale)


class Visualization3(HasTraits):
    engine = Instance(Engine, args=())
    scene = Instance(MlabSceneModel)
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)  # Make the view resizable.

    def _scene_default(self):
        self.engine.start()
        return MlabSceneModel(engine=self.engine)

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        #mlab.clf(figure=self.scene.mayavi_scene)
        custom_plot3(self.scene, scale)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Interactive Mayavi plots in PyQt')

        # Make the plots
        self.plots = []
        self.plots.append(Visualization())
        self.plots.append(Visualization())
        self.plots.append(Visualization3())

        # Creating the main widget
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # Creating a vertical layout to arrange elements
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Create a horizontal layout for the Mayavi views
        mayavi_layout = QtWidgets.QHBoxLayout()

        # Embedding Mayavi scenes as subplots
        for plot in self.plots:
            mayavi_layout.addWidget(plot.edit_traits(parent=self, kind='subpanel').control)

        # Add the Mayavi layout to the main layout
        layout.addLayout(mayavi_layout)

        # Adding a textbox for data input
        self.textbox = QtWidgets.QLineEdit()
        layout.addWidget(self.textbox)

        # Connect the textbox input event to an update function
        self.textbox.returnPressed.connect(self.on_text_enter)

        self.init_plots()

    def init_plots(self):
        for plot in self.plots:
            plot.update_plot(1.)

    def on_text_enter(self):
        # When the user presses Enter, this gets executed.
        text_value = self.textbox.text()
        try:
            # Convert the text box value to a float
            scale_factor = float(text_value)

            # Update the visualizations with the new data
            for plot in self.plots:
                plot.update_plot(scale_factor)


        except ValueError:
            # If the input is not a valid float, show a warning message.
            QtWidgets.QMessageBox.warning(self, 'Invalid Input', 'Please enter a valid number.')


def main():
    # Create the Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
