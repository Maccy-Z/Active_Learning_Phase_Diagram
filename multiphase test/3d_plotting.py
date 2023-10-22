import sys
from PyQt5 import QtWidgets
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import numpy as np


def custom_plot(scene, scale):
    points = np.array([[0, 0, 0], [0, 1, 1], [scale, 1, scale]])
    plot = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=1, color=(1, 1, 0), figure=scene)
    return plot


def custom_plot2(scene, scale):
    points = np.array([[0, scale, 0], [0, 1, 1], [0, 1, 0]])
    plot = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=1, color=(0, 1, 0), figure=scene)
    plot = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], scale_factor=1, color=(0, 1, 1), figure=scene)
    return plot


def custom_plot3(scene, scale):
    scene.mlab.test_points3d()


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)  # Make the view resizable.

    def __init__(self):
        HasTraits.__init__(self)
        self.update_plot(1.0)  # Creating an initial plot

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        mlab.clf(figure=self.scene.mayavi_scene)
        self.plot = custom_plot(self.scene.mayavi_scene, scale)
        print(self.scene.mayavi_scene)


class Visualization2(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=200, show_label=False),
                resizable=True)  # Make the view resizable.

    def __init__(self):
        HasTraits.__init__(self)
        self.update_plot(1.0)  # Creating an initial plot

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        mlab.clf(figure=self.scene.mayavi_scene)
        custom_plot2(self.scene.mayavi_scene, scale)


class Visualization3(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=100, show_label=False),
                resizable=True)  # Make the view resizable.

    def __init__(self):
        HasTraits.__init__(self)
        self.update_plot(1.0)  # Creating an initial plot

    def update_plot(self, scale):
        # Check if there is an existing plot, if yes, remove it
        mlab.clf(figure=self.scene.mayavi_scene)
        custom_plot3(self.scene, scale)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Interactive Mayavi plots in PyQt')

        # Creating the main widget
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # Creating a vertical layout to arrange elements
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Create a horizontal layout for the Mayavi views
        mayavi_layout = QtWidgets.QHBoxLayout()

        # Embedding Mayavi scenes as subplots
        self.visualization1 = Visualization()
        mayavi_layout.addWidget(self.visualization1.edit_traits(parent=self, kind='subpanel').control)
        self.visualization2 = Visualization2()
        mayavi_layout.addWidget(self.visualization2.edit_traits(parent=self, kind='subpanel').control)
        self.visualization3 = Visualization3()
        mayavi_layout.addWidget(self.visualization3.edit_traits(parent=self, kind='subpanel').control)

        # Add the Mayavi layout to the main layout
        layout.addLayout(mayavi_layout)

        # Adding a textbox for data input
        self.textbox = QtWidgets.QLineEdit()
        layout.addWidget(self.textbox)

        # Connect the textbox input event to an update function
        self.textbox.returnPressed.connect(self.on_text_enter)

    def on_text_enter(self):
        # When the user presses Enter, this gets executed.
        text_value = self.textbox.text()
        try:
            # Convert the text box value to a float
            scale_factor = float(text_value)

            # Update the visualizations with the new data
            self.visualization1.update_plot(scale_factor)
            self.visualization2.update_plot(scale_factor)
            self.visualization3.update_plot(scale_factor)
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
