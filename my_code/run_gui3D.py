from DistanceSampler3D import DistanceSampler3D
from utils import Config

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QDialog
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from mayavi import mlab
from mayavi.core.api import Engine
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item

import sys
import numpy as np


class Visualization(HasTraits):
    engine = Instance(Engine, args=())
    scene = Instance(MlabSceneModel)
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True)  # Make the view resizable.

    def _scene_default(self):
        self.engine.start()
        return MlabSceneModel(engine=self.engine)

    def clear_plot(self):
        mlab.clf(figure=self.scene.mayavi_scene)


class MainWindow(QMainWindow):

    def __init__(self, gui_to_sampler: DistanceSampler3D, cfg, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.cfg = cfg

        # Make the plots
        self.plots = []
        self.plots.append(Visualization())
        self.plots.append(Visualization())
        self.plots.append(Visualization())

        self.gui_to_sampler = gui_to_sampler
        self.gui_to_sampler.set_scenes(self.plots)

        plot_bar = QHBoxLayout()
        for plot in self.plots:
            plot_bar.addWidget(plot.edit_traits(parent=self, kind='subpanel').control)

        # Create the informational label
        self.infoLabel = QLabel("Enter your new observation point and phase")
        self.dynamicLabel = QLabel("Initial text goes here.")

        # Create the labels and text boxes
        self.label1 = QLabel("Phase:")
        self.text_phase = QLineEdit(self)
        self.text_phase.setValidator(QIntValidator())

        self.label_x = QLabel("x:")
        self.text_x = QLineEdit(self)
        self.text_x.setValidator(QDoubleValidator())
        self.label_y = QLabel("y:")
        self.text_y = QLineEdit(self)
        self.text_y.setValidator(QDoubleValidator())
        self.label_z = QLabel("z:")
        self.text_z = QLineEdit(self)
        self.text_z.setValidator(QDoubleValidator())

        # Layout for the text boxes and labels
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.label1)
        hbox1.addWidget(self.text_phase)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.label_x)
        hbox2.addWidget(self.text_x)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.label_y)
        hbox3.addWidget(self.text_y)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.label_z)
        hbox4.addWidget(self.text_z)

        # Add a button to update the plot
        self.button = QPushButton("Update Plot")
        self.button.clicked.connect(self.next_point)

        # Join everything together
        layout = QVBoxLayout()
        layout.addLayout(plot_bar)
        layout.addWidget(self.dynamicLabel)  # Add the dynamic label below the plot
        layout.addWidget(self.infoLabel)
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addLayout(hbox4)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.sample_and_plot()
        self.show()

    def next_point(self):
        print()
        self.dynamicLabel.setText("Calculating ... ")
        self.dynamicLabel.setStyleSheet("color: red;")
        self.dynamicLabel.repaint()

        # Get next observation
        phase = int(self.text_phase.text())
        x = float(self.text_x.text().replace(',', '.'))
        y = float(self.text_y.text().replace(',', '.'))
        z = float(self.text_z.text().replace(',', '.'))

        if phase > self.cfg.N_phases - 1 or phase < 0:
            self.dynamicLabel.setText(f'Invalid phase entered. Maximum phase is {self.cfg.N_phases - 1}')
            self.dynamicLabel.repaint()
            return

        # Add to model
        self.gui_to_sampler.add_obs(phase, np.array([x, y, z]))
        # Sample next point
        new_point, prob_at_point = self.sample_and_plot()

        self.infoLabel.setText(f'Predicted probabilities: {prob_at_point}')
        self.dynamicLabel.setStyleSheet("color: black;")

    def sample_and_plot(self):
        self.dynamicLabel.setText("Calculating ... ")
        self.dynamicLabel.setStyleSheet("color: red;")
        self.dynamicLabel.repaint()

        new_point, prob_at_point = self.gui_to_sampler.single_obs()

        x, y, z = new_point
        self.text_x.setText(f'{x:.4g}')
        self.text_y.setText(f'{y:.4g}')
        self.text_z.setText(f'{z:.4g}')

        new_text = "Done"
        self.dynamicLabel.setText(new_text)
        self.infoLabel.setText(f'Predicted probabilities: {prob_at_point}')

        return new_point, prob_at_point


class InputWindow(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super(InputWindow, self).__init__(parent)
        self.cfg = cfg
        assert cfg.N_dim == 3, "Only works for 3D"

        self.N_phases = cfg.N_phases
        self.N_dim = cfg.N_dim

        # GUI Elements
        self.layout = QVBoxLayout(self)

        self.instruction_box = QLabel("Enter initial measurements, each seperated by a comma (,)", self)

        # First Input Box
        self.label1 = QLabel("Phases:", self)
        self.input_phase = QLineEdit(self)

        # # Second Input Box
        self.labels, self.inputs = [], []
        for dim in range(self.N_dim):
            self.labels.append(QLabel(f'Coordinate {dim + 1}'))
            self.inputs.append(QLineEdit(self))

        self.okButton = QPushButton("OK", self)

        # Add elements to the layout
        self.layout.addWidget(self.instruction_box)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.input_phase)

        for lab, inp in zip(self.labels, self.inputs):
            self.layout.addWidget(lab)
            self.layout.addWidget(inp)

        self.layout.addWidget(self.okButton)

        self.okButton.clicked.connect(self.check_input)

    def check_input(self):
        self.instruction_box.setStyleSheet("color: red;")

        phases = self.input_phase.text()
        init_coords = []
        for input_box in self.inputs:
            init_coords.append(input_box.text())

        try:
            phases = [int(p.strip()) for p in phases.split(',')]
        except ValueError as e:
            print()
            print("\033[31mInvalid phase\033[0m")
            print(e)
            self.instruction_box.setText("Invalid phase entered")
            return

        if not np.all(np.isin(phases, range(self.N_phases))):
            print()
            print(f"\033[31mPhases out of range, must be in {list(range(self.N_phases))}\033[0m")
            self.instruction_box.setText("Phase out of allowed number of phases")
            return

        clean_coords = []
        for dim_no, init_coord in enumerate(init_coords):
            try:
                clean_coord = [float(x.strip()) for x in init_coord.split(',')]
            except ValueError as e:
                print()
                print(f"\033[31mInvalid {dim_no + 1} coordinates\033[0m")
                print(e)
                self.instruction_box.setText(f"Invalid {dim_no + 1} coordinates entered")
                return

            clean_coords.append(clean_coord)

        # Check all inputs are the same length
        phase_len = len(phases)
        if not all([len(coord) == phase_len for coord in clean_coords]):
            print("\033[31mNumber of entries must be equal for phase and all coordinates\033[0m")
            print(f'{len(phases) = }')
            for dim, coord in enumerate(clean_coords):
                print(f'len dimension {dim + 1} = {len(coord)}')
            self.instruction_box.setText("Number of observations is inconsistent")
            return

        self.phases = np.array(phases)
        self.Xs = np.array(clean_coords).T

        self.accept()


def initial_obs(cfg):
    initialDialog = InputWindow(cfg)
    dialog_result = initialDialog.exec()

    if dialog_result == QDialog.DialogCode.Rejected:
        print("\033[31mNothing entered, exiting\033[0m")
        sys.exit()

    print(initialDialog.phases, initialDialog.Xs)
    return initialDialog.phases, initialDialog.Xs

    # phases = np.array([0, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 2, 2, 0, 0, 0, 1, 2, 2])
    #
    # obs_pos = np.array([[0., 0., 0.],
    #                     [0., 1., 0.],
    #                     [1., 0.5, 1.],
    #                     [0., 0.5, 0.83333333],
    #                     [1., 0.83333333, 0.],
    #                     [0.83333333, 0., 0.],
    #                     [0., 0.16666667, 0.16666667],
    #                     [0.16666667, 0., 0.83333333],
    #                     [0.5, 1., 1.],
    #                     [0.5, 0.16666667, 1.],
    #                     [0.5, 0.83333333, 0.16666667],
    #                     [1., 1., 1.],
    #                     [1., 0., 0.83333333],
    #                     [1., 1., 0.83333333],
    #                     [0.16666667, 0.33333333, 0.],
    #                     [0.5, 0.33333333, 0.5],
    #                     [0.5, 0.5, 0.],
    #                     [0.66666667, 1., 0.],
    #                     [0.83333333, 0.83333333, 1.],
    #                     [0.83333333, 0., 0.83333333]])
    # return phases, obs_pos


if __name__ == "__main__":
    app = QApplication(sys.argv)

    cfg = Config()

    phases, Xs = initial_obs(cfg)

    passer = DistanceSampler3D(init_phases=phases, init_Xs=Xs, cfg=cfg)
    window = MainWindow(passer, cfg=cfg)

    # app.exec()
    sys.exit(app.exec_())
