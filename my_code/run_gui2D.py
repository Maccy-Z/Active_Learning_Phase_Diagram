from DistanceSampler2D import DistanceSampler2D
from utils import Config

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget, QPushButton, QDialog
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=15, height=5, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)

        # Create three subplots
        self.axes1 = self.fig.add_subplot(131)
        self.axes2 = self.fig.add_subplot(132)
        self.axes3 = self.fig.add_subplot(133)
        self.fig.tight_layout()

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

    def get_axes(self):
        return self.axes1, self.axes2, self.axes3,


class MainWindow(QMainWindow):

    def __init__(self, gui_to_sampler: DistanceSampler2D, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.gui_to_sampler = gui_to_sampler
        self.canvas = MplCanvas(self, width=12, height=5, dpi=100)
        self.gui_to_sampler.set_plots(self.canvas.get_axes(), self.canvas.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create the informational label
        self.infoLabel = QLabel("Enter your new observation point and phase")
        self.dynamicLabel = QLabel("Initial text goes here.")

        # Create the labels and text boxes
        self.label1 = QLabel("Phase:")
        self.text_phase = QLineEdit(self)
        self.text_phase.setValidator(QIntValidator())

        self.label2 = QLabel("x:")
        self.text_x = QLineEdit(self)
        self.text_x.setValidator(QDoubleValidator())

        self.label3 = QLabel("y:")
        self.text_y = QLineEdit(self)
        self.text_y.setValidator(QDoubleValidator())

        self.label_prob = QLabel("Measurement probability (can be empty)")
        self.text_prob = QLineEdit(self)
        self.text_prob.setValidator(QDoubleValidator())

        # Layout for the text boxes and labels
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.label1)
        hbox1.addWidget(self.text_phase)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.label2)
        hbox2.addWidget(self.text_x)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.label3)
        hbox3.addWidget(self.text_y)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.label_prob)
        hbox4.addWidget(self.text_prob)

        # Add a button to update the plot
        self.button = QPushButton("Update Plot")
        self.button.clicked.connect(self.update_plot)

        # Fill in layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
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

    def update_plot(self):
        print()
        phase = int(self.text_phase.text())
        x = float(self.text_x.text().replace(',', '.'))
        y = float(self.text_y.text().replace(',', '.'))
        if self.text_prob.text() == "":
            prob = None
        else:
            prob = float(self.text_prob.text().replace(',', '.'))
            if prob <= 0 or prob >= 1:
                self.dynamicLabel.setText("Probability must be between 0 and 1")
                self.dynamicLabel.setStyleSheet("color: red;")
                return

        # Add to model
        self.gui_to_sampler.add_obs(phase, np.array([x, y]), prob=prob)
        # Sample next point
        new_point, prob_at_point = self.sample_and_plot()

        # Update info for user
        self.infoLabel.setText(f'Predicted probabilities: {prob_at_point}')
        self.dynamicLabel.setStyleSheet("color: black;")

    def sample_and_plot(self):
        self.dynamicLabel.setText("Calculating ... ")
        self.dynamicLabel.setStyleSheet("color: red;")
        self.dynamicLabel.repaint()

        new_point, prob_at_point = self.gui_to_sampler.single_obs()
        self.canvas.draw()

        x, y = new_point
        self.text_x.setText(f'{x:.2f}')
        self.text_y.setText(f'{y:.2f}')

        new_text = "Done"
        self.dynamicLabel.setText(new_text)
        self.dynamicLabel.repaint()

        return new_point, prob_at_point


class InputWindow(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super(InputWindow, self).__init__(parent)
        self.cfg = cfg
        assert cfg.N_dim == 2, "Only works for 2D"

        self.N_phases = cfg.N_phases

        # GUI Elements
        self.layout = QVBoxLayout(self)

        self.instruction_box = QLabel("Enter initial measurements, each seperated by a comma (,)", self)

        # First Input Box
        self.label1 = QLabel("Phases:", self)
        self.input_phase = QLineEdit(self)

        # Second Input Box
        self.label2 = QLabel("X coordinates:", self)
        self.input_x = QLineEdit(self)

        # Third Input Box
        self.label3 = QLabel("Y coordinates:", self)
        self.input_y = QLineEdit(self)

        # Probability of error
        self.label4 = QLabel("Measurement prob (can be empty):", self)
        self.input_prob = QLineEdit(self)

        # OK button
        self.okButton = QPushButton("OK", self)

        # Add elements to the layout
        self.layout.addWidget(self.instruction_box)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.input_phase)

        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.input_x)

        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.input_y)

        self.layout.addWidget(self.label4)
        self.layout.addWidget(self.input_prob)

        self.layout.addWidget(self.okButton)

        self.okButton.clicked.connect(self.check_input)

    def check_input(self):
        self.instruction_box.setStyleSheet("color: red;")

        phases = self.input_phase.text()
        xs = self.input_x.text()
        ys = self.input_y.text()
        probs = self.input_prob.text()

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

        try:
            xs = [float(x.strip()) for x in xs.split(',')]
        except ValueError as e:
            print()
            print("\033[31mInvalid x coordinates\033[0m")
            print(e)
            self.instruction_box.setText("Invalid X coordinates entered")
            return

        try:
            ys = [float(x.strip()) for x in ys.split(',')]
        except ValueError as e:
            print()
            print("\033[31mInvalid y coordinates\033[0m")
            print(e)
            self.instruction_box.setText("Invalid Y coordinates entered")
            return

        # Probability can be left blank and must be between 0 and 1
        try:
            if probs == '':
                probs = None
            else:
                probs = [float(x.strip()) for x in probs.split(',')]
                p = np.array(probs)
                if np.any(p <= 0) or np.any(p >= 1):
                    raise ValueError("Probabilities must be between 0 and 1")
        except ValueError as e:
            print()
            print("\033[31mInvalid probabilities\033[0m")
            print(e)
            self.instruction_box.setText("Invalid probabilities entered")
            return

        if not len(xs) == len(phases) == len(ys):
            print("\033[31mNumber of entries must be equal for xs, ys, and phase\033[0m")
            print(f'{len(phases) = }, {len(xs) = }, {len(ys) = }')
            self.instruction_box.setText("Number of observations is inconsistent")
            return

        if probs is not None and not len(probs) == len(phases):
            print("\033[31mNumber of entries must be equal for phases and probs\033[0m")
            print(f'{len(phases) = }, {len(probs) = }')
            self.instruction_box.setText("Number of probabilities is inconsistent")
            return

        self.phases = np.array(phases)
        self.Xs = np.stack([xs, ys]).T
        self.probs = None if probs is None else np.array(probs)
        self.accept()


def initial_obs(cfg):
    initialDialog = InputWindow(cfg)
    dialog_result = initialDialog.exec()

    if dialog_result == QDialog.DialogCode.Rejected:
        print("\033[31mNothing entered, exiting\033[0m")
        sys.exit()

    return initialDialog.phases, initialDialog.Xs, initialDialog.probs


if __name__ == "__main__":
    app = QApplication(sys.argv)

    cfg = Config()

    phases, Xs, probs = initial_obs(cfg)

    passer = DistanceSampler2D(init_phases=phases, init_Xs=Xs, init_probs=probs, cfg=cfg)
    window = MainWindow(passer)

    # app.exec()
    sys.exit(app.exec())
