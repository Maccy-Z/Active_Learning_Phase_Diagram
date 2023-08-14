import PyQt6
print(PyQt6)
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes as Axes

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget, QPushButton, QDialog
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QLineEdit, QLabel, QHBoxLayout
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from Distance_Sampler import suggest_point
from utils import make_grid, Config, ObsHolder

class GUIToSampler:
    def __init__(self, init_phases, init_Xs):
        self.cfg = Config()
        self.obs_holder = ObsHolder(self.cfg)

        # X_init, _, _ = make_grid(self.cfg.N_init)
        # for xs in X_init:
        #     self.obs_holder.make_obs(xs)

        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)


    # Load in axes for plotting
    def set_plots(self, axs):
        self.p_obs_ax: Axes = axs[0]
        self.acq_ax: Axes = axs[1]
        self.pd_ax: Axes = axs[2]

    def plot(self, new_point, pd_old, avg_dists, max_probs):
        self.p_obs_ax.clear()
        self.acq_ax.clear()
        self.pd_ax.clear()

        # Reshape 1d array to 2d
        side_length = int(np.sqrt(len(pd_old)))
        pd_old = pd_old.reshape((side_length, side_length))
        side_length = int(np.sqrt(len(avg_dists)))
        avg_dists = avg_dists.reshape((side_length, side_length))
        side_length = int(np.sqrt(len(max_probs)))
        max_probs = max_probs.reshape((side_length, side_length))

        # Plot probability of observaion
        self.p_obs_ax.set_title("P(obs)")
        self.p_obs_ax.imshow(1 - max_probs, extent=(-2, 2, -2, 2),
                   origin="lower", vmax=1, vmin=0)

        # Plot acquisition function
        self.acq_ax.set_title(f'Acq. fn')
        sec_low = np.sort(np.unique(avg_dists))[1]
        self.acq_ax.imshow(avg_dists, extent=(-2, 2, -2, 2),
                   origin="lower", vmin=sec_low)  # Phase diagram

        # Plot current phase diagram and next sample point
        X_obs, phase_obs = self.obs_holder.get_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        self.pd_ax.set_title(f"PD and points")
        self.pd_ax.imshow(pd_old, extent=(-2, 2, -2, 2), origin="lower")  # Phase diagram
        self.pd_ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')  # Existing observations
        self.pd_ax.scatter(new_point[0], new_point[1], s=80, c='tab:orange')  # New observations
        # plt.colorbar()

    def get_input(self, phase:int, X:np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)

        new_point, (pd_old, avg_dists, max_probs) = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, max_probs)

        return new_point

    def init_plot(self):
        new_point, (pd_old, avg_dists, max_probs) = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, max_probs)

        return new_point


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)

        # Create three subplots
        self.axes1 = fig.add_subplot(131)
        self.axes2 = fig.add_subplot(132)
        self.axes3 = fig.add_subplot(133)


        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


    def get_axes(self):
        return self.axes1, self.axes2, self.axes3


class MainWindow(QMainWindow):

    def __init__(self, gui_to_sampler: GUIToSampler, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.gui_to_sampler = gui_to_sampler
        self.canvas = MplCanvas(self, width=8, height=4, dpi=100)
        self.gui_to_sampler.set_plots(self.canvas.get_axes())
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


        # Add a button to update the plot
        self.button = QPushButton("Update Plot")
        self.button.clicked.connect(self.update_plot)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.dynamicLabel)  # Add the dynamic label below the plot
        layout.addWidget(self.infoLabel)
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.initial_plot()
        self.show()

    def update_plot(self):
        self.dynamicLabel.setText("Calculating ... ")
        self.dynamicLabel.setStyleSheet("color: red;")
        self.dynamicLabel.repaint()

        phase = int(self.text_phase.text())
        x = float(self.text_x.text().replace(',', '.'))
        y = float(self.text_y.text().replace(',', '.'))

        # Draw updated plots

        new_point = self.gui_to_sampler.get_input(phase, np.array([x, y]))
        self.canvas.draw()

        x, y = new_point
        self.text_x.setText(str(x))
        self.text_y.setText(str(y))

        new_text = "Done"
        self.dynamicLabel.setText(new_text)
        self.dynamicLabel.setStyleSheet("color: black;")

    def initial_plot(self):
        self.dynamicLabel.setText("Initialising ... ")
        self.dynamicLabel.repaint()

        new_point = self.gui_to_sampler.init_plot()
        self.canvas.draw()

        x, y = new_point
        self.text_x.setText(str(x))
        self.text_y.setText(str(y))

        new_text = "Done"
        self.dynamicLabel.setText(new_text)
        self.dynamicLabel.repaint()


class InputWindow(QDialog):
    def __init__(self, parent=None):
        super(InputWindow, self).__init__(parent)

        # GUI Elements
        self.layout = QVBoxLayout(self)

        self.instruction_box = QLabel("Enter initial measurements, each seperated by a comma (,)", self)

        # First Input Box
        self.label1 = QLabel("Phases:", self)
        self.input_phase = QLineEdit(self)
        self.input_phase.setText("Hi")

        # Second Input Box
        self.label2 = QLabel("X coordinates:", self)
        self.input_x = QLineEdit(self)

        # Third Input Box
        self.label3 = QLabel("Y coordinates:", self)
        self.input_y = QLineEdit(self)

        self.okButton = QPushButton("OK", self)

        # Add elements to the layout
        self.layout.addWidget(self.instruction_box)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.input_phase)

        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.input_x)

        self.layout.addWidget(self.label3)
        self.layout.addWidget(self.input_y)

        self.layout.addWidget(self.okButton)

        self.okButton.clicked.connect(self.check_input)

    def check_input(self):
        phases = self.input_phase.text()
        xs = self.input_x.text()
        ys = self.input_y.text()

        try:
            phases = [int(p.strip()) for p in phases.split(',')]
        except ValueError as e:
            print()
            print("\033[31mInvalid phase\033[0m")
            print(e)
            self.instruction_box.setText("Invalid phase entered")
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

        if not len(xs) == len(phases) == len(ys):
            print("\033[31mNumber of entries must be equal for xs, ys, and phase\033[0m")
            print(f'{len(phases) = }, {len(xs) = }, {len(ys) = }')
            self.instruction_box.setText("Number of observations is inconsistent")
            return

        self.phases = np.array(phases)
        self.Xs = np.stack([xs, ys]).T
        self.accept()



def initial_obs():
    initialDialog = InputWindow()
    dialog_result = initialDialog.exec()

    if dialog_result == QDialog.rejected:
        sys.exit()

    return initialDialog.phases, initialDialog.Xs

if __name__ == "__main__":
    app = QApplication(sys.argv)
    phases, Xs = initial_obs()

    passer = GUIToSampler(init_phases=phases, init_Xs=Xs)
    window = MainWindow(passer)

    app.exec()
    sys.exit(app.exec())
