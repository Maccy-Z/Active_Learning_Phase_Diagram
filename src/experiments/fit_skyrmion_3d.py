import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))
print(os.getcwd())

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QDialog
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import QTimer

import numpy as np

from pd import skyrmion_pd_3D
from src.DistanceSampler3D import DistanceSampler3D, Visualization
from src.utils import synth_3d_pd


class MainWindow(QMainWindow):
    def __init__(self, gui_to_sampler: DistanceSampler3D, cfg, pd_fn):
        super(MainWindow, self).__init__()
        self.cfg = cfg
        self.steps = 0
        self.pd_fn = pd_fn

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

        # Join everything together
        layout = QVBoxLayout()
        layout.addLayout(plot_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # self.sample_and_plot()

        self.show()
        # Set up a timer to call the update function every 3 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample_and_plot)
        self.timer.start(100)  # 3000 milliseconds = 3 seconds

    def sample_and_plot(self):
        print()
        print("Next point")
        new_point, prob_at_point = self.gui_to_sampler.single_obs()
        new_phase = self.pd_fn(new_point)
        print(new_point, new_phase, f'{prob_at_point = }')
        self.gui_to_sampler.add_obs(new_phase, np.array(new_point), prob=0.95)

        self.steps += 1
        if self.steps >= 500:
            self.timer.stop()  # Stop the timer
            QApplication.quit()  # Exit the application


if __name__ == "__main__":
    from src.config import Config
    pd_fn = skyrmion_pd_3D

    save_dir = "../saves"

    print(save_dir)
    cfg = Config()

    X_init = [[0.0, 0.4, 0.5, 0.6, 0, 0.7], [0, 0.1, 0, 0.5, 0, 0.8], [0.1, 0.1, 0.5, 0.3, 0.4, 1]]
    X_init = np.array(X_init).T
    phase_init = [pd_fn(x) for x in X_init]


    app = QApplication(sys.argv)
    distance_sampler = DistanceSampler3D(phase_init, X_init, cfg, save_dir=save_dir)

    window = MainWindow(distance_sampler, cfg, pd_fn=pd_fn)
    sys.exit(app.exec_())
