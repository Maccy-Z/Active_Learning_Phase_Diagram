import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

with open("./allthereeDPhaseIndex.npy", "rb") as f:
    data = np.load(f)

x, y, z = np.indices(data.shape)

# Flatten everything to 1D arrays
x = x#.flatten()
y = y#.flatten()
z = z.flatten()
values = data.flatten()  # Scalar values

d = data[:, :, 0].flatten()
x = x[:, :, 0].flatten()
y = y[:, :, 0].flatten()

plt.imshow(data[:, :, 1])
plt.show()
# print(data.shape)
#
# # Create the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Scatter plot: mapping 'values' to the colormap 'viridis'
# scat = ax.scatter(x, y, z, c=values, cmap='viridis')
#
# # Adding a color bar to indicate the scale of scalar values
# # plt.colorbar(scat, label='Scalar values')
#
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Show plot
# plt.show()

