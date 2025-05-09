import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from solvers import rk4, adams_bashforth

# Constants
g = 9.81
L = 1.0

def spherical_pendulum_odes(t, y):
    theta, phi, theta_dot, phi_dot = y
    dtheta_dt = theta_dot
    dphi_dt = phi_dot
    dtheta_dot_dt = np.sin(theta) * np.cos(theta) * phi_dot**2 - (g / L) * np.sin(theta)
    dphi_dot_dt = 0.0 if np.abs(np.sin(theta)) < 1e-6 else -2 * theta_dot * phi_dot / np.tan(theta)
    return np.array([dtheta_dt, dphi_dt, dtheta_dot_dt, dphi_dot_dt])

# Time settings
t0 = 0
tfinal = 10
step_size = .01
n_steps = int((tfinal - t0) / step_size)
dt = (tfinal - t0) / n_steps
time_array = np.linspace(t0, tfinal, n_steps + 1)

# Initial condition
# y0 = [theta, phi, theta^dot, phi^dot]
y0 = [np.pi / 3, 0, 0, 1]

# Solve using rk4
t_vals_rk, sol_rk = rk4(spherical_pendulum_odes, n_steps, y0, t0, tfinal)
# Solve using adams bashforth
t_vals_ab, sol_ab = adams_bashforth(spherical_pendulum_odes, n_steps, y0, t0, tfinal)

# Extract Cartesian coordinates
def to_cartesian(theta, phi):
    x = L * np.sin(theta) * np.cos(phi)
    y = L * np.sin(theta) * np.sin(phi)
    z = -L * np.cos(theta)
    return x, y, z

x_rk, y_rk, z_rk = to_cartesian(sol_rk[:, 0], sol_rk[:, 1])
x_ab, y_ab, z_ab = to_cartesian(sol_ab[:, 0], sol_ab[:, 1])

# RK4 plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
bob_rk, = ax1.plot([], [], [], 'o-', lw=2, label='Bob')
trail_rk, = ax1.plot([], [], [], 'r-', lw=1, alpha=0.7, label='Trail')

ax1.set_xlim(-L, L)
ax1.set_ylim(-L, L)
ax1.set_zlim(-L, 0)
ax1.set_title('Spherical Pendulum RK4')

def init_rk():
    bob_rk.set_data([], [])
    bob_rk.set_3d_properties([])
    trail_rk.set_data([], [])
    trail_rk.set_3d_properties([])
    return bob_rk, trail_rk

def update_rk(i):
    bob_rk.set_data([0, x_rk[i]], [0, y_rk[i]])
    bob_rk.set_3d_properties([0, z_rk[i]])
    trail_rk.set_data(x_rk[:i+1], y_rk[:i+1])
    trail_rk.set_3d_properties(z_rk[:i+1])
    return bob_rk, trail_rk

ani_rk = FuncAnimation(fig1, update_rk, frames=len(t_vals_rk), init_func=init_rk, blit=True, interval=dt*1000)

# Adams-Bashforth plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
bob_ab, = ax2.plot([], [], [], 'o-', lw=2, label='Bob')
trail_ab, = ax2.plot([], [], [], 'g-', lw=1, alpha=0.7, label='Trail')

ax2.set_xlim(-L, L)
ax2.set_ylim(-L, L)
ax2.set_zlim(-L, 0)
ax2.set_title('Spherical Pendulum Adams-Bashforth')

def init_ab():
    bob_ab.set_data([], [])
    bob_ab.set_3d_properties([])
    trail_ab.set_data([], [])
    trail_ab.set_3d_properties([])
    return bob_ab, trail_ab

def update_ab(i):
    bob_ab.set_data([0, x_ab[i]], [0, y_ab[i]])
    bob_ab.set_3d_properties([0, z_ab[i]])
    trail_ab.set_data(x_ab[:i+1], y_ab[:i+1])
    trail_ab.set_3d_properties(z_ab[:i+1])
    return bob_ab, trail_ab

ani_ab = FuncAnimation(fig2, update_ab, frames=len(t_vals_ab), init_func=init_ab, blit=True, interval=dt*1000)

plt.show()
