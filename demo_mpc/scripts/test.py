"""
Model Predictive Control Unicycle
================================

"""

from rockit import *
from casadi import *

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Problem parameters
# -------------------------------
max_vel = 0.1
min_vel = 0
max_omega = 0.5
max_accel = 1
max_omega_dot = 1

nx    = 5                   # the system is composed of 5 states
nu    = 2                   # the system has 2 inputs
Tf    = 1                   # control horizon [s]
Nhor  = 10                  # number of control intervals
dt    = Tf/Nhor             # sample time

current_X = vertcat(0.0, 0.0, 0.0, 0.0, 0.0)  # initial state

Nsim  = int(10 * Nhor / Tf)                  # how much samples to simulate

# -------------------------------
# Logging variables
# -------------------------------
x_history       = np.zeros(Nsim+1)
y_history       = np.zeros(Nsim+1)
xd_history      = np.zeros(Nsim+1)
yd_history      = np.zeros(Nsim+1)
u1_history      = np.zeros(Nsim)
u2_history      = np.zeros(Nsim)

# -------------------------------
# Set OCP
# -------------------------------
ocp = Ocp(T=Tf)

# Define states
x = ocp.state()
y = ocp.state()
theta = ocp.state()
v = ocp.state()
w = ocp.state()

# Defince controls
u1 = ocp.control()
u2 = ocp.control()

# Define parameter
X_0 = ocp.parameter(nx)

# Specify ODE
ocp.set_der(x, (v*cos(theta)))
ocp.set_der(y, (v*sin(theta)))
ocp.set_der(theta, w)
ocp.set_der(v, u1)
ocp.set_der(w, u2)

# Define a placeholder for concrete waypoints to be defined on edges of the control grid
trajectory = ocp.parameter(2, grid='control')

# Lagrange objective
ocp.add_objective(ocp.integral((x-trajectory[0])**2 + (y-trajectory[1])**2))
ocp.add_objective(ocp.at_tf((x-trajectory[0])**2 + (y-trajectory[1])**2))

# Path constraints
ocp.subject_to( (min_vel <= v) <= max_vel )
ocp.subject_to( (-max_omega <= w) <= max_omega )
ocp.subject_to( (-max_accel <= u1) <= max_accel )
ocp.subject_to( (-max_omega_dot <= u2) <= max_omega_dot )

# Initial constraints
X = vertcat(x, y, theta, v, w)
ocp.subject_to(ocp.at_t0(X)==X_0)

trajectory_N = np.zeros([2,Nhor])
x_multiplier = 0.1
y_amplitude = 0.05
y_freq = 3*pi/40
for i in range(Nhor):
    timer = i * dt
    x_d = x_multiplier * timer
    y_d = y_amplitude*sin((y_freq) * timer)
    trajectory_N[0,i] = x_d
    trajectory_N[1,i] = y_d

ocp.set_value(trajectory, trajectory_N)

# Pick a solution method
options = {"ipopt": {"print_level": 5, "tol":1e-3}}
options["expand"] = True
options["print_time"] = False
ocp.solver('ipopt',options)

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

# -------------------------------
# Solve the OCP wrt a parameter value (for the first time)
# -------------------------------
# Set initial value for parameters
ocp.set_value(X_0, current_X)
# Solve
sol = ocp.solve()

# Get discretisd dynamics as CasADi function
#Sim_pendulum_dyn = ocp._method.discrete_system(ocp)
Sim_asv_dyn = ocp._method.discrete_system(ocp)

# Log data for post-processing
x_history[0]   = current_X[0]
y_history[0] = current_X[1]
xd_history[0]   = trajectory_N[0,0]
yd_history[0] = trajectory_N[1,0]

# -------------------------------
# Simulate the MPC solving the OCP (with the updated state) several times
# -------------------------------

for i in range(Nsim):
    print("timestep", i+1, "of", Nsim)
    # Get the solution from sol
    tsa, f1sol = sol.sample(u1, grid='control')
    _, f2sol = sol.sample(u2, grid='control')
    # Simulate dynamics (applying the first control input) and update the current state
    current_X = Sim_asv_dyn(x0=current_X, u=vertcat(f1sol[0],f2sol[0]), T=dt)["xf"]
    # Set the parameter X0 to the new current_X
    ocp.set_value(X_0, current_X[:nx])
    # Set the new trajectory
    for j in range(Nhor):
        timer = ((j+1)*dt) + (i*dt)
        x_d = x_multiplier * timer
        y_d = y_amplitude*sin((y_freq) * timer)
        trajectory_N[0,j] = x_d
        trajectory_N[1,j] = y_d
    ocp.set_value(trajectory, trajectory_N)
    # Solve the optimization problem
    sol = ocp.solve()
    ocp._method.opti.set_initial(ocp._method.opti.x, ocp._method.opti.value(ocp._method.opti.x))

    # Log data for post-processing
    x_history[i+1]   = current_X[0].full()
    y_history[i+1]   = current_X[1].full()
    u1_history[i]    = f1sol[0]
    u2_history[i]    = f2sol[0]
    xd_history[i+1]  = trajectory_N[0,0]
    yd_history[i+1]  = trajectory_N[1,0]

# -------------------------------
# Plot the results
# -------------------------------
time_sim = np.linspace(0, dt*Nsim, Nsim+1)
time_sim2 = np.linspace(0, dt*Nsim, Nsim)

fig2, ax3 = plt.subplots()
ax3.plot(yd_history, xd_history, 'r-')
ax3.plot(y_history, x_history, 'b--')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('X [m]')

fig3, ax4 = plt.subplots()
ax4.plot(time_sim2, f1_history, 'r-')
ax4.plot(time_sim2, f2_history, 'b-')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('f [N]')
fig3.tight_layout()

plt.show()