#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MTHB5010A – Maths for Scientists B
# Further Mathematical Methods for Science and Engineering: Python Coursework
# 2025
import math # maths content like pi, sin, cos, sqrt, etc.
import numpy as np # for arrays, matrices, linspace, random numbers, etc
import matplotlib.pyplot as plt # for plotting graphs
from matplotlib import cm # colour maps for 3D surface plots
###########################################
# Question 1(a): surface plots for f(x,y)
###########################################
def surface_data(n, x_min=0.0, x_max=10 * math.pi,
y_min=-math.pi, y_max=math.pi,
num_x=200, num_y=200):
"""
Build the X, Y, Z arrays for f(x,y) = e^{-n y} sin(n x)
over the chosen rectangle in the (x,y) plane.
"""
# x and y are 1D lists of evenly spaced points
x = np.linspace(x_min, x_max, num_x)
y = np.linspace(y_min, y_max, num_y)
# meshgrid takes those 1D lists and turns them into a full 2D grid
X, Y = np.meshgrid(x, y)
# now evaluating the formula at every point on the grid
Z = np.exp(-n * Y) * np.sin(n * X)
# sends back all three so we can plot the surface
return X, Y, Z
def plot_q1a():
"""
 Make 3D surface plots of f(x,y) = e^{-n y} sin(n x) for a few n values.
"""
# these are the different n values we want to compare
ns = [1, 2, 3, 4]
# create a larger figure with space for 4 subplots (2x2 grid)
fig = plt.figure(figsize=(12, 10))
# go through each n, and each one gets its own 3D plot
for idx, n in enumerate(ns, start=1):
ax = fig.add_subplot(2, 2, idx, projection='3d')
X, Y, Z = surface_data(n)
ax.plot_surface(X, Y, Z, cmap=cm.magma)
ax.set_title(f"f(x,y) for n = {n}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
# overall title for the whole figure
fig.suptitle("Question 1(a): Surface plots for different n", fontsize=14)
# tidy up spacing so labels don’t sit on top of each other
plt.tight_layout()
plt.show()
###########################################
# Question 1(b): Euler method for ODE
###########################################
def euler_method(y0, h=0.1, T=10):
"""Euler solution of dy/dt = y(t-1) - y - 1/4 sin(2*pi*t)."""
# build the time axis from 0 to T in steps of size h
t_values = np.arange(0, T + h, h)
# create an array to store the solution y(t) at each time
y_values = np.zeros(len(t_values))
# set the starting value y(0) = y0
y_values[0] = y0
# step forward in time using Euler's method
for i in range(1, len(t_values)):
# current time and current y value
t = t_values[i - 1]
y = y_values[i - 1]
# this is the right-hand side of the ODE (the slope dy/dt)
dydt = (np.sin(2 * t)) - y # matches the simplified DE we’re using
# Euler update: new y = old y + step * slope
y_values[i] = y + h * dydt

# give back the full time array and the corresponding y values
return t_values, y_values
def plot_euler_solutions():
"""
 Quick test plot for the Euler method with a few starting values.
Looks like “me” checking the method.
"""
y0_values = [0, 1, -1] # try a few different y(0) values
plt.figure()
for y0 in y0_values:
t, y = euler_method(y0)
plt.plot(t, y, label=f"y(0) = {y0}")
plt.title("Euler Method Solutions")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()
# this call is just a test run so I can see the Euler method working
plot_euler_solutions()
###########################################
# Question 1(c): series approximation of pi
###########################################
def approx_pi_series(tol):
"""
 Use the series
s_M = sqrt(12) * sum_{k=0}^M (-3)^(-k) / (2k+1)
to approximate pi, and keep adding terms until the error is < tol.
"""
s_partial = 0.0 # running sum of the series
k = 0 # current index k
M_list = [] # to store each k we use
err_list = [] # to store the error at each step
error = float("inf") # start with “huge” error so the loop runs
# keep going while the approximation is not accurate enough
while error >= tol:
# compute the next term in the series
term = (-3.0) ** (-k) / (2 * k + 1)
# add it to the running sum
s_partial += term
# turn the sum into a pi-approximation
approx = math.sqrt(12.0) * s_partial
# measure how far we are from the real pi
error = abs(approx - math.pi)
# record this k and the error so we can plot later
M_list.append(k)
err_list.append(error)
# move on to the next term
k += 1
# i incremented k one extra time at the end, so the last true M is k-1
return approx, k - 1, M_list, err_list
def plot_q1c():
"""
 Run the pi series with tol = 1e-5 and plot how the error behaves as we add terms.
"""
tol = 0.00001
approx, M, M_list, err_list = approx_pi_series(tol)
plt.figure(figsize=(10, 6))
# semilogy = normal x-axis, log-scale y-axis (helps see small errors better)
plt.semilogy(M_list, err_list, 'b-', linewidth=2)
plt.xlabel("M (number of terms)")
plt.ylabel(r"$|s_M - \pi|$")
plt.title(
f"Question 1(c): Convergence of series to pi\n(tol = {tol}, final M = {M}, approx = {approx:.6f})"
)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

###########################################
# Question 1(d): special upper-triangular matrix
###########################################
def special_matrix(n):
"""
Build an n x n matrix A where:
- everything below the main diagonal is 0
- row i is filled with i (from the diagonal to the right) if i is even
and stays 0 if i is odd.
"""
# start with an all-zero n x n matrix
A = np.zeros((n, n))
# go through each row (Python rows are 0..n-1)
for i in range(n):
# convert to 1,2,3,... so it matches the maths description
row_index = i + 1
# only even-numbered rows get non-zero entries
if row_index % 2 == 0:
# fill from the diagonal (j=i) to the end of the row
for j in range(i, n):
A[i, j] = row_index
return A
###########################################
# Question 1(e): random 0/1 matrix from U(0,1)
###########################################
def random_binary_matrix(n, m):
"""
 Make an n x m matrix of random numbers in [0,1),
then turn anything >= 0.5 into 1 and the rest into 0.
Basically: flip a “coin” for each entry.
"""
# random matrix with values between 0 and 1
A = np.random.random((n, m))
# threshold at 0.5: True 1, False 0
return np.where(A >= 0.5, 1, 0)
###########################################
# Question 2: RK4 for dy/dt + p*t*y = cos(π*t)
###########################################
def rk4(f, t0, y0, t_end, steps):
"""
Classic 4th-order Runge–Kutta stepper for y' = f(t,y).
This is like a “fancy Euler” that samples the slope 4 times per step.
"""
# build the time grid and get the step size h at the same time
t, h = np.linspace(t0, t_end, steps + 1, retstep=True)
# array to hold the solution y(t)
y = np.zeros_like(t)
y[0] = y0
# move along the time grid and update y with RK4
for i in range(steps):
# k1 = slope at the beginning
k1 = f(t[i], y[i])
# k2 = slope in the middle (using k1)
k2 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
# k3 = another middle slope (using k2)
k3 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
# k4 = slope at the end (using k3)
k4 = f(t[i] + h, y[i] + h * k3)
# combine them with weights 1,2,2,1 and update y
y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
return t, y
def rk4_ode_p(p, y0, t0=0.0, t_end=10.0, steps=1000):
"""
Solve dy/dt + p*t*y = cos(pi*t) on [t0, t_end] with RK4.
 Rearranged form: dy/dt = cos(pi*t) - p*t*y
"""
# wrap the specific right-hand side in a little function f(t,y)
def f(t, y):
return math.cos(math.pi * t) - p * t * y
# then use the general RK4 solver with this f
return rk4(f, t0, y0, t_end, steps)

def plot_q2():
"""
Plot RK4 solutions:
- first: fix y(0) and change p
- second: fix p and change y(0)
Just to see how the parameter and initial condition affect the solution.
"""
# ---------- Figure 1: fixed y(0), vary p ----------
y0_fixed = 0.5
p_values = [0.1, 0.5, 1.0, 2.0]
plt.figure(figsize=(10, 6))
for p in p_values:
t, y = rk4_ode_p(p, y0_fixed)
plt.plot(t, y, label=f"p = {p}, y(0) = {y0_fixed}")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Question 2: RK4 – influence of p (fixed y(0) = 0.5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------- Figure 2: fixed p, vary y(0) ----------
p_fixed = 1.0
y0_values = [-1.0, 0.0, 0.5, 1.0]
plt.figure(figsize=(10, 6))
for y0 in y0_values:
t, y = rk4_ode_p(p_fixed, y0)
plt.plot(t, y, label=f"y(0) = {y0}")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(f"Question 2: RK4 – influence of y(0) (p = {p_fixed})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
###########################################
# Question 3: Essay (300–500 words)
###########################################
"""
During my degree I would like to use Python to support a coursework in Structures
and Materials where we analyse the deflection and internal forces in a simply
supported beam carrying several point loads. This mathematical model involves
equilibrium equations, shear-force and bending-moment diagrams, and in some cases
a differential equation for the deflection curve. This links well with the material
from MTHB5010A, because it combines systems of linear equations, numerical methods and plotting.
I would structure the code script in a similar way to this coursework. At the top,
I would first import numpy for arrays and linear algebra and matplotlib.pyplot for
graphs (I would also import sympy if I wanted to derive an exact formula for the
bending moment or check my hand calculations). I would then define a small collection
of functions that sets up the loads and support reactions. In addition to that,
I would implement another function that returns the bending moment as a function of position, and a numerical routine that uses a method like 
the trapezium rule or Runge–Kutta to
approximate the deflection along the beam.
I would avoid one huge script and instead write short, well named functions that
each perform a clear task to maintain organisation. For example, a function
setup parameter could store the length, Young's modulus and second moment of area
in a dictionary, while build load vector could construct the right-hand side for
the equilibrium equations. If I later wanted to change the loading case or material,
I would only need to edit the parameter function rather than rewrite the whole code.
I would use plotting commands to visualise the structural response, once the numerical
core is running. I could create two figures, one figure showing the shear force and
bending moment diagrams and the other figure showing the deflection curve for different materials or cross-sections. I would label axes, add 
legends, use subplots, etc in the
same style as the lab sessions, so that the figures can be copied directly into the
coursework report. Short precise comments and a top-level docstring would explain the
purpose of each function, making the code easier to mark and allowing me to reuse the
same structure in future design projects.
"""
###########################################
# Main
###########################################

def main():
"""
 Main function to run the different parts and show a couple of examples.
"""
print("="*60)
print("MTHB5010A Python Coursework - Running All Questions")
print("="*60)
# little demo for Q1(d)
print("\nQuestion 1(d): Special matrix example (n=6):")
print(special_matrix(6))
# little demo for Q1(e)
print("\nQuestion 1(e): Random 5x5 binary matrix:")
print(random_binary_matrix(5, 5))
print("\n" + "="*60)
print("Generating plots for all questions...")
print("="*60 + "\n")
# plotting functions for Q1(a), Q1(c) and Q2
plot_q1a()
plot_q1c()
plot_q2()
print("\nAll plots generated successfully!")
# Only run main() if we run this file directly (not if we import it somewhere else)
if __name__ == "__main__":
main()
