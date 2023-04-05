using Pkg
Pkg.activate("../.")

using OrdinaryDiffEq

push!(LOAD_PATH, "../src")

using Pendulum, SymReg

m1 = 1 #units of kg
m2 = 1 
l1 = 1 #units of m
l2 = 1
g = 10 #units of kg*m/s^2

pendulum = DoublePendulum([m1, m2, l1, l2, g])

x0 = [3.141f0, 3.141f0, 0, 0] 
#x0 = [Float32(π/2), Float32(π/2), 0f0, 0f0]

t_transient = 0f0
N_t = 3000
dt = 0.1f0

sol = trajectory(pendulum, x0, N_t, dt, t_transient)

gsr = GeneticSymReg(sol, collect(t_transient:dt:t_transient + N_t * dt); niter=80, maxsize=30, maxdepth=30)
save_gsr(gsr, "../models/genetic300_pi.ser")
