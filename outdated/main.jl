println("Importing modules")

using Plots, OrdinaryDiffEq

#push!(LOAD_PATH, "../src")

include("../src/Pendulum.jl")
include("../src/SymReg.jl")

using .Pendulum, .SymReg

println("Generate model trajectory")

m1 = 1 #units of kg
m2 = 1 
l1 = 1 #units of m
l2 = 1
g = 10 #units of kg*m/s^2

pendulum = DoublePendulum([m1, m2, l1, l2, g])

x0 = [3.141f0, 3.141f0, 0, 0] 

t_transient = 0f0
N_t = 500
dt = 0.1f0

sol = trajectory(pendulum, x0, N_t, dt, t_transient)
display(plot(sol))
plot_trajectory(pendulum, sol)

println("Fit SINDy")

sindy = SINDy(sol)

println("Generate SINDy trajectory")

sol_sindy = solve(sindy,x0,(0f0, Float32(t_transient + N_t * dt)))
display(plot(sol_sindy))
plot_trajectory(pendulum, sol_sindy)

println("Import ODE")

include("../src/ODE.jl")
using .ODE

println("Create NODE model")

train_data = generate_train_data(pendulum, 3, x0, N_t, dt, t_transient)

p, re_nn = NODE_ANN(4,4,4,2)

#node(u, p, t) = sindy_fct(u,p,t) + re_nn(p)(u)
node(u, p, t) = sindy(u,sindy.sol.prob.p,t) + re_nn(p)(u)
#node(u, p, t) = re_nn(p)(sindy(u))
node_prob = ODEProblem(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob)

println("Train model")

loss_log = train_NODE(model, train_data, 2, 1f-3)
plot(loss_log)