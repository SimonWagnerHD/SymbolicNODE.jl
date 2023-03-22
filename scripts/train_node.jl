using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])
print_every = 1

using Plots, OrdinaryDiffEq

push!(LOAD_PATH, "../src")

using Pendulum, SymReg

m1 = 1 #units of kg
m2 = 1 
l1 = 1 #units of m
l2 = 1
g = 10 #units of kg*m/s^2

pendulum = DoublePendulum([m1, m2, l1, l2, g])

x0 = [3.141f0, 3.141f0, 0, 0] 

t_transient = 0f0
N_t = 10000
dt = 0.02f0

sol = trajectory(pendulum, x0, N_t, dt, t_transient)

using NeuralODE

train_data, valid_data = generate_train_data(pendulum, 2, x0; N_t=N_t, dt=dt, t_transient=t_transient, valid_set=0.5)

p, re_nn = NODE_ANN(4,4,64,3)
node(u, p, t) = re_nn(p)(u)
node_prob = ODEProblem(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob)

train_losses, valid_losses = train_NODE(model, train_data, epochs; valid_data=valid_data, η=1f-3, decay=0.1, print_every=print_every)

using DelimitedFiles

writedlm(string("../data/node_train",epochs,".csv"), train_losses, ',')
writedlm(string("../data/node_valid",epochs,".csv"), valid_losses, ',')
save_ANN(re_nn(model.p), string("../models/node",epochs,".bson"))