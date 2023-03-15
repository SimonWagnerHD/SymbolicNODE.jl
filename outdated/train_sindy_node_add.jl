using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])
print_every = 5

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
N_t = 500
dt = 0.1f0

times = t_transient:dt:t_transient + N_t * dt

sol = trajectory(pendulum, x0, N_t, dt, t_transient)

sindy = SINDy(sol)

using NeuralODE

train_data, valid_data = generate_train_data(pendulum, 3, x0; N_t=N_t, dt=dt, t_transient=t_transient, valid_set=0.4)

p, re_nn = NODE_ANN(4,4,16,4)
node(u, p, t) = re_nn(p)(sindy(u) + u)
node_prob = ODEProblem(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob)

train_losses, valid_losses = train_NODE(model, train_data, epochs; valid_data=valid_data, η=1f-3, print_every=print_every)

using DelimitedFiles, JLD2

writedlm(string("../data/sindy_node_add_train",epochs,".csv"), train_losses, ',')
writedlm(string("../data/sindy_node_add_valid",epochs,".csv"), valid_losses, ',')
save_ANN(re_nn(model.p), string("../models/sindy_node_add",epochs,".bson"))
save_object(string("../models/sindy_node_add",epochs,".jld2"), sindy)