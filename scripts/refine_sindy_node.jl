using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])
series_length = parse(Int64, ARGS[2])
print_every = 5

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

t_transient = 0f0
N_t = 500
dt = 0.1f0

sindy = load_object(string("../models/sindy_node.jld2"))

using NeuralODE

train_data, valid_data = generate_train_data(pendulum, series_length, x0; N_t=N_t, dt=dt, t_transient=t_transient, valid_set=0.4)

p, re_nn = load_ANN("../models/sindy_node.bson")
node(u, p, t) = re_nn(p)(sindy(u))
node_prob = ODEProblem(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob)

train_losses, valid_losses = train_NODE(model, train_data, epochs; valid_data=valid_data, η=1f-3, print_every=print_every)

using DelimitedFiles

writedlm("../data/sindy_node_train.csv", train_losses, ',')
writedlm("../data/sindy_node_valid.csv", valid_losses, ',')
save_ANN(re_nn(model.p), "../models/sindy_node.bson")
save_object("../models/sindy_node.jld2", sindy)