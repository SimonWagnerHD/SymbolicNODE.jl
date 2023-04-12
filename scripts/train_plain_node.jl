using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])

using Plots, OrdinaryDiffEq

push!(LOAD_PATH, "../src")

using Pendulum, SymReg

m1 = 1 #units of kg
m2 = 1 
l1 = 1 #units of m
l2 = 1
g = 10 #units of kg*m/s^2

pendulum = DoublePendulum([m1, m2, l1, l2, g])

#x0 = [3.141f0, 3.141f0, 0, 0] 
x0 = [Float32(π/2), Float32(π/2), 0f0, 0f0]

t_transient = 0f0
N_t = 8000
dt = 0.02f0

using NeuralODE

train_data, valid_data = generate_train_data(pendulum, 2, x0; N_t=N_t, dt=dt, t_transient=t_transient, periodic=true, valid_set=0.5)

p, re_nn = NODE_ANN(4,2,64,3,activation=tanh)
function node(du, u, p, t)
    θ₁, θ₂, ω₁, ω₂ = u

    du[1] = ω₁
    du[2] = ω₂
    du[3:4] .= re_nn(p)(u)

    return nothing
end

import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
import SciMLBase: FullSpecialize
node_prob = ODEProblem{true, FullSpecialize}(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob, trafo=pendulum_periodic, sensealg=BacksolveAdjoint(;autojacvec = ReverseDiffVJP(true),))

train_losses, valid_losses = train_NODE(model, train_data, epochs; loss=pendulum_loss, valid_data=train_data, re_nn=re_nn, η=1f-3, decay=1f-6, print_every=5, save_every=50, savefile="../models/plain_node")

using DelimitedFiles

writedlm(string("../data/plain_node_train",epochs,".csv"), train_losses, ',')
writedlm(string("../data/plain_node_valid",epochs,".csv"), valid_losses, ',')
save_ANN(re_nn(model.p), string("../models/plain_node",epochs,".bson"))