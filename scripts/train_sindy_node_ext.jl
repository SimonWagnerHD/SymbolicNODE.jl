using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])

using OrdinaryDiffEq

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

sol = trajectory(pendulum, x0, N_t, dt, t_transient)

sindy = SINDy(sol)
save_model(sindy, "../models/sindy_node_ext.ser")

using NeuralODE

train_data, valid_data = generate_train_data(pendulum, 2, x0; N_t=N_t, dt=dt, t_transient=t_transient, valid_set=0.5)

p, re_nn = NODE_ANN(8,2,64,2)

function node(du, u, p, t)
    θ₁, θ₂, ω₁, ω₂ = u

    du[1] = ω₁
    du[2] = ω₂
    du[3:4] .= re_nn(p)(vcat(sindy(u)[3:4],[ω₁, ω₂, sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂)]))

    return nothing
end

import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
import SciMLBase: FullSpecialize
node_prob = ODEProblem{true, FullSpecialize}(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob, sensealg=BacksolveAdjoint(;autojacvec = ReverseDiffVJP(true),))

train_losses, valid_losses = train_NODE(model, train_data, epochs; valid_data=valid_data, re_nn=re_nn, η=1f-3, decay=1f-6, print_every=5, save_every=50, savefile="../models/sindy_node_ext")

using DelimitedFiles

writedlm(string("../data/sindy_node_ext_train",epochs,".csv"), train_losses, ',')
writedlm(string("../data/sindy_node_ext_valid",epochs,".csv"), valid_losses, ',')
save_ANN(re_nn(model.p), string("../models/sindy_node_ext",epochs,".bson"))
