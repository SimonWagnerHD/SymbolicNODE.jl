using Pkg
Pkg.activate("../.")

epochs = parse(Int64, ARGS[1])
print_every = 1

using Plots, OrdinaryDiffEq

push!(LOAD_PATH, "../src")

function lorenz63(x,p,t)
    σ, β, ρ = p 
    [σ*(x[2] - x[1]), x[1]*(ρ - x[3]) - x[2], x[1]*x[2] - β*x[3]]
end

σ = 10
β = 8f0/3
ρ = 28
p = [σ, β, ρ]

x0 = [3.141f0, 3.141f0, 0, 0] 

t_transient = 0f0
N_t = 10000
dt = 0.02f0

t_transient = 100
N_t_train = 500
N_t_valid = N_t_train*3
N_t = N_t_train + N_t_valid
dt = 0.1f0
tspan = (0f0, Float32(t_transient + N_t * dt))

x0 = [0.1f0, 0.1f0, 0.1f0] 

prob = ODEProblem(lorenz63, x0, tspan, p) 
sol = solve(prob, Tsit5(), saveat=saveat=t_transient:dt:t_transient + N_t * dt)

t_train = t_transient:dt:t_transient+N_t_train*dt
data_train = Array(sol(t_train))

t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
data_valid = Array(sol(t_valid))

train_data = NODEDataloader(Float32.(data_train), t_train, 2)
valid_data = NODEDataloader(Float32.(data_valid), t_valid, 2)

using NeuralODE

#train_data, valid_data = generate_train_data(pendulum, 2, x0; N_t=N_t, dt=dt, t_transient=t_transient, valid_set=0.5)

p, re_nn = NODE_ANN(3,3,16,5)
node(u, p, t) = re_nn(p)(u)
node_prob = ODEProblem(node, x0, (Float32(0.),Float32(dt)), p)

model = NODE(node_prob)

train_losses, valid_losses = train_NODE(model, train_data, epochs; valid_data=valid_data, η=1f-3, decay=0.01, print_every=print_every)

using DelimitedFiles

writedlm(string("../data/node_train",epochs,".csv"), train_losses, ',')
writedlm(string("../data/node_valid",epochs,".csv"), valid_losses, ',')
save_ANN(re_nn(model.p), string("../models/node",epochs,".bson"))