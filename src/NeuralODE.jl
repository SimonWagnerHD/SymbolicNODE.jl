module NeuralODE

export NODE, NODE_ANN, save_ANN, load_ANN, train_NODE

using Flux
using OrdinaryDiffEq
using SciMLSensitivity
using BSON
using Statistics

abstract type AbstractNDEModel end 

"""
NODE{P,R,A,K} <: AbstractNDEModel

Model for setting up and training Neural Ordinary Differential Equations.

# Fields:

* `p` vector of trainable parameters 
* `prob` ODEProblem 
* `alg` Algorithm to use for the `solve` command 
* `kwargs` any additional keyword arguments that should be handed over
"""

struct NODE{P,R,A,K} <: AbstractNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
end 

function NODE(prob; alg=Tsit5(), kwargs...)
    p = prob.p 
    NODE{typeof(p), typeof(prob), typeof(alg), typeof(kwargs)}(p, prob, alg, kwargs)
end 

Flux.@functor NODE
Flux.trainable(m::NODE) = (p=m.p,)

function (m::NODE)(X,p=m.p)
    (t, x) = X 
    Array(solve(remake(m.prob; tspan=(t[1],t[end]),u0=x[:,1],p=p), m.alg; saveat=t, m.kwargs...))
end

#Function for the construction of an ANN used in a Neural ODE n_layer includes hidden and output layers
function NODE_ANN(in_dim, out_dim, n_weights, n_layers)
    layer_sizes = [in_dim]
    layer_activations = []
    for i in 1:(n_layers - 1)
        push!(layer_sizes, n_weights)
        push!(layer_activations, relu) 
    end
    push!(layer_sizes, out_dim)
    push!(layer_activations, identity)

    layers = [Dense(layer_sizes[i], layer_sizes[i+1],layer_activations[i]) for i in 1:(length(layer_sizes) - 1)]
    nn = Chain(layers...)
    p, re_nn = Flux.destructure(nn)
end

function save_ANN(model, filename)
    savemodel = model
    BSON.@save filename savemodel
end

function load_ANN(filename)
    Core.eval(Main, :(import NNlib, Flux))
    BSON.@load filename savemodel
    p, re_nn = Flux.destructure(savemodel)
end

#Train a given NODE
function train_NODE(model::AbstractNDEModel, train_data, epochs; valid_data=nothing, η=1f-3, decay=0.1, print_every=1)
    loss = Flux.Losses.mse
    opt = Flux.AdamW(η, (0.9, 0.999), decay)
    opt_state = Flux.setup(opt, model)

    train_losses = Float32[]
    valid_losses = Float32[]

    println("Begin training")
    for epoch in 1:epochs
        losses = Float32[]
        for (i, data) in enumerate(train_data)
            t, x = data
            val, grads = Flux.withgradient(model) do m
                result = m((t,x))
                loss(result, x)
            end
            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(losses, val)

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_state, model, grads[1])
        end
        train_loss = Statistics.mean(losses)
        push!(train_losses, train_loss)

        if !isnothing(valid_data)
            losses = Float32[]

            for (i, data) in enumerate(valid_data)
                t, x = data
                result = model((t,x))
                push!(losses, loss(result, x))
            end

            valid_loss = Statistics.mean(losses)
            push!(valid_losses, valid_loss)

            if (epoch % print_every) == 0 
                println("Epoch: $epoch; Train Loss: $train_loss; Validation Loss: $valid_loss")
                flush(stdout)
            end
        else
            if (epoch % print_every) == 0 
                println("Epoch: $epoch; Train Loss: $train_loss")
                flush(stdout)
            end
        end

        if (epoch % 30) == 0  # reduce the learning rate every 30 epochs
            η /= 2
            Flux.adjust!(opt_state, η)
        end
    end

    if isnothing(valid_data)
        return train_losses
    else
        return train_losses, valid_losses
    end
end

end