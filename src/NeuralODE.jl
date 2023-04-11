module NeuralODE

export NODE, NODE_ANN, save_ANN, load_ANN, train_NODE

using Flux
using OrdinaryDiffEq
using SciMLSensitivity
using BSON
using Statistics
using ParameterSchedulers: CosAnneal

abstract type AbstractNDEModel end 

"""
NODE{P,R,A,K} <: AbstractNDEModel

Model for setting up and training Neural Ordinary Differential Equations.

# Fields:

* `p` vector of trainable parameters 
* `prob` ODEProblem 
* `alg` Algorithm to use for the `solve` command 
* `trafo` transformation function which is applied to the output of the Neural ODE
* `kwargs` any additional keyword arguments that should be handed to the solve method
"""

struct NODE{P,R,A,T,K} <: AbstractNDEModel
    p::P 
    prob::R 
    alg::A
    trafo::T
    kwargs::K
end 

function NODE(prob; trafo=x->x, alg=Tsit5(), kwargs...)
    p = prob.p 
    NODE{typeof(p), typeof(prob), typeof(alg), typeof(trafo), typeof(kwargs)}(p, prob, alg, trafo, kwargs)
end 

Flux.@functor NODE
Flux.trainable(m::NODE) = (p=m.p,)

function (m::NODE)(X,p=m.p)
    (t, x) = X 
    sol = Array(solve(remake(m.prob; tspan=(t[1],t[end]),u0=x[:,1],p=p), m.alg; saveat=t, m.kwargs...))
    return m.trafo(sol)
end

#Function for the construction of an ANN used in a Neural ODE n_layer includes hidden and output layers
function NODE_ANN(in_dim, out_dim, n_weights, n_layers; activation=relu)
    layer_sizes = [in_dim]
    layer_activations = []
    for i in 1:(n_layers - 1)
        push!(layer_sizes, n_weights)
        push!(layer_activations, activation) 
    end
    push!(layer_sizes, out_dim)
    push!(layer_activations, identity)

    layers = [Dense(layer_sizes[i], layer_sizes[i+1],layer_activations[i]) for i in 1:(length(layer_sizes) - 1)]
    nn = Chain(layers...)
    p, re_nn = Flux.destructure(nn)
end

#Function for saving an ANN
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
function train_NODE(model::AbstractNDEModel, train_data, epochs; loss=Flux.Losses.mse, valid_data=nothing, re_nn=nothing, η=1f-3, η₂=1f-4, period=100, decay=1f-6, print_every=1, save_every=50, savefile="../models/node")
    schedule = CosAnneal(λ0=η, λ1=η₂, period=period)
    opt = Flux.AdamW(η, (0.9, 0.999), decay)
    opt_state = Flux.setup(opt, model)

    train_losses = Float32[]
    valid_losses = Float32[]

    println("Begin training")
    for (learn_rate, epoch) in zip(schedule, 1:epochs)
        Flux.adjust!(opt_state, learn_rate)
        losses = Float32[]
        for (i, data) in enumerate(train_data)
            t, x = data
	        val, grads = 0,0
            # Calculate the loss and gradients for the forward pass.
            # An error can occur if the ODE solution is instable and and consequently the solution specified at time t₂ is not defined.
            # In that case skip the update and continue with the next batch.
            try
                val, grads = Flux.withgradient(model) do m
                    result = m((t,x))
                    loss(result, x)
                end 
            catch e
                @error "An error occured while calculating the training loss, skipping update"
                println(e)
                continue
	        end

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(losses, val)
            Flux.update!(opt_state, model, grads[1])
        end
        train_loss = Statistics.mean(losses)
        push!(train_losses, train_loss)

        #If valid_data is specified, calculate the validation loss every epoch
        if !isnothing(valid_data)
            losses = Float32[]
            valid_loss = 0
            try
                for (i, data) in enumerate(valid_data)
                    t, x = data
                    result = model((t,x))
                    push!(losses, loss(result, x))
                end

                valid_loss = Statistics.mean(losses)
                push!(valid_losses, valid_loss)
            catch e
                @warn "An error occured while calculating the validation loss"
                push!(valid_losses, NaN)
            end

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
	
        #If savefile is specified, save the model every save_every epochs
        if ((epoch % save_every == 0) && !isnothing(re_nn))
            save_ANN(re_nn(model.p), string(savefile,epoch,".bson"))
        end

    end

    if isnothing(valid_data)
        return train_losses
    else
        return train_losses, valid_losses
    end
end

end