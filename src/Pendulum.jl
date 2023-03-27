module Pendulum

export DoublePendulum, plot_trajectory, create_animation, trajectory, generate_train_data

using Plots, OrdinaryDiffEq, NODEData

#include("./dsmodel.jl")

abstract type AbstractDSmodel end

function trajectory(model::AbstractDSmodel, x0, N_t=500, dt=0.1f0, t_transient=0)
    tspan = (0f0, Float32(t_transient + N_t * dt))
    prob = ODEProblem(model, x0, tspan, model.params) 
    sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)
end

function generate_train_data(model::AbstractDSmodel, series_length, x0; N_t=500, dt=0.1, t_transient=0, valid_set=nothing)
    t_train = t_transient:dt:t_transient+N_t*dt
    sol = trajectory(model, x0, N_t, dt, t_transient)
    data_train = Array(sol(t_train))
    if isnothing(valid_set)
        return NODEDataloader(Float32.(data_train), t_train, series_length)
    else
        return NODEDataloader(Float32.(data_train), t_train, series_length; valid_set=valid_set)
    end
end

struct DoublePendulum{P, E} <: AbstractDSmodel
    params::P
    m1::E
    m2::E
    l1::E
    l2::E
    g::E
end 

function DoublePendulum(params)
    DoublePendulum{typeof(params), typeof(params[1])}(params, params...)
end

#x = (θ_1, θ_2, ω_1, ω_2)
#p = (m_1, m_2, l_1, l_2, g)
function (m::DoublePendulum)(x,p,t)
    m1, m2, l1, l2, g = p
    dθ_1 = x[3]
    dθ_2 = x[4]
    dω_1 = begin 
        numerator = -g*(2m1 + m2)*sin(x[1]) - m2*g*sin(x[1] - 2x[2]) - 2sin(x[1] - x[2])*m2*(x[4]^2*l2 + x[3]^2*l1*cos(x[1] - x[2]))
        denominator = l1*(2m1 + m2 - m2*cos(2(x[1] - x[2])))
        numerator / denominator
        end     	
    dω_2 = begin 
        numerator = 2sin(x[1]-x[2])*(x[3]^2*l1*(m1 + m2) + g*(m1 + m2)*cos(x[1]) + x[4]^2*l2*m2*cos(x[1] - x[2]))
        denominator = l2*(2m1 + m2 - m2*cos(2(x[1] - x[2])))
        numerator / denominator
        end   

    return [dθ_1, dθ_2, dω_1, dω_2]
end

function plot_trajectory(model::DoublePendulum, sol)
    plot(sin.(sol[1,:])*model.l1 + sin.(sol[2,:])*model.l2, -cos.(sol[1,:])*model.l1 - cos.(sol[2,:])*model.l2)
end

function create_animation(model::DoublePendulum, sol)
    n = length(sol[1,:])
    m1, m2, l1, l2, g = model.params
    alpha_decay_length = 20
    alphas = [zeros(n - alpha_decay_length); collect(range(0,0.5,alpha_decay_length))]
    anim = @animate for i in 1:n
        plot([0,sin.(sol[1,i])*l1], [0,-cos.(sol[1,i])*l1], lw=3, legend=false)
        plot!([sin.(sol[1,i])*l1, sin.(sol[1,i])*l1 + sin.(sol[2,i])*l2], [-cos.(sol[1,i])*l1, -cos.(sol[1,i])*l1 - cos.(sol[2,i])*l2], lw=3, legend=false)
        plot!(sin.(sol[1,1:i])*l1 + sin.(sol[2,1:i])*l2, -cos.(sol[1,1:i])*l1 - cos.(sol[2,1:i])*l2, lw=1, linealpha=alphas[end-i+1:1:end], legend=false)
        xlims!(-(l1+l2),(l1+l2))
        ylims!(-(l1+l2),(l1+l2))
    end
    gif(anim, "pendulum.gif", fps = 10)
end

end