using OrdinaryDiffEq

abstract type AbstractDSmodel end

function trajectory(model::AbstractDSmodel, x0, N_t=500, dt=0.1f0, t_transient=0)
    tspan = (0f0, Float32(t_transient + N_t * dt))
    prob = ODEProblem(model, x0, tspan, model.params) 
    sol = solve(prob, Tsit5(), saveat=saveat=t_transient:dt:t_transient + N_t * dt)
end

function generate_train_data(model::AbstractDSmodel, series_length, x0, N_t=500, dt=0.1, t_transient=0)
    t_train = t_transient:dt:t_transient+N_t*dt
    sol = trajectory(model, x0, N_t, dt, t_transient)
    data_train = Array(sol(t_train))

    train = NODEDataloader(Float32.(data_train), t_train, series_length)
end