using OrdinaryDiffEq, SciMLSensitivity

function lorenz(u, p, t)
    x, y, z = u

    dx = 10.0 * (y - x)
    dy = x * (28.0 - z) - y
    dz = x * y - (8/3) * z
    
    return [dx, dy, dz]
end

function sol()
    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    dt = 0.1
    prob = ODEProblem(lorenz, u0, tspan)
    sol = solve(prob, Tsit5(), saveat = dt)
    return nothing
end