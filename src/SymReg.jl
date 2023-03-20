module SymReg

export SINDy, GeneticSymReg, save_sindy, load_sindy

using OrdinaryDiffEq
using DataDrivenDiffEq, DataDrivenSparse
using SymbolicRegression
using BSplineKit
using Suppressor
using Serialization

abstract type AbstractSymRegModel end

struct SINDy{S} <: AbstractSymRegModel
    sol::S
end

function SINDy(trajectory::AbstractArray)
    @variables t, (x(t))[1:size(trajectory)[1]]
    ddprob = DataDrivenProblem(trajectory)
    basis = Basis([polynomial_basis(x, 3); sin_basis(x,1); cos_basis(x,1)], x, iv = t)
    optimiser = STLSQ(Float32.(exp10.(-5:0.1:-1)))
    ddsol = solve(ddprob, basis, optimiser, options = DataDrivenCommonOptions(digits = 1))
    SINDy{typeof(ddsol)}(ddsol)
end

function (m::SINDy)(u,p,t)
    m.sol(u,p,t)
end

function (m::SINDy)(u)
    m.sol(u,m.sol.prob.p,0)
end


#Custom save and load functions for SINDy object using Serialization.jl as plain BSON and JLD2 solutions do not work properly
function save_sindy(m::SINDy, filename)
    serialize(filename, m)
    nothing
end

function load_sindy(filename)
    deserialize(filename)
end

#Estimate time derivative of a trajectory which is only measured at some points in time
#This estimation uses BSplinesKit.jl 
function data_diff(x, t)
    mapslices(x -> data_diff_1dim(x,t), x, dims=2)
end

function data_diff_1dim(x, t)
    spl = interpolate(t, x, BSplineOrder(6))
    Float32.(diff(spl, Derivative(1)).(t))
end

struct GeneticSymReg{O,S} <: AbstractSymRegModel
    options::O
    sol::S
end

function GeneticSymReg(X::AbstractArray, t::AbstractArray; npop=20, niter=10)
    X = Array(X)

    sol = []

    options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    unary_operators=[cos, sin, exp],
    npopulations=npop
    )

    #Estimate time derivatives of data
    X_diff = data_diff(X, t)

    hall_of_fame = EquationSearch(
        X, X_diff, niterations=niter, options=options
    )

    for i in 1:size(X)[1]
        push!(sol, calculate_pareto_frontier(X, X_diff[i,:], hall_of_fame[i], options)[end].tree)
    end
    
    GeneticSymReg{typeof(options), typeof(sol)}(options, sol)
end

function (m::GeneticSymReg)(u)
    vcat(transpose.([eval_tree_array(m.sol[i], u, m.options)[1] for i in 1:size(u)[1]])...)
end

end