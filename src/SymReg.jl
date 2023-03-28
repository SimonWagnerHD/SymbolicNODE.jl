module SymReg

export SINDy, GeneticSymReg, save_model, load_model

using OrdinaryDiffEq
using DataDrivenDiffEq, DataDrivenSparse
using SymbolicRegression
using BSplineKit
using Suppressor
using Serialization
using Symbolics

abstract type AbstractSymRegModel end

struct SINDy{S} <: AbstractSymRegModel
    sol::S
end

function SINDy(trajectory::AbstractArray; poly_base=2, trig_base=2)
    @variables t, (x(t))[1:size(trajectory)[1]]
    ddprob = DataDrivenProblem(trajectory)
    basis = Basis([polynomial_basis(x, poly_base); sin_basis(x,trig_base); cos_basis(x,trig_base)], x, iv = t)
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


#Custom save and load functions for SymRegModel object using Serialization.jl as plain BSON and JLD2 solutions do not work properly
function save_model(m::AbstractSymRegModel, filename)
    serialize(filename, m)
    nothing
end

function load_model(filename)
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

function get_vars(eqn, num_vars)
    @variables dummy[1:num_vars]
    vars = []
    var_dict = Dict([parse(Int64, string(x)[2:end]) => x for x in get_variables(eqn)])
    for i in 1:num_vars
        if i in keys(var_dict)
            push!(vars, var_dict[i])
        else
            push!(vars, dummy[i])
        end
    end
end

struct GeneticSymReg{O,S,E,I} <: AbstractSymRegModel
    options::O
    sol::S
    expr::E
    idx::I
end

function GeneticSymReg(X::AbstractArray, t::AbstractArray; npop=20, niter=10)
    X = Array(X)

    sol = []

    options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    unary_operators=[cos, sin],
    npopulations=npop,
    nested_constraints = [sin => [sin => 0, cos => 0], cos => [cos => 0, sin => 0]],
    enable_autodiff = true,
    save_to_file = false
    )

    #Estimate time derivatives of data
    X_diff = data_diff(X, t)
    hall_of_fame = @suppress begin EquationSearch(
        X, X_diff, niterations=niter, options=options
    )
    end

    expr = []
    idc = []

    for i in 1:size(X)[1]
        eqn = node_to_symbolic(calculate_pareto_frontier(X, X_diff[i,:], hall_of_fame[i], options)[end].tree, options)
        idx = [parse(Int64, string(x)[2:end]) for x in get_variables(eqn)]
        eqn_expr = build_function(eqn, get_variables(eqn))
        push!(expr, eval(eqn_expr))
        push!(idc, idx)
        push!(sol, convert(Node{Float32}, calculate_pareto_frontier(X, X_diff[i,:], hall_of_fame[i], options)[end].tree))
    end
    
    GeneticSymReg{typeof(options), typeof(sol), typeof(expr), typeof(idc)}(options, sol, expr, idc)
end

function (m::GeneticSymReg)(u)
    [m.expr[i](u[m.idx[i]]) for i in 1:length(m.expr)]
end

end