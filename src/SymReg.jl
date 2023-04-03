module SymReg

export SINDy, GeneticSymReg, SymbolicAugment, save_model, load_model, save_gsr, load_gsr

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


#Custom save and load functions for SINDy object using Serialization.jl as plain BSON and JLD2 solutions do not work properly
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

function GeneticSymReg(X::AbstractArray, t::AbstractArray; niter=10, opt_args...)
    X = Array(X)

    sol = []

    options = SymbolicRegression.Options(;
    binary_operators=[+, *, /, -],
    unary_operators=[cos, sin],
    nested_constraints = [sin => [sin => 0, cos => 0], cos => [cos => 0, sin => 0]],
    save_to_file = false,
    opt_args...
    )

    #Estimate time derivatives of data
    X_diff = data_diff(X, t)
    hall_of_fame = @suppress begin EquationSearch(
        X, X_diff, niterations=niter, options=options,parallelism=:multithreading
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
        #push!(sol, calculate_pareto_frontier(X, X_diff[i,:], hall_of_fame[i], options)[end].tree)
        push!(sol, eqn)
    end
    
    GeneticSymReg{typeof(options), typeof(sol), typeof(expr), typeof(idc)}(options, sol, expr, idc)
end

#Constructor for loading a model from a file (see save_gsr)
# function GeneticSymReg(sol, options)
#     expr = []
#     idc = []

#     for i in 1:size(sol)[1]
#         eqn = node_to_symbolic(sol[i], options)
#         idx = [parse(Int64, string(x)[2:end]) for x in get_variables(eqn)]
#         eqn_expr = build_function(eqn, get_variables(eqn))
#         push!(expr, eval(eqn_expr))
#         push!(idc, idx)
#     end
    
#     GeneticSymReg{typeof(options), typeof(sol), typeof(expr), typeof(idc)}(options, sol, expr, idc)
# end

function GeneticSymReg(sol, options)
    expr = []
    idc = []

    for eqn in sol
        idx = [parse(Int64, string(x)[2:end]) for x in get_variables(eqn)]
        eqn_expr = build_function(eqn, get_variables(eqn))
        push!(expr, eval(eqn_expr))
        push!(idc, idx)
    end
    
    GeneticSymReg{typeof(options), typeof(sol), typeof(expr), typeof(idc)}(options, sol, expr, idc)
end

function (m::GeneticSymReg)(u)
    [m.expr[i](u[m.idx[i]]) for i in 1:length(m.expr)]
end

#Custom save and load functions for GeneticSymReg object storing only sol and options as expr does not get stored properly
function save_gsr(m::GeneticSymReg, filename)
    serialize(filename, (m.sol, m.options))
    nothing
end

function load_gsr(filename)
    sol, options = deserialize(filename)
    GeneticSymReg(sol, options)
end

#Extract terms with a given size from a symbolic expression 
function split_expr!(x::Expr, split; min_size=2, max_size=6)
    expr_size = 1
    for xx in x.args[2:end]
        if isa(xx,Expr)
            expr_size += split_expr!(xx, split; min_size, max_size)
        end
    end
    if (expr_size <= max_size) && (expr_size >= min_size)
        push!(split, x)
    end
    return expr_size
end

#Parses string and adds Float32 literals to all numbers 
function parse32(str::String)
    out = ""
    num = ['0','1','2','3','4','5','6','7','8','9']
    status = 0
    for (i,s) in enumerate(str)
        if (status == 0 || status == 1) && s in num
            status = 1
        elseif status == 1 && s == '.'
            status = 2
        elseif status == 1 && s != '.'
            status = 0
        elseif status == 2 && !(s in num)
            out = string(out, "f0")
            status = 0
        elseif status == 2 && i == length(str)
            out = string(out, s, "f0")
            return Meta.parse(out)
        end
        out = string(out, s)
    end
    return Meta.parse(out)
end

abstract type AbstractSymbolicAugment end

struct SymbolicAugment{S, N <: Integer, E, I} <: AbstractSymbolicAugment
    split::S
    N_eqn::N
    expr::E
    idx::I
end

function SymbolicAugment(model::AbstractSymRegModel, eqn_idx; min_size=2, max_size=6)
    split = []
    for i in eqn_idx
        eqn = parse32(string(model.sol[i]))
        split_expr!(eqn, split; min_size=min_size, max_size=max_size)
    end

    expr = []
    idc = []

    for (i, eqn) in enumerate(split)
        eqn = Symbolics.parse_expr_to_symbolic(eqn, Main)
        idx = [parse(Int64, string(x)[2:end]) for x in get_variables(eqn)]
        if length(idx) == 0
            deleteat!(split, i)
            continue
        end
        eqn_expr = build_function(eqn, get_variables(eqn))
        push!(expr, eval(eqn_expr))
        push!(idc, idx)
    end

    N_eqn = length(split)

    SymbolicAugment{typeof(split), typeof(N_eqn),typeof(expr),typeof(idc)}(split, N_eqn, expr, idc)
end

function (m::SymbolicAugment)(u)
    [m.expr[i](u[m.idx[i]]) for i in 1:length(m.expr)]
end

end