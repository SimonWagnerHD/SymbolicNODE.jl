module TSData

abstract type AbstractTSDataloader{T,U,N} end

using SciMLBase, EllipsisNotation, Random

"""

TSDataloader{T,U,N} <: AbstractTSDataloader{T,U,N}

Data loader for time series data which can be used to train a NeuralODE.

# Fields:

* `data` array holding a phase space trajectory 
* `t` array of corresponding time points
* `N` number of samples in the data loader
* `N_length` length of one sample
* `idx` array of startindices which allows to shuffle the load order of samples

"""

struct TSDataloader{T<:AbstractArray,U<:AbstractVector,N<:Integer} <: AbstractTSDataloader{T,U,N,}
    data::T
    t::U
    N::N
    N_length::N
    idx::Vector{Int64}
end

function TSDataloader(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_length::Integer; valid_set=nothing, shuffle = true) where {T,U,N}
    @assert size(data)[end] == length(t) "Length of data and t should be equal"

    if isnothing(valid_set)
        if shuffle 
            idx = randperm(length(t) - N_length)
        else 
            idx = collect(1:length(t) - N_length)
        end

        return TSDataloader(Array(data), Array(t), length(t) - N_length, N_length, idx)
    else 
        @assert 0 <= valid_set < 1 "Valid_set should be ∈ [0,1]"

        N_t = length(t)
        N_t_valid = Int(floor(valid_set*N_t))
        N_t_train = N_t - N_t_valid

        if shuffle 
            idx_train = randperm(N_t_train - N_length)
            idx_valid = randperm(N_t_valid - N_length)
        else 
            idx_train = collect(1:N_t_train - N_length)
            idx_valid = collect(1:N_t_valid - N_length)
        end

        return TSDataloader(Array(data[..,1:N_t_train]), Array(t[1:N_t_train]), N_t_train - N_length, N_length, idx_train), TSDataloader(Array(data[..,N_t_train+1:N_t]), Array(t[N_t_train+1:N_t]), N_t_valid - N_length, N_length, idx_valid)
    end
end 

function Base.getindex(iter::TSDataloader{T,U,N}, i::Integer) where {T,U,N}
    @assert 0 < i <= iter.N
    i = iter.idx[i]
    return (iter.t[i:i+iter.N_length-1] ,iter.data[..,i:i+iter.N_length-1])
end

function Base.iterate(iter::AbstractTSDataloader, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::AbstractTSDataloader) = iter.N
Base.eltype(iter::AbstractTSDataloader) = eltype(iter.data)

Base.firstindex(iter::AbstractTSDataloader) = 1
Base.lastindex(iter::AbstractTSDataloader) = iter.N
Base.show(io::IO,seq::TSDataloader{T,U,N}) where {T,U,N} = print(io, "TSData{",T,",",N,"} with ",seq.N," batches of length ",seq.N_length)

export TSDataloader

end