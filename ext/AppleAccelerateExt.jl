module AppleAccelerateExt

import PowerNetworkMatrices as PNM
using AppleAccelerate
import SparseArrays
import LinearAlgebra

# Extend the factorization creation function
function PNM._create_apple_accelerate_factorization(ABA)
    K = AppleAccelerate.AAFactorization(ABA)
    AppleAccelerate.factor!(K, AppleAccelerate.SparseFactorizationLDLT)
    return K
end

# Extend the solve function for AppleAccelerate factorizations
function PNM._solve_factorization(
    K::AppleAccelerate.AAFactorization{Float64},
    b::Vector{Float64},
)
    return AppleAccelerate.solve(K, b)
end

"""
Function for internal use only.

Computes the PTDF matrix by means of AppleAccelerate for sparse matrices.

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence Matrix
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        vector containing the indexes of the reference slack buses.
- `dist_slack::Vector{Float64}`:
        vector containing the weights for the distributed slacks.
"""
function PNM._calculate_PTDF_matrix_AppleAccelerate(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
    @warn "AppleAccelerate solver is experimental and may produce unexpected results. If you need high confidence use KLU"
    linecount = size(BA, 2)
    buscount = size(BA, 1)

    ABA = PNM.calculate_ABA_matrix(A, BA, ref_bus_positions)
    K = AppleAccelerate.AAFactorization(ABA)
    AppleAccelerate.factor!(K, AppleAccelerate.SparseFactorizationLDLT)

    # initialize matrices for evaluation
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)
    copyto!(PTDFm_t, BA)
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) < buscount
        PTDFm_t[valid_ix, :] = AppleAccelerate.solve(K, PTDFm_t[valid_ix, :])
        PTDFm_t[collect(ref_bus_positions), :] .= 0.0
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        PTDFm_t[valid_ix, :] = AppleAccelerate.solve(K, PTDFm_t[valid_ix, :])
        PTDFm_t[collect(ref_bus_positions), :] .= 0.0
        slack_array = dist_slack / sum(dist_slack)
        slack_array = reshape(slack_array, 1, buscount)
        return PTDFm_t .- (slack_array * PTDFm_t)
    else
        error("Distributed bus specification doesn't match the number of buses.")
    end

    return
end

"""
Function for internal use only.

Computes the LODF matrix by means of AppleAccelerate for sparse matrices.

# Arguments
- `a::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence Matrix
- `ptdf::Matrix{Float64}`:
        PTDF matrix
"""
function PNM._calculate_LODF_matrix_AppleAccelerate(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = a * ptdf
    m_I = Int[]
    m_V = Float64[]
    for iline in 1:linecount
        if (1.0 - ptdf_denominator_t[iline, iline]) < PNM.LODF_ENTRY_TOLERANCE
            push!(m_I, iline)
            push!(m_V, 1.0)
        else
            push!(m_I, iline)
            push!(m_V, 1 - ptdf_denominator_t[iline, iline])
        end
    end
    Dem_LU = AppleAccelerate.AAFactorization(SparseArrays.sparse(m_I, m_I, m_V))
    lodf_t = AppleAccelerate.solve(Dem_LU, ptdf_denominator_t)
    lodf_t[LinearAlgebra.diagind(lodf_t)] .= -1.0

    return lodf_t
end

end # module
