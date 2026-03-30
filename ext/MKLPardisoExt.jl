module MKLPardisoExt

import PowerNetworkMatrices as PNM
using Pardiso
import SparseArrays
import LinearAlgebra

"""
Function for internal use only.

Computes the PTDF matrix by means of the MKL Pardiso for dense matrices.

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
function PNM._calculate_PTDF_matrix_MKLPardiso(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
    linecount = size(BA, 2)
    buscount = size(BA, 1)

    ABA = PNM.calculate_ABA_matrix(A, BA, ref_bus_positions)
    @assert LinearAlgebra.issymmetric(ABA)
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM)
    Pardiso.pardisoinit(ps)
    # Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    defaults = Pardiso.get_iparms(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    for (ix, v) in enumerate(defaults[2:end])
        Pardiso.set_iparm!(ps, ix + 1, v)
    end
    Pardiso.set_iparm!(ps, 2, 2)
    Pardiso.set_iparm!(ps, 59, 2)
    Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_iparm!(ps, 12, 1)
    Pardiso.set_iparm!(ps, 11, 0)
    Pardiso.set_iparm!(ps, 13, 0)
    Pardiso.set_iparm!(ps, 32, 1)

    # initialize matrices for evaluation
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)

    full_BA = Matrix(BA[valid_ix, :])
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) != buscount
        Pardiso.pardiso(ps, PTDFm_t[valid_ix, :], ABA, full_BA)
        PTDFm_t[valid_ix, :] = full_BA
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps)
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        Pardiso.pardiso(ps, PTDFm_t[valid_ix, :], ABA, full_BA)
        PTDFm_t[valid_ix, :] = full_BA
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps)
        slack_array = dist_slack / sum(dist_slack)
        slack_array = reshape(slack_array, 1, buscount)
        return PTDFm_t - ones(buscount, 1) * (slack_array * PTDFm_t)
    else
        error("Distributed bus specification doesn't match the number of buses.")
    end
    return
end

function PNM._pardiso_sequential_LODF!(
    lodf_t::Matrix{Float64},
    A::SparseArrays.SparseMatrixCSC{Float64, Int},
    ptdf_denominator_t::Matrix{Float64},
    chunk_size::Int = PNM.DEFAULT_LODF_CHUNK_SIZE,
)
    @info "Line Count too large for single compute using Pardiso. Employing Sequential Calculations using a chunk_size=$(chunk_size)"
    linecount = size(lodf_t, 1)
    @assert LinearAlgebra.ishermitian(A)
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM)
    Pardiso.pardisoinit(ps)
    # Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    defaults = Pardiso.get_iparms(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    for (ix, v) in enumerate(defaults[2:end])
        Pardiso.set_iparm!(ps, ix + 1, v)
    end
    Pardiso.set_iparm!(ps, 2, 2)
    Pardiso.set_iparm!(ps, 59, 2)
    Pardiso.set_iparm!(ps, 12, 1)
    Pardiso.set_iparm!(ps, 11, 0)
    Pardiso.set_iparm!(ps, 13, 0)
    Pardiso.set_iparm!(ps, 32, 1)
    #Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(
        ps,
        lodf_t,
        A,
        ptdf_denominator_t,
    )

    Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    Pardiso.pardiso(
        ps,
        A,
        Float64[],
    )
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    i_count = 1
    tmp = zeros(Float64, linecount, chunk_size)
    while i_count <= linecount
        edge = min(i_count + chunk_size - 1, linecount)
        if linecount - edge <= 0
            tmp = tmp[:, 1:(edge - i_count + 1)]
        end
        Pardiso.pardiso(
            ps,
            tmp,
            A,
            ptdf_denominator_t[:, i_count:edge],
        )
        lodf_t[:, i_count:edge] .= tmp
        i_count = edge + 1
    end
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps)
    return
end

function PNM._pardiso_single_LODF!(
    lodf_t::Matrix{Float64},
    A::SparseArrays.SparseMatrixCSC{Float64, Int},
    ptdf_denominator_t::Matrix{Float64},
)
    @assert LinearAlgebra.ishermitian(A)
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    defaults = Pardiso.get_iparms(ps)
    for (ix, v) in enumerate(defaults[2:end])
        Pardiso.set_iparm!(ps, ix + 1, v)
    end
    Pardiso.set_iparm!(ps, 2, 2)
    Pardiso.set_iparm!(ps, 59, 2)
    Pardiso.set_iparm!(ps, 12, 1)
    #Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    Pardiso.pardiso(
        ps,
        lodf_t,
        A,
        ptdf_denominator_t,
    )
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps)
    return
end

function PNM._calculate_LODF_matrix_MKLPardiso(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = similar(ptdf, size(a, 1), linecount)
    PNM.threaded_sparse_dense_mul!(ptdf_denominator_t, a, ptdf)
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
    lodf_t = zeros(linecount, linecount)
    A = SparseArrays.sparse(m_I, m_I, m_V)
    if linecount > PNM.DEFAULT_LODF_CHUNK_SIZE
        PNM._pardiso_sequential_LODF!(lodf_t, A, ptdf_denominator_t)
    else
        PNM._pardiso_single_LODF!(lodf_t, A, ptdf_denominator_t)
    end
    lodf_t[LinearAlgebra.diagind(lodf_t)] .= -1.0
    return lodf_t
end

end # module
