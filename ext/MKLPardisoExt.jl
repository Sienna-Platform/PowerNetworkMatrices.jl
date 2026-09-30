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

end # module
