"""
    _build_bus_to_valid_idx(n_buses, valid_ix) -> Vector{Int}

Build the inverse of `valid_ix`: a length-`n_buses` vector where entry `b`
is the position of bus `b` inside `valid_ix`, or 0 if `b` is a reference
bus. Used by the Virtual\\* row-computation hot path so it can iterate the
nonzeros of a sparse `BA` column directly (O(nnz_col)) instead of scanning
the full bus axis (O(n_buses)) and bisecting the CSC for each entry.
"""
function _build_bus_to_valid_idx(n_buses::Int, valid_ix::Vector{Int})
    bus_to_valid_idx = zeros(Int, n_buses)
    @inbounds for (i, b) in enumerate(valid_ix)
        bus_to_valid_idx[b] = i
    end
    return bus_to_valid_idx
end

"""
Return a sparse matrix given a dense one by dropping elements whose absolute
value is below a certain tolerance.

Uses optimized `droptol!` for better performance compared to element-wise iteration.

# Arguments
- `dense_array::Matrix{Float64}`:
        input matrix (e.g., PTDF matrix).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Matrix{Float64}, tol::Float64)
    sparse_array = SparseArrays.sparse(dense_array)
    SparseArrays.droptol!(sparse_array, tol)
    return sparse_array
end

"""
Return a sparse vector given a dense one by dropping elements whose absolute
value is at or below a certain tolerance (keeps `abs(x) > tol`).

# Arguments
- `dense_array::Vector{Float64}`:
        input vector (e.g., PTDF row from VirtualPTDF).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Vector{Float64}, tol::Float64)
    # Count-then-fill: allocate the SparseVector at exactly the survivor count
    # instead of materializing every nonzero (`sparsevec`) and then compacting
    # (`droptol!`). On the VirtualPTDF/LODF row path an almost-dense row collapses
    # to a sparse one, so this avoids an O(n) over-allocation per cached row.
    nnz = 0
    @inbounds for v in dense_array
        abs(v) > tol && (nnz += 1)
    end
    nzind = Vector{Int}(undef, nnz)
    nzval = Vector{Float64}(undef, nnz)
    k = 0
    @inbounds for i in eachindex(dense_array)
        v = dense_array[i]
        if abs(v) > tol
            k += 1
            nzind[k] = i
            nzval[k] = v
        end
    end
    return SparseArrays.SparseVector(length(dense_array), nzind, nzval)
end
