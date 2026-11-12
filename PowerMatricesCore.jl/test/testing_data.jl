import HDF5
import SparseArrays

function _five_bus_core(
    ref_key::Int = 1;
    arc_live_members = ones(Int, 6),
    b = [10.0, 5.0, 8.0, 4.0, 6.0, 7.0],
    tol = 0.0,
)
    bus_ax = [1, 2, 3, 4, 5]
    arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    n_bus = length(bus_ax)
    n_arc = length(arc_ax)
    A = SparseArrays.spzeros(Int8, n_arc, n_bus)
    BA = SparseArrays.spzeros(Float64, n_bus, n_arc)
    for (e, (f, t)) in enumerate(arc_ax)
        A[e, f] = 1
        A[e, t] = -1
        BA[f, e] = b[e]
        BA[t, e] = -b[e]
    end
    return PMC.FactorCore(
        BA,
        A,
        Set([1]),
        (arc_ax, bus_ax),
        Dict(ref_key => (copy(arc_ax), copy(bus_ax))),
        arc_live_members;
        network_data = nothing,
        linear_solver = "KLU",
        tol = tol,
    )
end

function _write_five_bus(
    path;
    ref_key::Int = 1,
    reverse_bus_map = Dict{Int, Int}(),
    arc_live_members = ones(Int, 6),
    tol = 0.0,
)
    core = _five_bus_core(ref_key; arc_live_members = arc_live_members, tol = tol)
    HDF5.h5open(path, "w") do f
        PMC.write_factor_core(
            f,
            core;
            shift_injection = zeros(length(PMC.get_arc_axis(core))),
            dist_slack = [0.0, 1.0, 1.0, 0.0, 2.0],
            reverse_bus_map = reverse_bus_map,
        )
    end
    return core
end

function _read_core(path)
    return PMC.read_factor_core(path; linear_solver = "KLU")
end

# Each case writes a valid file, changes one entry, and expects an error that names it.
function _hostile(change!, expected, reader = _read_core, writer = _write_five_bus)
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        writer(path)
        HDF5.h5open(change!, path, "r+")
        @test_throws expected reader(path)
    end
    return
end

function _set_column!(g, name::String, row::Int, value)
    column = read(g[name])
    column[row] = value
    _replace_entry!(g, name, column)
    return
end

function _replace_entry!(g, name::String, value)
    HDF5.delete_object(g, name)
    g[name] = value
    return
end

# Arc 1 holds a parallel group of two branches; the second one points against the arc. The
# arcs 2 to 5 hold one branch each. Arc 6 holds a series chain of two segments.
function _six_arc_table(; alpha3::Float64 = 0.0)
    return PMC.BranchArcTable(
        [9, 1, 1, 1, 2, 1, 1, 10, 1, 1],
        [0, 101, 102, 103, 104, 105, 106, 0, 107, 108],
        [-1, 1, 1, -2, -3, -4, -5, -6, 8, 8],
        [1, 1, -1, 1, 1, 1, 1, 1, 1, -1],
        [0.0, 4.0, 6.0, 5.0, 8.0, 4.0, 6.0, 0.0, 14.0, 14.0],
        [0.0, 0.0, alpha3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6,
    )
end

# Arc 1 is a parallel group of two members: "a" and "b" each trip one member, and "ab" opens
# the arc. "c" opens arc 4. "d" opens the arcs 4 and 5, which cuts the buses 4 and 5 from the
# network.
function _outage_mods(core)
    b = PMC._get_arc_susceptances(core)
    z = zero(PMC.YBUS_ELTYPE)
    partial(arc, delta) = PMC.ArcModification(arc, delta, 0.0, z, z, z, z, 1)
    opening(arc, n) = PMC.ArcModification(arc, -b[arc], 0.0, z, z, z, z, n)
    none = PMC.ShuntModification[]
    return [
        PMC.NetworkModification("a", [partial(1, -4.0)], none, false),
        PMC.NetworkModification("b", [partial(1, -6.0)], none, false),
        PMC.NetworkModification("ab", [opening(1, 2)], none, false),
        PMC.NetworkModification("c", [opening(4, 1)], none, false),
        PMC.NetworkModification("d", [opening(4, 1), opening(5, 1)], none, true),
    ]
end

const _OUTAGE_NAMES = ["a", "b", "ab", "c", "d"]
const _OUTAGE_IDS = [11, 12, 13, 14, 15]

function _write_outage_file(path; registered::Bool = true)
    core = _write_five_bus(path; arc_live_members = [2, 1, 1, 1, 1, 1])
    mods = _outage_mods(core)
    n_arc = length(PMC.get_arc_axis(core))
    HDF5.h5open(path, "r+") do f
        g = HDF5.create_group(f, "branch_outages")
        PMC._write_strings!(g, "type", fill("Line", length(mods)))
        PMC._write_strings!(g, "name", _OUTAGE_NAMES)
        g["id"] = _OUTAGE_IDS
        g["is_islanding"] = [m.is_islanding for m in mods]
        PMC._write_row_arrays(g, mods, n_arc)
        g["opens_arc"] = [false, false, true, true, true, true]
        g["member"] = [1, 2, 1, 0, 0, 0]
        if registered
            r = HDF5.create_group(f, "outages")
            r["id"] = [21, 22]
            PMC._write_strings!(r, "label", ["first", "second"])
            r["is_islanding"] = [false, true]
            PMC._write_row_arrays(r, mods[[4, 5]], n_arc)
        end
    end
    return core, mods
end
