# Precompilation workload: builds the network matrices the power-flow layer drives — Ybus (AC),
# ABA with KLU factorization, BA, IncidenceMatrix (DC) — and a direct KLU linear-solve cache, on a
# tiny in-memory system so first-call latency moves into the cached pkgimage.
#
# Scope is deliberately the power-flow path only. PTDF/LODF/MODF and the virtual/dense sensitivity
# factors are excluded: they are not on the common power-flow path and would inflate build time and
# image size for little first-call gain.
#
# To skip the workload during development iterations:
#     using PrecompileTools, Preferences
#     Preferences.set_preferences!(PowerNetworkMatrices, "precompile_workload" => false; force = true)

# Minimal in-memory system: 4 buses (REF/PV/PQ/PQ), a Source and ThermalStandard injector, two
# PowerLoads and a ZIP StandardLoad (exercises the constant-impedance-load Ybus path), and four
# Lines forming a mesh.
function _precompilation_workload_system()
    sys = PSY.System(100.0; time_series_in_memory = true)
    bus_types = (
        PSY.ACBusTypes.REF,
        PSY.ACBusTypes.PV,
        PSY.ACBusTypes.PQ,
        PSY.ACBusTypes.PQ,
    )
    buses = [
        PSY.ACBus(;
            number = i,
            name = "bus_$i",
            available = true,
            bustype = t,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.0, max = 2.0),
            base_voltage = 230.0,
        ) for (i, t) in enumerate(bus_types)
    ]
    for b in buses
        PSY.add_component!(sys, b)
    end
    PSY.add_component!(
        sys,
        PSY.Source(;
            name = "source_1",
            available = true,
            bus = buses[1],
            active_power = 0.0,
            reactive_power = 0.0,
            R_th = 1e-5,
            X_th = 1e-5,
        ),
    )
    PSY.add_component!(
        sys,
        PSY.ThermalStandard(;
            name = "thermal_2",
            available = true,
            status = true,
            bus = buses[2],
            active_power = 0.2,
            reactive_power = 0.0,
            rating = 1.0,
            active_power_limits = (min = 0.0, max = 1.0),
            reactive_power_limits = (min = -1.0, max = 1.0),
            ramp_limits = nothing,
            operation_cost = PSY.ThermalGenerationCost(nothing),
            base_power = 100.0,
            time_limits = nothing,
            prime_mover_type = PSY.PrimeMovers.OT,
            fuel = PSY.ThermalFuels.OTHER,
        ),
    )
    for (i, p, q) in ((3, 10.0, 5.0), (4, 8.0, 3.0))
        PSY.add_component!(
            sys,
            PSY.PowerLoad(;
                name = "load_$i",
                available = true,
                bus = buses[i],
                active_power = p,
                reactive_power = q,
                base_power = 1.0,
                max_active_power = 100.0,
                max_reactive_power = 100.0,
            ),
        )
    end
    PSY.add_component!(
        sys,
        PSY.StandardLoad(;
            name = "zip_3",
            available = true,
            bus = buses[3],
            base_power = 10.0,
            constant_active_power = 0.1,
            constant_reactive_power = 0.05,
            current_active_power = 0.05,
            current_reactive_power = 0.02,
            impedance_active_power = 0.05,
            impedance_reactive_power = 0.02,
            max_constant_active_power = 0.0,
            max_constant_reactive_power = 0.0,
            max_impedance_active_power = 0.0,
            max_impedance_reactive_power = 0.0,
            max_current_active_power = 0.0,
            max_current_reactive_power = 0.0,
        ),
    )
    line_params = ((1, 2, 0.10), (2, 3, 0.20), (3, 4, 0.05), (1, 4, 0.25))
    for (f, t, x) in line_params
        PSY.add_component!(
            sys,
            PSY.Line(;
                name = "line_$(f)_$(t)",
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = PSY.Arc(; from = buses[f], to = buses[t]),
                r = 1e-3,
                x = x,
                b = (from = 0.01, to = 0.01),
                rating = 1.0,
                angle_limits = (min = -pi / 2, max = pi / 2),
            ),
        )
    end
    return sys
end

PrecompileTools.@setup_workload begin
    # Component constructors and connectivity checks log; keep precompile output quiet.
    sys = Logging.with_logger(Logging.NullLogger()) do
        _precompilation_workload_system()
    end
    PrecompileTools.@compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            # AC network matrix.
            Ybus(sys)
            # DC matrices. `ABA_Matrix(sys; factorize = true)` cascades through Ybus,
            # IncidenceMatrix, BA_Matrix, the ABA assembly, and `klu_factorize`, warming the
            # whole DC-assembly-plus-factorization path in one call.
            ABA_Matrix(sys; factorize = true)
            BA_Matrix(sys)
            IncidenceMatrix(sys)
            # Direct KLU linear-solve cache — the hot path the AC power-flow layer factors the
            # Jacobian through every iteration. Warm the full KLU API the power-flow layer drives:
            # cache build, symbolic + numeric factorization, forward/transpose solve, refactor, and
            # the condition estimate. Both index widths: the Jacobian uses Int64 on Apple (libSparse
            # ABI) and Int32 elsewhere.
            rows = [1, 2, 3, 1, 2]
            cols = [1, 2, 3, 2, 1]
            vals = [2.0, 2.0, 2.0, -1.0, -1.0]
            for Ti in (Int32, Int64)
                A = SparseArrays.SparseMatrixCSC{Float64, Ti}(
                    SparseArrays.sparse(rows, cols, vals, 3, 3),
                )
                cache = KLULinSolveCache(A)
                full_factor!(cache, A)
                solve!(cache, [1.0, 2.0, 3.0])
                tsolve!(cache, [1.0, 2.0, 3.0])
                condest!(cache)
                symbolic_factor!(cache, A)
                numeric_refactor!(cache, A)
            end
            # Virtual (lazy, memory-light) sensitivity factors: construct and touch one row /
            # post-contingency entry to warm the row-cache and Woodbury paths. The costly ABA +
            # KLU foundation they build on is already warmed above, so this compiles only the
            # virtual-specific machinery. VirtualPTDF/VirtualMODF serve sensitivity/contingency
            # analysis, not the core power-flow solve.
            vptdf = VirtualPTDF(sys)
            vptdf[get_arc_axis(vptdf)[1], :]
            vmodf = VirtualMODF(sys)
            _compute_modf_entry(
                vmodf,
                1,
                NetworkModification(
                    "precompile",
                    [ArcModification(1, -vmodf.arc_susceptances[1])],
                ),
            )
        end
    end
end
