@testset "Aqua" begin
    import Aqua
    Aqua.test_unbound_args(PowerNetworkMatrices)
    Aqua.test_undefined_exports(PowerNetworkMatrices)
    Aqua.test_ambiguities(PowerNetworkMatrices)
    # The `*OpenAPIModels` packages are `[deps]` only so the root `[sources]` can pin them for
    # PSY (PSY's own pins are ignored once it is a dependency rather than the root project, and
    # a `[sources]` entry is dropped unless the package is also in `[deps]`). PNM never imports
    # them, so they are stale by construction.
    Aqua.test_stale_deps(
        PowerNetworkMatrices;
        ignore = [
            :Pardiso,
            :InfrastructureCoreOpenAPIModels,
            :InfrastructureTimeSeriesOpenAPIModels,
            :PowerCoreOpenAPIModels,
            :PowerDynamicsOpenAPIModels,
            :PowerInvestmentsOpenAPIModels,
            :PowerOpenAPIModels,
            :PowerOperationsOpenAPIModels,
        ],
    )
    Aqua.test_deps_compat(PowerNetworkMatrices)
    # `find_persistent_tasks_deps`/`test_persistent_tasks` are deliberately not run: they
    # precompile PNM inside a throwaway temp project that does not inherit this repo's
    # `[sources]` git pins, so PowerSystems resolves to the *registered* release instead of the
    # psy6 branch and PNM fails to load there (`UndefVarError: TransformerCircuit not defined in
    # PowerSystems`) for reasons unrelated to persistent tasks. PSY and PowerFlows stop at
    # `test_deps_compat` for the same reason; restore these once psy6 is released and the pins
    # come off.
end
