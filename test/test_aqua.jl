import Aqua

@testset "Aqua" begin
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
    # test_persistent_tasks is skipped: its temp project ignores [sources], resolves
    # registry PSY, and PNM fails to load. Restore after the psy6 release.
end
