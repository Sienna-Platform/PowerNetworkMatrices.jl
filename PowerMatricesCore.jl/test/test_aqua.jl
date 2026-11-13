import Aqua

@testset "Aqua" begin
    Aqua.test_unbound_args(PowerMatricesCore)
    Aqua.test_undefined_exports(PowerMatricesCore)
    Aqua.test_ambiguities(PowerMatricesCore)
    Aqua.test_stale_deps(PowerMatricesCore)
    Aqua.test_deps_compat(PowerMatricesCore)
    Aqua.test_persistent_tasks(PowerMatricesCore)
end
