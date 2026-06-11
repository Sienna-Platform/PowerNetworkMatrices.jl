# Flowgates

A *flowgate* pairs a contingency (an outaged branch) with the branches its outage most
affects, ranked by how hard the post-contingency flow loads each monitored branch relative
to its rating. `flowgates` derives them from a built `LODF` and returns a vector of
`FlowgateResult` values, each carrying a `PowerSystems.FixedForcedOutage` ready to attach to
the system.

This tutorial computes flowgates for a small system, inspects them through getter functions,
selects the ones worth studying, and registers their outages on the `System`.

## Computing flowgates

```@repl tutorial_Flowgates
using PowerNetworkMatrices
import PowerSystems as PSY
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");
lodf = LODF(sys);

flowgate_list = flowgates(lodf; top_n = 3)
```

`top_n` caps the size of each monitored set; the returned vector holds one flowgate per
meshed contingency, ranked by importance.

## Inspecting a flowgate

Retrieve the parts of a flowgate through its getters rather than by field access:

```@repl tutorial_Flowgates
top = first(flowgate_list);

get_flowgate_contingency_branch(top)   # the outaged branch

get_flowgate_monitored(top)            # branches it most affects, most impacted first

get_flowgate_impacts(top)              # signed impact as a fraction of each monitored rating

get_flowgate_score(top)                # importance score in [0, 1]
```

The contingency and monitored elements are `PowerSystems` branch components, so the usual
`PowerSystems` getters apply:

```@repl tutorial_Flowgates
PSY.get_name(get_flowgate_contingency_branch(top))

PSY.get_name.(get_flowgate_monitored(top))
```

## Selecting which flowgates to keep

Not every ranked flowgate is worth carrying into a study. Keep the highest-ranked few:

```@repl tutorial_Flowgates
selected = first(flowgate_list, 2)
```

Any selection rule works — for example, keep only flowgates whose score clears a threshold
with `filter(fg -> get_flowgate_score(fg) >= 0.5, flowgate_list)`.

## Adding the outages to the system

Each flowgate carries a `FixedForcedOutage` for its contingency branch. Attach the outages of
the selected flowgates so downstream tools see them as contingencies:

```@repl tutorial_Flowgates
for fg in selected
    PSY.add_supplemental_attribute!(
        sys,
        get_flowgate_contingency_branch(fg),
        get_flowgate_outage(fg),
    )
end

collect(PSY.get_supplemental_attributes(PSY.FixedForcedOutage, sys))
```

!!! note
    
    On a reduced network a contingency branch may be an aggregated equivalent rather than a
    single `System` component; the attach step applies only to flowgates whose contingency
    branch is a real branch in `sys`.

!!! note
    
    When the `LODF` was built with a sparsification tolerance (`LODF(sys; tol = ...)`) entries
    below that tolerance were permanently dropped from the stored matrix. The radial/meshed
    classification and the breadth/magnitude ranking both run on this stored data, so
    `radial_tol` is only meaningful when it is at least as large as the build tolerance, and
    breadth/magnitude reflect only the surviving entries.
