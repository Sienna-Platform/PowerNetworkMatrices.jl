# The DC Power Flow Approximation

Many network matrices in PowerNetworkMatrices.jl rely on the DC power flow
approximation. It linearizes the AC power flow equations into a purely
algebraic relationship between active-power injections and voltage angles,
which is what makes the sensitivity matrices (`PTDF`, `LODF`) fast to build and
cheap to reason about.

### Assumptions

 1. **Voltage magnitude**: all bus voltages are approximately 1.0 per unit
 2. **Small angles**: voltage angle differences are small (< 15°), so
    ``\sin(\theta_i - \theta_j) \approx \theta_i - \theta_j``
 3. **Resistance**: line resistance is negligible compared to reactance
 4. **Active power**: only active power flows are considered

Under these assumptions the branch flow becomes a linear function of the angle
difference, ``P_{ij} \approx (\theta_i - \theta_j)/X_{ij}``, and the whole
network collapses to the susceptance-weighted graph Laplacian that the
[`BA_Matrix`](@ref) and [`ABA_Matrix`](@ref) encode.

### When the DC approximation works well

  - Transmission systems (high voltage)
  - Normal operating conditions
  - Security and market analysis
  - Planning studies

### When to be cautious

  - Distribution systems (high R/X ratios)
  - Large angle differences
  - Voltage-constrained systems
  - Detailed reactive power analysis
