## The DC Power Flow Approximation

Many network matrices in PowerNetworkMatrices.jl rely on the DC power flow approximation.

### Assumptions:

 1. **Voltage Magnitude**: All bus voltages are approximately 1.0 per unit
 2. **Small Angles**: Voltage angle differences are small (< 15°)
 3. **Resistance**: Line resistance is negligible compared to reactance
 4. **Active Power**: Only active power flows are considered

### When DC Approximation Works Well:

  - Transmission systems (high voltage)
  - Normal operating conditions
  - Security and market analysis
  - Planning studies

### When to Be Cautious:

  - Distribution systems (high R/X ratios)
  - Large angle differences
  - Voltage-constrained systems
  - Detailed reactive power analysis
