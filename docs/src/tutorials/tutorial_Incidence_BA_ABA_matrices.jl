# # Incidence, BA and ABA matrices

# In this tutorial the `IncidenceMatrix`, `BA_matrix` and `ABA_matrix` are presented.
# The methods used for their evaluation, as well as how data is stored is shown in
# the following subsections.

# The matrices here presented are the building blocks for the computation of the PTDF and LODF matrices.

# ## IncidenceMatrix

# The `PowerNetworkMatrices` package defines the structure `IncidenceMatrix`, which
# store the Incidence Matrix of the considered system as well as the most relevant network data.

# At first, the `System` data is loaded.

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# Then the Incidence Matrix is computed as follows:

incidence_matrix = PNM.IncidenceMatrix(sys)

# The `incidence_matrix` variable is a structure of type `IncidenceMatrix`. Getter functions are available to access additional
# information.

# `axes` lists the row and column names. The row names are tuples of the arcs
# (from bus number, to bus number); the column names are the bus numbers:

PNM.get_axes(incidence_matrix)

# `data` is the incidence matrix data:

PNM.get_data(incidence_matrix)

# `lookup` is a dictionary linking the arc tuples and bus numbers with the row
# and column numbers, respectively:

PNM.get_lookup(incidence_matrix)

# `ref_bus_positions` is a set containing the positions of the reference buses.
# This represents the positions where to add the column of zeros. Please refer to the
# example in the BA matrix for more details:

PNM.get_ref_bus_position(incidence_matrix)

# Note that the number of columns is lower than the actual number of system buses since
# the column related to the reference bus is discarded.

# ## BA_Matrix

# The `BA_Matrix` is a structure containing the matrix coming from the product of the
# `IncidenceMatrix` and the diagonal matrix containing the impedance of the system's branches ("B" matrix).

# The `BA_Matrix` is computed as follows:

ba_matrix = PNM.BA_Matrix(sys)

# Note that the axes order matches the `IncidenceMatrix` (arcs x buses), but for computational considerations the raw data is the transposed matrix.

# The matrix data can similarly be accessed with getter functions:

PNM.get_data(ba_matrix)

# Note that the number of columns is lower than the actual number of system buses since
# the column related to the reference bus is discarded.

# To add the column of zeros related to the reference bus, it is necessary to use the
# information from `get_ref_bus_position`.

## assumes a single reference bus
ref_bus_position = first(PNM.get_ref_bus_position(ba_matrix))
new_ba_matrix = hcat(
    ba_matrix.data[:, 1:(ref_bus_position - 1)],
    zeros(size(ba_matrix, 1), 1),
    ba_matrix.data[:, ref_bus_position:end],
)

# However, trying to change the data field with a matrix of different dimension
# will result in an error.

try
    ba_matrix.data = new_ba_matrix
catch e
    showerror(stdout, e)
end

# ## ABA_Matrix

# The `ABA_Matrix` is a structure containing the matrix coming from the product of the
# `IncidenceMatrix` and the `BA_Matrix`.
# It features the same fields as the `IncidenceMatrix` and the `BA_Matrix`, plus the `K` one.
# The field `ABA_Matrix.K` stores the LU factorization matrices (using the
# methods contained in the package `KLU`).

# To evaluate the `ABA_Matrix`, the following command is sufficient:

aba_matrix = ABA_Matrix(sys);

# By default the LU factorization matrices are not computed, leaving the `K` field empty:

isnothing(aba_matrix.K)

# In case these are wanted, the keyword `factorize` must be true.

aba_matrix_with_LU = ABA_Matrix(sys; factorize = true);

#

aba_matrix_with_LU.K

# If the `ABA_Matrix` is already computed but the LU factorization was not performed, this can be done by considering the following command:

aba_matrix = factorize(aba_matrix);

#

aba_matrix.K

# The following command can then be used to check if the `ABA_Matrix` contains the LU factorization matrices:

is_factorized(aba_matrix)
