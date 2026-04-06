const DC_ABA_Matrix_Factorized = ABA_Matrix{
    Tuple{Vector{Int64}, Vector{Int64}},
    Tuple{Dict{Int64, Int64}, Dict{Int64, Int64}},
    KLU.KLUFactorization{Float64, Int64},
}
const DC_ABA_Matrix_Unfactorized = ABA_Matrix{
    Tuple{Vector{Int64}, Vector{Int64}},
    Tuple{Dict{Int64, Int64}, Dict{Int64, Int64}},
    Nothing,
}
# VirtualPTDF can use various factorization types when extensions are loaded
const DC_vPTDF_Matrix = VirtualPTDF{
    Tuple{Vector{Tuple{Int, Int}}, Vector{Int64}},
    Tuple{Dict{Tuple{Int, Int}, Int64}, Dict{Int64, Int64}},
    <:LinearAlgebra.Factorization,
}
const DC_PTDF_Matrix = PTDF{
    Tuple{Vector{Int64}, Vector{Tuple{Int, Int}}},
    Tuple{Dict{Int64, Int64}, Dict{Tuple{Int, Int}, Int64}},
    Matrix{Float64},
}
const DC_BA_Matrix = BA_Matrix{
    Tuple{Vector{Int64}, Vector{Tuple{Int, Int}}},
    Tuple{Dict{Int64, Int64}, Dict{Tuple{Int, Int}, Int64}},
}
const AC_Ybus_Matrix = Ybus{
    Tuple{Vector{Int64}, Vector{Int64}},
    Tuple{Dict{Int64, Int64}, Dict{Int64, Int64}},
}
