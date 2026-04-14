@testset "Ybus - ACTIVSg10k" begin
    sys = build_system(
        MatpowerTestSystems,
        "matpower_ACTIVSg10k_sys",
    )
    matpower_cols =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_10k_cols.csv"), Int)
    matpower_rows =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_10k_rows.csv"), Int)
    matpower_vals_re =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_10k_vals_re.csv"), real(YBUS_ELTYPE))
    matpower_vals_im =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_10k_vals_im.csv"), real(YBUS_ELTYPE))
    ybus_pnm = Ybus(sys)
    @test nnz(ybus_pnm.data) ==
          length(filter(!iszero, matpower_vals_re .+ im .* matpower_vals_im))
    for (col, row, val_re, val_im) in
        zip(matpower_cols, matpower_rows, matpower_vals_re, matpower_vals_im)
        @test isapprox(ybus_pnm.data[row, col], Complex(val_re, val_im))
    end
end

@testset "Ybus - CATS" begin
    sys = build_system(MatpowerTestSystems, "matpower_cats")
    matpower_cols =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_cats_cols.csv"), Int)
    matpower_rows =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_cats_rows.csv"), Int)
    matpower_vals_re =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_cats_vals_re.csv"), real(YBUS_ELTYPE))
    matpower_vals_im =
        readdlm(joinpath(TEST_DATA_DIR, "ybus_cats_vals_im.csv"), real(YBUS_ELTYPE))
    ybus_pnm = Ybus(sys)
    @test nnz(ybus_pnm.data) ==
          length(filter(!iszero, matpower_vals_re .+ im .* matpower_vals_im))
    for (col, row, val_re, val_im) in
        zip(matpower_cols, matpower_rows, matpower_vals_re, matpower_vals_im)
        @test isapprox(ybus_pnm.data[row, col], Complex(val_re, val_im))
    end
end

@testset "Ybus - 5bus - transformers" begin
    sys = build_system(MatpowerTestSystems, "matpower_case5_transformer")
    transformer_connected_buses = [1, 2, 3, 4, 5]
    matpower_cols =
        readdlm(
            joinpath(TEST_DATA_DIR, "ybus_case5_transformers_cols.csv"),
            Int,
        )
    matpower_rows =
        readdlm(
            joinpath(TEST_DATA_DIR, "ybus_case5_transformers_rows.csv"),
            Int,
        )
    matpower_vals_re =
        readdlm(
            joinpath(TEST_DATA_DIR, "ybus_case5_transformers_vals_re.csv"),
            real(YBUS_ELTYPE),
        )
    matpower_vals_im =
        readdlm(
            joinpath(TEST_DATA_DIR, "ybus_case5_transformers_vals_im.csv"),
            real(YBUS_ELTYPE),
        )
    ybus_pnm = Ybus(sys)
    @test nnz(ybus_pnm.data) ==
          length(filter(!iszero, matpower_vals_re .+ im .* matpower_vals_im))
    for (col, row, val_re, val_im) in
        zip(matpower_cols, matpower_rows, matpower_vals_re, matpower_vals_im)
        if col == 1 && row == 1
            @test isapprox(
                imag(ybus_pnm.data[row, col]) - val_im,
                real(YBUS_ELTYPE)((3.126 / 2) / 1.3^2),
            )
            continue
        end
        if col == 2 && row == 2
            @test isapprox(
                imag(ybus_pnm.data[row, col]) - val_im,
                real(YBUS_ELTYPE)(1.852 / 2),
            )
            continue
        end
        if col == 3 && row == 3
            @test isapprox(
                imag(ybus_pnm.data[row, col]) - val_im,
                real(YBUS_ELTYPE)((0.674 / 2) / 1.5^2 - 1.852 / 2),
            )
            continue
        end
        if col == 4 && row == 4
            @test isapprox(
                imag(ybus_pnm.data[row, col]) - val_im,
                real(YBUS_ELTYPE)(-0.674 / 2),
            )
            continue
        end
        if col == 5 && row == 5
            @test isapprox(
                imag(ybus_pnm.data[row, col]) - val_im,
                real(YBUS_ELTYPE)(-3.126 / 2),
            )
            continue
        end
        @test isapprox(ybus_pnm.data[row, col], Complex(val_re, val_im); atol = 1e-5)
    end
    N_1_5 = get_tap(get_component(TapTransformer, sys, "bus-1-bus-5-i_3"))
end
