@testset "Test Ybus Matrix" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses_5 = nodes5()
    branches_5 = branches5(buses_5)
    sys_5 = System(100.0)
    for b in buses_5
        add_component!(sys_5, b)
    end
    for br in branches_5
        add_component!(sys_5, br)
    end
    buses_14 = nodes14()
    branches_14 = branches14(buses_14)
    sys_14 = System(100.0)
    for b in buses_14
        add_component!(sys_14, b)
    end
    for br in branches_14
        add_component!(sys_14, br)
    end
    Ybus5 = Ybus(sys_5)

    I, J, V = findnz(Ybus5.data)
    indices = collect(zip(I, J))

    for i in indices
        @test isapprox(Ybus5[i[1], i[2]], Ybus5_matpower[i[1], i[2]], atol = 1e-2)
    end
    Ybus14 = Ybus(sys_14)
    I, J, V = findnz(Ybus14.data)
    indices = collect(zip(I, J))

    for i in indices
        @test isapprox(Ybus14[i[1], i[2]], Ybus14_matpower[i[1], i[2]], atol = 1e-2)
    end

    bus5 = get_component(Bus, sys, "nodeE")
    set_number!(bus5, 10)
    Y5NS = Ybus(sys)
    @test isapprox(Y5NS[10, 4], -3.3336667 + 33.336667im, atol = 1e-4)

    Y5NS = Ybus(sys_5)
    for buf in Bu5NS_ids
        for but in Bu5NS_ids
            @test isapprox(Y5NS[buf, but], Ybus5_matpower[buf, but], atol = 1e-3)
        end
    end

    @test isapprox(
        Ybus5[buses_5[1], buses_5[2]],
        (-3.5234840209999647 + 35.234840209999646im),
        atol = 1e-4,
    )

    t_sys5_re = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    # Make 2 islands. Island 1: 1 - 5. Island 2: 2, 3 ,4
    remove_component!(Line, t_sys5_re, "1")
    remove_component!(Line, t_sys5_re, "2")
    remove_component!(Line, t_sys5_re, "6")
    @test length(Ybus(t_sys5_re).subnetwork_axes) == 2

    t2_sys5_re = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    # Remove lines. Don't cause islands
    remove_component!(Line, t2_sys5_re, "3")
    remove_component!(Line, t2_sys5_re, "5")
    @test length(Ybus(t2_sys5_re).subnetwork_axes) == 1

    sys_3bus = PSB.build_system(PSB.PSYTestSystems, "psse_3bus_gen_cls_sys")
    bus_103 = PSY.get_component(PSY.ACBus, sys_3bus, "BUS 3")
    fix_shunt = FixedAdmittance("FixAdm_Bus3", true, bus_103, 0.0 + 0.2im)
    add_component!(sys_3bus, fix_shunt)
    Ybus3 = Ybus(sys_3bus)
    I, J, V = findnz(Ybus3.data)
    indices = collect(zip(I, J))
    for i in indices
        @test isapprox(Ybus3.data[i[1], i[2]], Ybus3_matpower[i[1], i[2]], atol = 1e-5)
    end
end

@testset "Test modification of units base when constructing Ybus" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    PSY.set_units_base_system!(sys, "NATURAL_UNITS")
    ybus = @test_logs (
        :warn,
        r"Setting the system unit base from NATURAL_UNITS to SYSTEM_BASE for matrix construction",
    ) match_mode = :any Ybus(sys)
end
