## mtk model
@mtkmodel ECM begin
    @parameters begin
        Q = 4.8
        R0 = 0.5e-3
        R1 = 0.5e-3
        τ1 = 600.0
        R2 = 0.5e-3
        τ2 = 3600.0
    end
    @variables begin
        i(t)
        v(t)
        vr(t)
        v1(t) = 0.0
        v2(t) = 0.0
        ocv(t)
        soc(t)
    end
    @structural_parameters begin
        focv
        fi
    end
    @equations begin
        D(soc) ~ i / (Q * 3600.0)
        D(v1) ~ -v1 / τ1 + i * (R1 / τ1)
        D(v2) ~ -v2 / τ2 + i * (R2 / τ2)
        vr ~ i * R0
        ocv ~ focv(soc)
        i ~ fi(t)
        v ~ ocv + vr + v1 + v2
    end
end


## fit model
function loss(u, m)
    (; ode, ecm, df) = m
    (; u0, p) = u
    ps = replace(Tunable(), parameter_values(ode), p)
    prob = remake(ode; u0, p=ps)

    sol = solve(prob, Tsit5(); saveat=df.t)
    # https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/training_tips/divergence/#Handling-Divergent-and-Unstable-Trajectories
    if sol.retcode == ReturnCode.Success
        v = sol[ecm.v]
        return sum(abs2, v - df.v)
    else
        return Inf
    end
end


function fit_ecm_lbfgs(df, focv, fi)
    # build model
    @mtkbuild ecm = ECM(; focv, fi)

    tspan = (df[begin, :t], df[end, :t])
    ode = ODEProblem(ecm, [ecm.soc => 0.5], tspan, [])

    # _getu = getu(ode, variable_symbols(ode))
    # _getp = getp(ode, parameter_symbols(ode))
    # setu! = setu(ode, variable_symbols(ode))
    # setp! = setp(ode, parameter_symbols(ode))

    # build optimization problem
    m = (; ode, ecm, df) # params of loss function
    p0 = ComponentArray((;
        u0=ode.u0,
        p=ode.p.tunable
    ))
    adtype = Optimization.AutoForwardDiff() # auto-diff framework
    optf = OptimizationFunction(loss, adtype)
    opt = OptimizationProblem(optf, p0, m)

    # optimize with multiple initial values, select best fit
    solutions = []
    for Q in (4.8, 4.4, 4.0, 3.6)
        p0.p[2] = Q
        opt = remake(opt; u0=p0)

        sol = solve(opt, LBFGS(); reltol=1e-4, show_trace=true)
        if sol.retcode == ReturnCode.Success
            push!(solutions, sol)
        end
    end
    opt_sol = argmin(sol -> sol.objective, solutions)

    (; u0, p) = opt_sol.u
    ode = remake(ode; u0, p, tspan)

    return (; ecm, ode)
end


function loss2(u, m)
    (; ode, ecm, df) = m
    u0 = @view u[1:3]
    p = @view u[4:end]
    ps = replace(Tunable(), parameter_values(ode), p)
    prob = remake(ode; u0, p=ps)

    sol = solve(prob, Tsit5(); saveat=df.t)
    # https://docs.sciml.ai/SciMLSensitivity/stable/tutorials/training_tips/divergence/#Handling-Divergent-and-Unstable-Trajectories
    if sol.retcode == ReturnCode.Success
        v = sol[ecm.v]
        return sum(abs2, v - df.v)
    else
        return Inf
    end
end

function fit_ecm(df, focv, fi, alg; kwargs...)
    # build model
    @mtkbuild ecm = ECM(; focv, fi)

    tspan = (df[begin, :t], df[end, :t])
    ode = ODEProblem(ecm, [ecm.soc => 0.5], tspan, [])

    # build optimization problem
    p0 = ComponentArray((;
        u0=ode.u0,
        p=ode.p.tunable
    ))
    lb = ComponentArray((;
        u0=[0.35, -0.1, -0.1],
        p=[0.0, 4.0, 0.0, 600, 0.0, 0.0],
    ))
    ub = ComponentArray((;
        u0=[0.5, 0.1, 0.1],
        p=[50e-3, 4.9, 50e-3, 6000, 50e-3, 600],
    ))

    m = (; ode, ecm, df) # params of loss function
    optf = OptimizationFunction(loss2)
    prob = OptimizationProblem(optf, p0, m; lb, ub)

    # solve optimization and recover results
    sol = solve(prob, alg; kwargs...)
    # (; u0, p) = sol.u
    u0 = sol.u[1:3]
    p = sol.u[4:end]

    ode = remake(ode; u0, p, tspan)

    return (; ecm, ode)
end

