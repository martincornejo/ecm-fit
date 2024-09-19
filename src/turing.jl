using Turing



@model function ecm_mcmc(data, model)
    # Priors
    Q ~ Uniform(4.4, 4.9)
    R0 ~ Uniform(1e-3, 50e-3)
    R1 ~ Uniform(1e-3, 50e-3)
    R2 ~ Uniform(1e-3, 50e-3)
    τ1 ~ Uniform(10, 300)
    τ2 ~ Uniform(300, 3600)
    σ ~ InverseGamma(2, 3)
    p = [Q, R0, R1, R2, τ1, τ2]

    # simulate
    (; ecm, ode) = model
    prob = remake(ode; p)
    sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat=data.t)
    v = sol[ecm.v]

    # observations
    v ~ MvNormal(data.v, σ^2)
end


function fit_ecm_mcmc(df, focv, fi)

    @mtkbuild ecm = ECM(; focv, fi)
    tspan = (df[begin, :t], df[end, :t])
    ode = ODEProblem(ecm, [ecm.soc => 0.5], tspan, [])

    chain = sample(ecm_mcmc(df, (; ecm, ode)), NUTS(), MCMCSerial(), 1000)

    return chain
end


