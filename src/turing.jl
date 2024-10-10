using Turing

using StaticArrays

using SciMLStructures: replace!

@model function ecm_mcmc(ecm, ode, t)
    # Priors
    ## params
    Q ~ Uniform(4.7, 4.9)
    R0 ~ Uniform(1e-3, 50e-3)
    R1 ~ Uniform(1e-3, 50e-3)
    R2 ~ Uniform(1e-3, 50e-3)
    τ1 ~ Uniform(10, 300)
    τ2 ~ Uniform(300, 3600)
    ## initial conditions
    soc0 ~ Uniform(0.35, 0.5)
    v1 ~ Uniform(-0.1, 0.1)
    v2 ~ Uniform(-0.1, 0.1)
    ## noise
    σ ~ Uniform(0.001, 0.01)

    # Model
    ## update params
    θ = @SVector[R1, Q, R2, τ2, R0, τ1] # TODO: improve parameter update
    u0 = @SVector[soc0, v1, v2]
    p = replace(Tunable(), parameter_values(ode), θ)
    prob = remake(ode; p, u0)
    ## simulate
    sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat=t)
    v̂ = sol[ecm.v]

    # Observations
    v ~ MvNormal(v̂, σ^2)

    return nothing
end

function fit_ecm_mcmc(df, focv, fi)

    # build ecm model
    @mtkbuild ecm = ECM(; focv, fi)
    tspan = (df[begin, :t], df[end, :t])
    ode = ODEProblem{false}(ecm, [ecm.soc => 0.5], tspan, [])

    # MCMC model + inference
    model = ecm_mcmc(ecm, ode, df.t) | (; v=df.v)

    chain = Turing.sample(model, NUTS(250, 0.1), 50)
    # chain = Turing.sample(model, NUTS(1000, 0.65), MCMCThreads(), 50, 4)

    return chain
end

@info "starting: $(now())"
chains = fit_ecm_mcmc(df, focv, fi)

let chain = chains #[2]
    p = [
        chain[:R1] |> mean,
        chain[:Q] |> mean,
        chain[:R2] |> mean,
        chain[:τ2] |> mean,
        chain[:R0] |> mean,
        chain[:τ1] |> mean
    ]
    p = replace(Tunable(), parameter_values(ode), p)
    u0 = @SVector[
        chain[:soc0] |> mean,
        chain[:v1] |> mean,
        chain[:v2] |> mean,
    ]
    σ = chain[:σ] |> mean
    prob = remake(ode; u0, p)

    sol = solve(prob, Tsit5(); saveat=df.t)

    v̂ = sol[ecm.v]

    RMSE = mean(abs2, v̂ - df.v) * 1e3
    MAXE = maximum(abs, v̂ - df.v) * 1e3
    @info "Model error (in mV)" RMSE MAXE

    fig, ax = lines(df.t, v̂)
    band!(ax, df.t, v̂ .+ σ, v̂ .- σ; alpha=0.5)
    lines!(ax, df.t, df.v)
    fig
end


