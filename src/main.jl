using CSV
using DataFrames
using DataInterpolations

using Statistics

using OrdinaryDiffEq
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using SymbolicIndexingInterface: parameter_values, state_values, getu, getp, setu, setp
using SciMLStructures: Tunable, replace

using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using OptimizationMetaheuristics

import ComponentArrays: ComponentArray

using GLMakie


include("data.jl")
include("ecm.jl")

# load data
file = "data/2098LG_INR21700-M50L_SammyLGL13818NewFullCU.txt"
df = read_basytec(file)

profile = load_profile(df)

tt = 0:(3*24*3600.0)
df_train = sample_dataset(profile, tt)

focv = calc_pocv(df) # OCV curve lookup-table (function)
fi = profile.i # current profile lookup-table (function)

# fit models
ecm_lbfgs = fit_ecm_lbfgs(df_train, focv, fi)

ecm_eca = fit_ecm(df_train, focv, fi, ECA(); maxiters=100_000)
ecm_pso = fit_ecm(df_train, focv, fi, PSO(N=100); maxiters=100_000)

ecm_eca = fit_ecm_meta(df_train, focv, fi)
ecm_pso = fit_ecm_pso(df_train, focv, fi)

# evaluate models
sol_lbfgs = solve(ecm_lbfgs.ode, Tsit5(), saveat=df_train.t)
sol_eca = solve(ecm_eca.ode, Tsit5(), saveat=df_train.t) # simulate model
sol_pso = solve(ecm_pso.ode, Tsit5(), saveat=df_train.t)

v_lbfgs = sol_lbfgs[ecm_lbfgs.ecm.v]
v_eca = sol_eca[ecm_eca.ecm.v] # exctract terminal voltage
v_pso = sol_pso[ecm_pso.ecm.v]

mean(abs2, v_eca - df_train.v) # RMSE
mean(abs2, v_pso - df_train.v)

begin
    fig = Figure()
    ax = [Axis(fig[i, 1]) for i in 1:2]
    lines!(ax[1], df_train.t, df_train.v)
    lines!(ax[1], df_train.t, v_eca)
    lines!(ax[1], df_train.t, v_pso)
    lines!(ax[2], df_train.t, v_eca - df_train.v)
    lines!(ax[2], df_train.t, v_pso - df_train.v)
    linkxaxes!(ax...)
    fig
end
