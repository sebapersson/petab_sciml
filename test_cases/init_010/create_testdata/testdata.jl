#=
    Hard-coded likelihood and simulated values for test case 001
=#

using YAML
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# Read neural net parameters, and assign to x, and then set test-value for layer 1
x = deepcopy(p_ode)
set_ps_net!(x.p_net1, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"), nn_model)

## Write test values
# YAML problem file
solutions = Dict(
    :test => Dict(
    :mean => Dict(
        :ps_files => Dict(
            :net1 => "net1_ps_mean_ref.hf5"),
        :tol => 2e-2,
        :nsamples => 50000),
    :variance => Dict(
        :ps_files => Dict(
            :net1 => "net1_ps_var_ref.hf5"),
        :tol => 2e-2,
        :nsamples => 50000)))
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Parameter values
nsamples = 500000
xnet1 = [deepcopy(x.p_net1) .* 0.0 for _ in 1:nsamples]
for i in 1:nsamples
    for layerid in keys(xnet1[i])
        for parameter_id in keys(xnet1[i][layerid])
            _ps = xnet1[i][layerid][parameter_id]
            @views xnet1[i][layerid][parameter_id] .= glorot_normal(rng, Float64, size(_ps)...; gain = 1.0)
        end
    end
end
xnet1_mean = sum([x for x in xnet1]) ./ nsamples
xnet1_var = sum([(x - xnet1_mean).^2 for x in xnet1]) ./ (nsamples - 1)
nn_ps_to_h5(nn_model, xnet1_mean, joinpath(@__DIR__, "..", "net1_ps_mean_ref.hf5"))
nn_ps_to_h5(nn_model, xnet1_var, joinpath(@__DIR__, "..", "net1_ps_var_ref.hf5"))

## Write PEtabProblem files
mapping_table = DataFrame(petabEntityId = ["prey", "predator", "gamma"],
                          modelEntityId = ["net1.input1", "net1.input2", "net1.output1"])
CSV.write(joinpath(@__DIR__, "..", "petab", "mapping_table.tsv"), mapping_table; delim = '\t')
problem_yaml = Dict(
    :format_version => 2,
    :parameter_file => "parameters_ude.tsv",
    :problems => [Dict(
        :condition_files => ["conditions.tsv"],
        :measurement_files => ["measurements.tsv"],
        :observable_files => ["observables.tsv"],
        :model_files => Dict(
            :lv_ude => Dict(
                :language => "sbml",
                :location => "lv.xml")),
        :mapping_files => ["mapping_table.tsv"])],
    :extensions => Dict(
        :petab_sciml => Dict(
            :net1 => Dict(
                :file => "net1.yaml",
                :parameters => "net1_ps.h5",
                :hybridization => Dict(
                    :input => "ode",
                    :output => "ode")))))
YAML.write_file(joinpath(@__DIR__, "..", "petab", "problem_ude.yaml"), problem_yaml)
