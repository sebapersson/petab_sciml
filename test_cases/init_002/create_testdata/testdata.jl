#=
    Hard-coded likelihood and simulated values for test case 001
=#

using FiniteDifferences, YAML
include(joinpath(@__DIR__, "models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "julia", "helper.jl"))
Random.seed!(123)

# Read neural net parameters, and assign to x, and then set test-value for layer 1
x = deepcopy(p_ode)
set_ps_net!(x.p_net1, joinpath(@__DIR__, "..", "petab", "net1_ps.hf5"), nn_model)
x.p_net1.layer1.weight .= 0.0

## Write test values
# YAML problem file
solutions = Dict(:tol_ps => 1e-3,
                 :nsimulate => 1,
                 :ps_files =>
                 Dict(:net1 => "net1_ps_ref.hf5"))
YAML.write_file(joinpath(@__DIR__, "..", "solutions.yaml"), solutions)
# Parameter values
nn_ps_to_h5(nn_model, x.p_net1, joinpath(@__DIR__, "..", "net1_ps_ref.hf5"))

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
