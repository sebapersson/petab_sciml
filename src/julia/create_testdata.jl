using Lux, CSV, DataFrames, StableRNGs, YAML

include(joinpath(@__DIR__, "lux_layers.jl"))
include(joinpath(@__DIR__, "helper.jl"))

dirtests = joinpath(@__DIR__, "..", "..", "test_cases")
for i in 1:41
    @info "Creating result data test-case $i"
    if i < 10
        test_case = "net_00$i"
    else
        test_case = "net_0$i"
    end
    dirtest = joinpath(dirtests, test_case)
    include(joinpath(dirtest, "create_testdata", "net.jl"))
end
