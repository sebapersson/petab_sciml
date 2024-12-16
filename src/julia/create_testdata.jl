using Lux, CSV, DataFrames, StableRNGs, YAML, HDF5

include(joinpath(@__DIR__, "helper.jl"))

dirtests = joinpath(@__DIR__, "..", "..", "test_cases")
for i in 1:51
    @info "Creating result data test-case $i"
    test_case = i < 10 ? "net_00$i" : "net_0$i"
    dirtest = joinpath(dirtests, test_case)
    include(joinpath(dirtest, "create_testdata", "net.jl"))
end
