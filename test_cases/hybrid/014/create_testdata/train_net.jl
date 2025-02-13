## Net1
# Reuse the net from test case 003
cp(joinpath(@__DIR__, "..", "..", "003", "petab", "net1_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5");
            force = true)

## Net2
# Reuse net2 from case 013
cp(joinpath(@__DIR__, "..", "..", "013", "petab", "net2_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net2_ps.hdf5");
            force = true)
