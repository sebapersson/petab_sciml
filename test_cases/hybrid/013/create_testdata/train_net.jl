## Net1
# Reuse the net from test case 001 (as it takes ages to train)
cp(joinpath(@__DIR__, "..", "..", "001", "petab", "net1_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net1_ps.hdf5");
            force = true)

## Net2
# Reuse net2 from case 010
cp(joinpath(@__DIR__, "..", "..", "010", "petab", "net2_ps.hdf5"),
            joinpath(@__DIR__, "..", "petab", "net2_ps.hdf5");
            force = true)
