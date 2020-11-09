# HyPhyESN

Class project for CMSC 727

## Getting Started

### Project Dependencies

* Julia 1.5 ([install link](https://julialang.org/downloads/))

### In the REPL

You can use the interactive Julia REPL either through your terminal or in Juno.
To experiment in the REPL, you need to first activate the project environment:

```bash
% cd $(project_root)                # move to the HyPhyESN root directory
% julia                             # start the Julia REPL
...

julia> ]                            # open the package manager context

(@v1.5) pkg>  activate .            # activate our Project.toml
(HyPhyESN) pkg> instantiate         # instantiate the project (adding deps, etc.)
(HyPhyESN) pkg> st                  # print project status (should show deps)
  [1dea7af3] OrdinaryDiffEq v5.43.0
  [65888b18] ParameterizedFunctions v5.6.0
  [7c2d2b1e] ReservoirComputing v0.6.1


# optional, will save start-up time later
(HyPhyESN) pkg> precompile
Precompiling HyPhyESN...
...
```

The `activate` step must be performed for every new Julia context (e.g. whenever you restart the REPL).
The `instantiate` and `precompile` steps only need to be performed once.

### In a File

Running a Julia file creates a new Julia context, so we need to explicitly activate our project environment.
You can do this by adding the following line to the head of your main `.jl` file:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

If you plan to run the file from a different folder, replace `"."` with the path to `Project.toml`'s parent folder.

### In Juno

You can set up the Juno console using the steps in ["In the REPL"](#in-the-repl) above.

If you plan to run from a file, use the steps in ["In a File"](#in-a-file) above.

### Verify Setup

You should be able to run this script without errors:

```bash
% julia main.jl
```
## Climate Forecast Models

The `./data` directory contains three variants of simplified climate forecast models with varying degrees of complexity.
These codes utilize the classical spectral atmosphere model developed in the NOAA Geophysical Fluid Dynamics Laboratory,
and implementations are provided by [JGCM](https://github.com/CliMA/IdealizedSpectralGCM.jl).

These spectral methods solve the primitive equations before any physical parameterization. This "spectral core" is used in
ALL numerical weather forecast models (albeit with some slight modifications). Parameterizations are then applied afterwards
to correct the solutions to sub-grid phenomena.

NOTE: I am still rather unclear on the details, and anything I say has a high chance of being incorrect.

Further details on these specific models can be found [here](https://www.gfdl.noaa.gov/idealized-spectral-models-quickstart/).

In short, `barotropic.jl` is the simplest model, implementing only vorticity equations for the evolution of a 2D non-divergent
flow on the surface of a sphere.

`shallow_water.jl` builds on this by adding in the equation for hydrostatic balance, to the approximation that the ratio of vertical scale
to horizontal scale is small.

`spectral_dynamics.jl` contains the full spectral method, solving all 6 primitive equations. It is very slow. I've modified it to easily
change the timescales and grid system.

All three of these output a simple plot to `./data/data_plots`. This output can be modified to suit our needs moving forward.
