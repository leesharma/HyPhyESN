# HyPhyESN

Class project for CMSC 727

## Getting Started

### Project Dependencies

* Julia 1.5 ([install link](https://julialang.org/downloads/))

### In the REPL

You can use the interactive Julia REPL either through your terminal or in Juno.
To experiment in the REPL, you need to first activate the project environment:

```
$ cd $(project_root)            # move to the HyPhyESN root directory
$ julia                         # start the Julia REPL

   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.5.2 (2020-09-23)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> ]                        # open the package manager context

(@v1.5) pkg>  activate .        # activate our Project.toml
(HyPhyESN) pkg> instantiate     # instantiate the project (adding deps, etc.)
(HyPhyESN) pkg> st              # print project status (should show deps)
  [1dea7af3] OrdinaryDiffEq v5.43.0
  [65888b18] ParameterizedFunctions v5.6.0
  [7c2d2b1e] ReservoirComputing v0.6.1


# optional, will save start-up time later
(HyPhyESN) pkg> precompile
Precompiling HyPhyESN...
.
.
.
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
