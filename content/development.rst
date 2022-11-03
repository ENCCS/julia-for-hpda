Developing in Julia
-------------------

.. _datascience_env:

.. exercise:: Creating a new environment

   In preparation for the next section on data science techniques in Julia, 
   create a new environment named `datascience` in a new directory, 
   activate it and install the following packages:

   - `DataFrames <https://github.com/JuliaData/DataFrames.jl>`_
   - `PalmerPenguins <https://github.com/devmotion/PalmerPenguins.jl>`_
   - `Plots <https://github.com/JuliaPlots/Plots.jl>`_
   - `StatsPlots <https://github.com/JuliaPlots/StatsPlots.jl>`_
   - `Flux <https://github.com/FluxML/Flux.jl>`_
   - `MLJ <https://alan-turing-institute.github.io/MLJ.jl/dev/>`_

   **Suggestion**: run this in a new VSCode window (*File* > *New Window*)
   because it will take some time and you can then continue working in your first window.

   .. solution::

      First create a new directory in a preferred location:
      
      .. code-block:: julia
         
         mkdir("datascience")

      Then add the packages:

      .. code-block:: julia

         # navigate to the datascience directory
         using Pkg
         Pkg.activate(".")
         Pkg.add(["DataFrames", "PalmerPenguins", "Plots", "StatsPlots", "Flux", "MLJ"])

