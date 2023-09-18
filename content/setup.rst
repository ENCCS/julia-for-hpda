Installing packages
-------------------

A number of Julia packages is used in this lesson. These can be installed on-the-fly 
during a workshop, but you can also follow the instructions below to install all packages 
in your global Julia environment.

Copy-paste the following text into a file called Project.toml, which 
you can for example place under a new directory `julia` in your home directory:

.. code-block:: toml

   name = "Julia-for-HPC"
   [deps]
   BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
   Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
   LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
   MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
   MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
   Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
   Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
   SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
   StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

Then open a Julia REPL and specify the location of the Project.toml file:

.. code-block:: console

   $ julia --project=/path/to/Project.toml

Then copy-paste the following code into your Julia session:

.. code-block:: julia

   using Pkg
   Pkg.instantiate()

It could take a couple of minutes to download and install the packages. 
After it completes you should be able to list all installed packages:

.. code-block:: julia 

   Pkg.status()

   # Status `~/julia/Project.toml`
   #   [6e4b80f9] BenchmarkTools v1.3.1
   #   [da04e1cc] MPI v0.20.3
   #   [3da0fdf6] MPIPreferences v0.1.6
   #   [91a5bcdd] Plots v1.35.6
   #   [90137ffa] StaticArrays v1.5.9
   #   [8ba89e20] Distributed
   #   [37e2e46d] LinearAlgebra
   #   [9abbd945] Profile
   #   [1a1011a3] SharedArrays   
