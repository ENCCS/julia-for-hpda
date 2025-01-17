Installing packages
===================

A number of Julia packages is used in this lesson. These can be installed on-the-fly 
during a workshop, but you can also follow the instructions below to install all packages 
in your global Julia environment.

Creating an environment (optional)
----------------------------------

Copy-paste the following text into a file called Project.toml, which 
you can for example place under a new directory `julia` in your home directory:

.. code-block:: toml

   name = "Julia-for-HPDA"
   [deps]
   BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
   BetaML = "024491cd-cc6b-443e-8034-08ea7eb7db2b"
   Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
   CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
   ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
   Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
   ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
   DataDrivenDiffEq = "2445eb08-9709-466a-b3fc-47e12bd697a2"
   DataDrivenSparse = "5b588203-7d8b-4fab-a537-c31a7f73f46b"
   DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
   DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
   Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
   DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
   Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
   DiffEqFlux = "aae7a2af-3d4f-5e19-a356-7da93b79d9d0"
   DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
   Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
   FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
   Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
   GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
   GLMNet = "8d5ece8b-de18-5317-b113-243142960cc6"
   JLD = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
   JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
   HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
   IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
   LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
   LIBSVM = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
   LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
   LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
   Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
   MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
   MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
   MLJDecisionTreeInterface = "c6f25543-311c-4c74-83dc-3ea6d1015661"
   MLJScikitLearnInterface = "5ae90465-5518-4432-b9d2-8a1def2f0cab"
   ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
   Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
   OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
   OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
   OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
   NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
   PalmerPenguins = "8b842266-38fa-440a-9b57-31493939ab85"
   Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
   PrettyPrinting = "54e16d92-306c-5ea0-a30b-337be88ac337"
   PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
   Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
   RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
   SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
   SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
   StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"
   Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
   StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
   Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
   VegaDatasets = "0ae4a718-28b7-58ec-9efb-cded64d6d5b4"
   VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a"

Then open a Julia REPL and specify the location of the Project.toml file:

.. code-block:: console

   $ julia --project=/path/to/Project.toml

Then copy-paste the following code into your Julia session:

.. code-block:: julia

   using Pkg
   Pkg.instantiate()

It could take a couple of minutes to download and install the packages. 
After it completes you should be able to list all installed packages:

.. code-block:: text

   Pkg.status()
   Status `~/julia/Project.toml`
   [6e4b80f9] BenchmarkTools v1.6.0
   [024491cd] BetaML v0.7.1
   [336ed68f] CSV v0.10.15
   [aaaa29a8] Clustering v0.15.8
   [35d6a980] ColorSchemes v3.27.1
   [5ae59095] Colors v0.12.11
   [b0b7db55] ComponentArrays v0.15.22
   [2445eb08] DataDrivenDiffEq v1.5.0
   [5b588203] DataDrivenSparse v0.1.2
   [a93c6f00] DataFrames v1.7.0
   [864edb3b] DataStructures v0.18.20
   [7806a523] DecisionTree v0.12.4
   [aae7a2af] DiffEqFlux v4.1.0
   [0c46a032] DifferentialEquations v7.15.0
   [b4f34e82] Distances v0.10.12
   [31c24e10] Distributions v0.25.116
   [7a1cc6ca] FFTW v1.8.0
   [587475ba] Flux v0.16.1
   [38e38edf] GLM v1.9.0
   [8d5ece8b] GLMNet v0.7.4
   [cd3eb016] HTTP v1.10.15
   [7073ff75] IJulia v1.26.0
   [4138dd39] JLD v0.13.5
   [682c06a0] JSON v0.21.4
   [b1bec4e5] LIBSVM v0.8.1
   [b964fa9f] LaTeXStrings v1.4.0
   [d3d80556] LineSearches v7.3.0
   [b2108857] Lux v1.5.1
   [f0e99cf1] MLBase v0.9.2
   [add582a8] MLJ v0.20.7
   [c6f25543] MLJDecisionTreeInterface v0.4.2
   [5ae90465] MLJScikitLearnInterface v0.7.0
   [961ee093] ModelingToolkit v9.60.0
   [b8a86587] NearestNeighbors v0.4.21
   [7f7a1694] Optimization v4.0.5
   [36348300] OptimizationOptimJL v0.4.1
   [42dfb2eb] OptimizationOptimisers v0.3.7
   [1dea7af3] OrdinaryDiffEq v6.90.1
   [8b842266] PalmerPenguins v0.1.4
   [91a5bcdd] Plots v1.40.9
   [54e16d92] PrettyPrinting v0.4.2
   [d330b81b] PyPlot v2.11.5
   [ce6b1742] RDatasets v0.7.7
   [1ed8b502] SciMLSensitivity v7.72.0
   [860ef19b] StableRNGs v1.0.2
   [10745b16] Statistics v1.11.1
   [2913bbd2] StatsBase v0.34.4
   [0ae4a718] VegaDatasets v2.1.1
   [112f6efa] VegaLite v3.3.0
   [e88e6eb3] Zygote v0.6.75
   [ade2ca70] Dates v1.11.0
   [37e2e46d] LinearAlgebra v1.11.0
   [9a3f8284] Random v1.11.0
   [2f01184e] SparseArrays v1.11.0

Activating your environment in VS Code
--------------------------------------

Open VS Code from terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you run VS Code from a terminal you can activate your enviroment as follows.
Open a terminal and go to the directory where the Project.toml file resides.
Now start VS Code with ``code .``

Change environment in VS Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using Visual Studio Code for julia development and using your own environment,
you can activate it as follows. Start a Julia REPL in VS Code, for example by runnig a script.
Start the package mode by typing ``]``. Typically you will be in the stardard environment v1.9:

.. code-block:: julia

   (@v1.9) pkg>

To activate another environment, type

.. code-block:: julia

   (@v1.9) pkg>activate path-to-projectfile

where you specify the path to the directory where you put your Project.toml file.

The same procedure applies when running Julia from the terminal and you want to switch
enviroments. For example, if you start Julia from the terminal by simply typing ``julia``
(without the ``--project`` argument) you will end up in the standard environment and can
switch enviroment as described above.

Manual installation and updates
-------------------------------

It is convenient also to add packages as they are needed.
This can be done in several ways. For instance, to install the package Plots
one may do as follows. Open the Julia REPL as above and type:

.. code-block:: julia

   using Pkg
   Pkg.add("Plots")

Alternatively we may enter the package mode in the REPL by typing ``]``
and then add the package:

.. code-block:: julia

   (@v1.9) pkg> add Plots

To update all your packages, you can type ``up`` in the package mode in REPL:

.. code-block:: julia

   (@v1.9) pkg> up

Installing JupyterLab and a Julia kernel
----------------------------------------

One way to use Julia is through Jupyter notebooks.
Jupyter notebooks can be installed via the Python package manager ``pip``::

  pip install jupyterlab

Also, JupyterLab can most easily be installed through the full
Anaconda distribution of Python packages or the minimal
Miniconda distribution.

To install Anaconda, visit
https://www.anaconda.com/products/individual , download an installer
for your operating system and follow the instructions. JupyterLab and
an IPython kernel are included in the distribution.

To install Miniconda, visit
https://docs.conda.io/en/latest/miniconda.html , download an installer
for your operating system and follow the instructions.  After
activating a ``conda`` environment in your terminal, you can install
JupyterLab with the command ``conda install jupyterlab``.

Add Julia to JupyterLab
^^^^^^^^^^^^^^^^^^^^^^^

To be able to use a Julia kernel in a Jupyter notebook you need to
install the ``IJulia`` Julia package. Open the Julia REPL and type::

  using Pkg
  Pkg.add("IJulia")

Create a Julia notebook
^^^^^^^^^^^^^^^^^^^^^^^

Now you should be able to open up a JupyterLab session by typing
``jupyter-lab`` in a terminal, and create a Julia notebook by clicking
on Julia in the JupyterLab Launcher or by selecting File > New > Notebook
and selecting a Julia kernel in the drop-down menu that appears.