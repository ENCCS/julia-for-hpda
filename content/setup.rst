Installing packages
===================

A number of Julia packages are used in this lesson. These can be installed on-the-fly
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
   CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
   ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
   DataDrivenDiffEq = "2445eb08-9709-466a-b3fc-47e12bd697a2"
   DataDrivenSparse = "5b588203-7d8b-4fab-a537-c31a7f73f46b"
   DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
   Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
   FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
   Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
   GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
   HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
   IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
   Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
   JLD = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
   JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
   JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
   JSONTables = "b9914132-a727-11e9-1322-f18e41205b0b"
   LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
   LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
   Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
   MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
   MLJDecisionTreeInterface = "c6f25543-311c-4c74-83dc-3ea6d1015661"
   MLJScikitLearnInterface = "5ae90465-5518-4432-b9d2-8a1def2f0cab"
   ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
   Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
   OptimizationOptimJL = "36348300-93cb-4f02-beb5-3c3902f8871e"
   OptimizationOptimisers = "42dfb2eb-d2b4-4451-abcd-913932933ac1"
   OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
   PalmerPenguins = "8b842266-38fa-440a-9b57-31493939ab85"
   Parquet2 = "98572fba-bba0-415d-956f-fa77e587d26d"
   Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
   RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
   SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
   Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
   StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
   StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
   Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

Then open a Julia REPL and specify the location of the Project.toml file:

.. code-block:: console

   $ julia --project=/path/to/Project.toml

Then copy-paste the following code into your Julia session:

.. code-block:: julia

   using Pkg
   Pkg.instantiate()

It can take a while (10-15 minutes) to download and install the packages and precompile the project.
After it completes you should be able to list all installed packages with ``Pkg.status()``, which will look something like this:

.. code-block:: text

   julia> Pkg.status()
   Status `~/julia-kurs-test/Project.toml`
   [6e4b80f9] BenchmarkTools v1.8.0
   [336ed68f] CSV v0.10.16
   [b0b7db55] ComponentArrays v0.15.37
   [2445eb08] DataDrivenDiffEq v1.15.0
   [5b588203] DataDrivenSparse v0.1.4
   [a93c6f00] DataFrames v1.8.2
   [31c24e10] Distributions v0.25.125
   [7a1cc6ca] FFTW v1.10.0
   [587475ba] Flux v0.16.10
   [38e38edf] GLM v1.9.4
   [cd3eb016] HTTP v1.11.0
   [7073ff75] IJulia v1.34.4
   [a98d9a8b] Interpolations v0.16.2
   [4138dd39] JLD v0.13.5
   [033835bb] JLD2 v0.6.4
   [0f8b85d8] JSON3 v1.14.3
   [b9914132] JSONTables v1.0.3
   [d3d80556] LineSearches v7.6.2
   [b2108857] Lux v1.31.4
   [add582a8] MLJ v0.23.2
   [c6f25543] MLJDecisionTreeInterface v0.4.4
   [5ae90465] MLJScikitLearnInterface v0.7.0
   [961ee093] ModelingToolkit v11.25.0
   [7f7a1694] Optimization v5.5.1
   [36348300] OptimizationOptimJL v0.4.14
   [42dfb2eb] OptimizationOptimisers v0.3.17
   [1dea7af3] OrdinaryDiffEq v7.0.0
   [8b842266] PalmerPenguins v0.1.4
   [98572fba] Parquet2 v0.2.33
   [91a5bcdd] Plots v1.41.6
   [ce6b1742] RDatasets v0.8.1
   [1ed8b502] SciMLSensitivity v7.108.0
   [10745b16] Statistics v1.11.1
   [2913bbd2] StatsBase v0.34.10
   [f3b207a7] StatsPlots v0.15.8
   [e88e6eb3] Zygote v0.7.10
   [37e2e46d] LinearAlgebra v1.12.0

Activating your environment in VS Code
--------------------------------------

Open VS Code from terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you run VS Code from a terminal you can activate your environment as follows.
Open a terminal and go to the directory where the Project.toml file resides.
Now start VS Code with ``code .``

Change environment in VS Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using Visual Studio Code for julia development and using your own environment,
you can activate it as follows. Start a Julia REPL in VS Code, for example by running a script.
Start the package mode by typing ``]``. Typically you will be in the standard environment v1.12:

.. code-block:: julia

   (@v1.12) pkg>

To activate another environment, type

.. code-block:: julia

   (@v1.12) pkg>activate path-to-projectfile

where you specify the path to the directory where you put your Project.toml file.

The same procedure applies when running Julia from the terminal and you want to switch
environments. For example, if you start Julia from the terminal by simply typing ``julia``
(without the ``--project`` argument) you will end up in the standard environment and can
switch environment as described above.

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

   (@v1.12) pkg> add Plots

To update all your packages, you can type ``up`` in the package mode in REPL:

.. code-block:: julia

   (@v1.12) pkg> up

Installing JupyterLab and a Julia kernel
----------------------------------------

One way to use Julia is through Jupyter notebooks.
Jupyter notebooks can be installed via the Python package manager ``pip``::

  pip install jupyterlab

Also, JupyterLab can be installed through the full
Anaconda distribution of Python packages or the minimal
Miniconda distribution.

To install Anaconda, visit
https://www.anaconda.com/products/individual, download an installer
for your operating system and follow the instructions. JupyterLab and
an IPython kernel are included in the distribution.

To install Miniconda, visit
https://docs.conda.io/en/latest/miniconda.html, download an installer
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