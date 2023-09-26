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
   CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
   ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
   Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
   DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
   Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
   Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
   FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
   Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
   GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
   HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
   LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
   MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
   MLJDecisionTreeInterface = "c6f25543-311c-4c74-83dc-3ea6d1015661"
   MLJFlux = "094fc8d1-fd35-5302-93ea-dabda2abf845"
   MLJScikitLearnInterface = "5ae90465-5518-4432-b9d2-8a1def2f0cab"
   Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
   PrettyPrinting = "54e16d92-306c-5ea0-a30b-337be88ac337"
   PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
   RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
   StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"
   StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

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
   # [6e4b80f9] BenchmarkTools v1.3.2
   # [024491cd] BetaML v0.10.3
   # [336ed68f] CSV v0.10.11
   # [35d6a980] ColorSchemes v3.24.0
   # [5ae59095] Colors v0.12.10
   # [a93c6f00] DataFrames v1.6.1
   # [31c24e10] Distributions v0.25.100
   # [7a1cc6ca] FFTW v1.7.1
   # [587475ba] Flux v0.14.6
   # [38e38edf] GLM v1.9.0
   # [cd3eb016] HTTP v1.10.0
   # [b964fa9f] LaTeXStrings v1.3.0
   # [add582a8] MLJ v0.19.5
   # [c6f25543] MLJDecisionTreeInterface v0.4.0
   # [094fc8d1] MLJFlux v0.3.1
   # [5ae90465] MLJScikitLearnInterface v0.5.0
   # [91a5bcdd] Plots v1.39.0
   # [54e16d92] PrettyPrinting v0.4.1
   # [d330b81b] PyPlot v2.11.2
   # [ce6b1742] RDatasets v0.7.7
   # [860ef19b] StableRNGs v1.0.0
   # [2913bbd2] StatsBase v0.34.0
   # [ade2ca70] Dates

Activating your environment in VS Code
-------------------------------------

When using Visual Studio Code for julia development and using your own environment,
you need to activate it. Start a Julia REPL in VS Code, for example by runnig a script.
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

