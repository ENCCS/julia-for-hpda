Installing packages
-------------------

A number of Julia packages is used in this lesson. These can be installed on-the-fly 
during a workshop, but you can also follow the instructions below to install all packages 
in your global Julia environment.

Copy-paste the following text into a file called Project.toml, which 
you can for example place under a new directory `julia` in your home directory:

.. code-block:: toml

   name = "Julia-for-HPDA"
   [deps]
   BetaML = "024491cd-cc6b-443e-8034-08ea7eb7db2b"
   CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
   ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
   Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
   DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
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
   # [024491cd] BetaML v0.5.5
   # [336ed68f] CSV v0.10.11
   # [35d6a980] ColorSchemes v3.21.0
   # [5ae59095] Colors v0.12.10
   # [a93c6f00] DataFrames v0.22.7
   # [7a1cc6ca] FFTW v1.7.1
   # [587475ba] Flux v0.12.10
   # [38e38edf] GLM v1.8.3
   # [cd3eb016] HTTP v0.9.17
   # [b964fa9f] LaTeXStrings v1.3.0
   # [add582a8] MLJ v0.16.5
   # [c6f25543] MLJDecisionTreeInterface v0.1.3
   # [094fc8d1] MLJFlux v0.1.10
   # [5ae90465] MLJScikitLearnInterface v0.1.10
   # [91a5bcdd] Plots v1.38.16
   # [54e16d92] PrettyPrinting v0.4.1
   # [d330b81b] PyPlot v2.11.1
   # [ce6b1742] RDatasets v0.7.7
   # [860ef19b] StableRNGs v1.0.0
   # [2913bbd2] StatsBase v0.33.21
