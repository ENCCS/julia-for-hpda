.. _data_science:

Data science and machine learning
=================================

.. questions::

   - Can I use Julia for machine learning?
   - What are the key steps in data preprocessing in Julia?
   - How can you handle missing data in Julia?
   - How can you save your current environment in Julia?  
   - What are some popular machine learning algorithms available in Julia?
   - How does Julia handle large datasets in machine learning?
   - How can you implement clustering in Julia?
   - What are some classification techniques available in Julia?
     
.. instructor-note::

   - 100 min teaching
   - 50 min exercises


Working with data
-----------------

In the Data Formats and Dataframes lesson, we explored a Julian approach
to manipulation and visualisation of data.


Here we will learn and clustering, classification, machine learning and deep learning with some toy examples. 


Download a dataset
^^^^^^^^^^^^^^^^^^

We start by downloading a dataset containing measurements 
of characteristic features of different penguin species.


.. figure:: img/lter_penguins.png
   :align: center

   Artwork by @allison_horst

.. exercise::
      
   To obtain the data we simply add the PalmerPenguins package.

   .. code-block:: julia

      using Pkg
      Pkg.add("PalmerPenguins")
      using PalmerPenguins


   As it was done in the Data Formats and Dataframes lesson, we can
   
   .. code-block:: julia
   
      dropmissing!(df)
   
The main features we are interested in for each penguin observation are 
`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` and `body_mass_g`.
What the first three features mean is illustrated in the picture below.

.. figure:: img/culmen_depth.png
   :align: center

   Artwork by @allison_horst


Saving the Current Setup
------------------------

There are several ways to save the current setup in Julia.
This section will cover three parts: saving the environment to
have reproducible code and saving data using CSV files or ``JLD``.

1. Saving the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

.. exercise::
   To check the current status of your Julia environment, you can use the status command in the package manager. 

   .. code-block:: julia

      using Pkg
      Pkg.status()

   .. code-block:: text
      
      Status `~/.julia/environments/v1.9/Project.toml`
         [336ed68f] CSV v0.10.11
         [aaaa29a8] Clustering v0.15.4
         [a93c6f00] DataFrames v1.6.1
         [682c06a0] JSON v0.21.4
         [8b842266] PalmerPenguins v0.1.4

   This will display the list of packages in the current environment along with their versions.

   To save the state of your environment, Julia uses two files: ``Project.toml`` and ``Manifest.toml``.
   The ``Project.toml`` file specifies the packages that you explicitly added to your environment,
   while the ``Manifest.toml`` file records the exact versions of these packages and all their dependencies1.

   When you add packages using ``Pkg.add()``, Julia automatically updates these files.
   Therefore, your environment’s state (i.e., the set of loaded packages) is automatically saved.
   ``Project.toml`` and ``Manifest.toml`` are located in the directory of your current Julia environment; in our case, ``~/.julia/environments/v1.9/``.

   If you want to replicate this environment on another machine or in another folder, you can do the following:

   1. Copy both ``Project.toml`` and ``Manifest.toml`` to the new location.
   2. In Julia, navigate to that folder and activate the environment using ``Pkg.activate(".")``.
   3. Use ``Pkg.instantiate()`` to download all the necessary packages.
   
   More information in section `Environments` at https://enccs.github.io/julia-intro/development/

2. Saving Data as a CSV File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As shown in the Data Formats and DataFrames lesson, a DataFrame can easily dumped into a CSV file using
the ``CSV.jl`` package, which also allows for reading tabular data.

.. exercise::

   You can use the CSV.jl package to save a DataFrame as a CSV file, which can be re-read later.

   .. code-block:: julia

         # using Pkg
         # Pkg.add("CSV")
         using CSV
         CSV.write("penguins.csv", df)

   And you can load it back with:

   .. code-block:: julia

         df = CSV.read("penguins.csv", DataFrame)

3. Saving Data Using JLD/JLD2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Another option is to use `JLD.jl <https://github.com/JuliaIO/JLD.jl>`_ 
   The ``JLD.jl`` package provides a way to save and load Julia variables while preserving native types.
   It is based on HDF5, a cross-platform, multi-language data storage format most frequently used for scientific data.
   However, it is written in pure Julia and does not require any of the original C HDF5 implementation.

   The ``JLD`` package can be imported in the usual way:

   .. code-block:: julia

      using Pkg
      Pkg.add("JLD")

   A DataFrame can be saved to file in the following way:

   .. code-block:: julia

      using JLD
      save("penguins.jld", "df", df)

   Here we're saving ``df`` as "df" within ``penguins.jld``. You can load this DataFrame back in with:

   .. code-block:: julia

      df = load("penguins.jld", "df")

   This will return the DataFrame ``df`` from the file and assign it back to ``df``.
   In the past years, the ``JLD2.jl`` package came forward as an alternative to ``JLD``. It 
   is also based on HDF5 and can read h5 files saved by other HDF5 implementations. It exposes an interface
   similar to ``JLD`` with  ``save()`` and ``load()`` functions, but the more user-friendly function ``jldsave()``
   is also available:

   .. code-block:: julia
    
      using JLD2
      jldsave("penguins.jld2"; df) # This is equivalent to the save command above
      df = load("penguins.jld2", "df")

   Moreover, a ``jldopen()`` function provides a file-like interface. More information can be found
   `here <https://github.com/JuliaIO/JLD2.jl>`__.

Machine learning
----------------

Machine learning (ML) is a branch of artificial intelligence (AI) and computer
science that focuses on the use of data and algorithms to imitate the way that
humans learn, gradually improving its accuracy. It is an umbrella term for
solving problems for which development of algorithms by human programmers would
be cost-prohibitive, and instead the problems are solved by helping machines
"discover" patterns and algorithms to deal with data.
Classical machine learning algorithms include (non-)linear regression, logistic
regression,support vector machines (SVM), k-Nearest Neigbours, xgboost and many
others, spanning supervised learning (where the algorithm is shown examples of
what it should do), unsupservised learning (where it learns autonomously) and
reinforcement learning (where a reward policy is used to "teach").
Deep learning, which will be discussed in more depth in a later section, is a
type of machine learning algorithm.


References:

- What is Machine Learning? – IBM. https://www.ibm.com/topics/machine-learning 
- Machine learning - Wikipedia. https://en.wikipedia.org/wiki/Machine_learning

Machine learning in Julia
^^^^^^^^^^^^^^^^^^^^^^^^^

Despite being a relatively new language, Julia already has a strong and rapidly
expanding ecosystem of libraries for machine learning and deep learning. An
important advantage of Julia for ML is that it is possible to extract very good
performance by writing pure Julia code, without resorting to backends written
in compiled languages.
A distinct feature of the Julian ML ecosystem is a well-developed stack for
`"scientific machine learning" (SciML) <https://sciml.ai/>`_ (a.k.a.
physics-informed learning), which is a flavour of machine learning that
incorporates physics (ODEs, PDEs...) into the learning process instead of
relying only on data. SciML relies heavily on `automatic differentiation` - the
ability to automatically compute derivatives of any function and thus
incorporate it into predictive models.

Traditional machine learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia has packages for traditional (non-deep) machine learning:

- `ScikitLearn.jl <https://scikitlearnjl.readthedocs.io/en/latest/>`_ is a port
  of the popular Python package. 
- `MLJ.jl
  <https://alan-turing-institute.github.io/MLJ.jl/dev/>`_ provides a common
  interface and meta-algorithms for selecting, tuning, evaluating, composing
  and comparing over 150 machine learning models.
- `Machine Learning · Julia Packages
  <https://juliapackages.com/c/machine-learning/>`_ 
  lists various Julia packages related to machine learning, such as MLJ.jl,
  Knet.jl, TensorFlow.jl, DiffEqFlux.jl, FastAI.jl, ScikitLearn.jl, and many
  more. You can browse the packages by their popularity, alphabetical order, or
  update date. Each package has a brief description and a link to its GitHub
  repository. 
- `AI · Julia Packages <https://www.juliapackages.com/c/ai>`_ lists Julia
  packages related to Artificial Intelligence broadly

We will use a few utility functions from ``MLJ.jl`` in our deep learning 
exercise below, so we will need to add it to our environment:

.. code-block:: julia

   using Pkg
   Pkg.add("MLJ")


Clustering and Classification
-----------------------------

In this lesson, we will be exploring the use of Julia for HPDA in a Jupyter
notebook environment within Visual Studio Code (VSCode).

To set up your environment, you can follow the instructions provided in the
`JuliaIntro lesson
<https://enccs.github.io/julia-intro/setup/#optional-installing-jupyterlab-and-a-julia-kernel>`_.
This guide will walk you through the process of installing Julia, setting up
JupyterLab, and adding a Julia kernel. Jupyter notebooks offer an interactive
computing environment where you can combine code execution, rich text,
mathematics, plots, and rich media.

Once your environment is set up, you can start using Julia in Jupyter notebooks
within VSCode. This setup provides a powerful interface for writing and
debugging your code. It also allows you to easily visualize your data and
results.

After setting up your environment, we will dive into the adapted lessons about
Clustering and Classification from the `Julia MOOC on Julia Academy
<https://juliaacademy.com/>`_. These lessons provide comprehensive tutorials on
various topics in Julia. By following these lessons, you will gain a deeper
understanding of how to use Julia for high-performance data analysis.

Clustering notebook:
https://github.com/ENCCS/julia-for-hpda/blob/main/notebooks/Clustering.ipynb

Classification notebook:
https://github.com/ENCCS/julia-for-hpda/blob/main/notebooks/Classification.ipynb

Deep learning
-------------

`Deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_ is a subset of
ML. Neural networks are a particular type of ML approach which tries to loosely
mimic the functioning of a human brain and consists of connected computational
units called *neurons*; each neuron has one or more inputs and (mostly) performs three fundamental operations:
- Take a weighted sum of the inputs with a vector of weights 
- Add an extra constant weight (**bias**)
- Apply an (usually non-linear) **activation function** to this weighted sum.

In essence, given a input vector :math:`(x_1,x_2,...,x_N)`, a neuron computes:

.. math:: 

    y = f(\sum_i(x_iw_i) + b)

where *f* is the activation function, :math:`w_i` are the weights and *b* is the
bias. 

These neural networks attempt to simulate the behavior of the human brain —
albeit far from matching its ability — allowing it to “learn” from large
amounts of data. Deep learning drives many AI applications and services that
improve automation, performing analytical and physical tasks without human
intervention. Large language models and other generative AI models are very
large neural networks with specific architectures. 

For more detailed information, the `Intro to deep learning course
<https://enccs.github.io/deep-learning-intro/>`_ gives a good overview of the
building blocks of a neural network. 

At the time of writing (2026), two main frameworks are used when dealing with
neural networks in Julia: Flux.jl and Lux.jl.


`Flux.jl <https://fluxml.ai/>`_ comes "batteries-included" with many useful
tools built in, but also enables the user to write own Julia code for DL
components.

- Flux has relatively few explicit APIs for features like regularisation or
  embeddings. Core components are available, but there are few rigid,
  high-level APIs. Things like regularisation and embeddings are written using
  standard Julia code patterns 
- All of Flux is straightforward Julia code and can be inspected/extended (no DSLs)
- Flux works well with other Julia libraries, like dataframes, images and
  differential equation solvers. One can build complex data processing
  pipelines that integrate Flux models.

`Lux.jl <https://lux.csail.mit.edu/stable/>`_, on the other hand, is a newer
framework that has a more functional programming style. While the building
blocks are similar (layers, activation functions,etc.), the models are pure
functions and the state is passed around as an argument. While this may look
clunkier at first, it does have some advantages when trying to optimise code,
and models can be composed, transformed and differentiated with standard Julia
tools without special abstractions.

Generally speaking, Lux shines when integration with SciML is needed, or any
time neural networks are embedded in larger scientific computing applications,
and new developments tend to land in Lux first. Conversely, Flux is the more mature,
general purpose deep learning framework. From a Python perspective, Flux feels
more like PyTorch/Keras, whereas Lux is more akin to Jax/Flax.

To install Flux:

.. code-block:: julia

   using Pkg
   Pkg.add("Flux")


.. exercise:: Training a deep neural network to classify penguins

   To train a model we need four things:

   - A collection of data points that will be provided to the objective
     function.
   - A objective (cost or loss) function, that evaluates how well a model 
     is doing given some input data.
   - The definition of a model and access to its trainable parameters.
   - An optimiser that will update the model parameters appropriately.

   First we import the required modules and load the data:

   .. code-block:: julia

      using Flux
      using MLJ: partition, ConfusionMatrix
      using DataFrames
      using PalmerPenguins

      table = PalmerPenguins.load()
      df = DataFrame(table)
      dropmissing!(df)

   We can now preprocess our dataset to make it suitable for training a network:

   .. code-block:: julia

      # select feature and label columns
      X = select(df, Not([:species, :sex, :island]))
      Y = df[:, :species]
      
      # split into training and testing parts
      (xtrain, xtest), (ytrain, ytest) = partition((X, Y), 0.8, shuffle=true, rng=123, multi=true)
      
      # use single precision and transpose arrays
      xtrain, xtest = Float32.(Array(xtrain)'), Float32.(Array(xtest)')
      
      # one-hot encoding
      ytrain = Flux.onehotbatch(ytrain, ["Adelie", "Gentoo", "Chinstrap"])
      ytest = Flux.onehotbatch(ytest, ["Adelie", "Gentoo", "Chinstrap"])
      
      # count penguin classes to see if it's balanced
      sum(ytrain, dims=2)
      sum(ytest, dims=2)

   Next up is the loss function which will be minimized during the training.
   We also define another function which will give us the accuracy of the model:

   .. code-block:: julia

      # we use the cross-entropy loss function typically used for classification
      loss(model, x, y) = Flux.crossentropy(model(x), y)

      # onecold (opposite to onehot) gives back the original representation
      function accuracy(x, y)
          return sum(Flux.onecold(model(x)) .== Flux.onecold(y)) / size(y, 2)
      end

   ``model`` will be our neural network, so we go ahead and define it:

   .. code-block:: julia

      n_features, n_classes, n_neurons = 4, 3, 10
      model = Chain(
              Dense(n_features, n_neurons, sigmoid),
              Dense(n_neurons, n_classes),
              softmax)  

   We now set up our optimizer. We have selected the standard optimizer ADAM:

   .. code-block:: julia

      opt_state = Flux.setup(Adam(), model)

   Before training the model, let's have a look at some initial predictions 
   and the accuracy:

   .. code-block:: julia

      # predictions before training
      model(xtrain[:,1:5])
      ytrain[:,1:5]
      # accuracy before training
      accuracy(xtrain, ytrain)
      accuracy(xtest, ytest)

   Finally we are ready to train the model. Let's run 100 epochs:

   .. code-block:: julia

      # the training data and the labels can be passed as tuples to train!
      for i in 1:100
          Flux.train!((m,x,y) -> loss(m, x, y), model, [(xtrain, ytrain)], opt_state)
      end

      # check final accuracy
      accuracy(xtrain, ytrain)
      accuracy(xtest, ytest)

   The performance of the model is probably somewhat underwhelming, but you will 
   fix that in an exercise below!

   We finally create a confusion matrix to quantify the performance of the model:

   .. code-block:: julia

      predicted_species = Flux.onecold(model(xtest), ["Adelie", "Gentoo", "Chinstrap"])
      true_species = Flux.onecold(ytest, ["Adelie", "Gentoo", "Chinstrap"])
      ConfusionMatrix()(predicted_species, true_species)


Exercises
---------

.. _DLexercise:

.. exercise:: Improve the deep learning model

   Improve the performance of the neural network we trained above! 
   The network is not improving much because of the large numerical 
   range of the input features (from around 15 to around 6000) combined 
   with the fact that we use a ``sigmoid`` activation function. A standard 
   method in machine learning is to normalize features by "batch 
   normalization". Replace the network definition with the following and 
   see if the performance improves:
   
   .. code-block:: julia

      n_features, n_classes, n_neurons = 4, 3, 10
      model = Chain(
                 Dense(n_features, n_neurons),
                 BatchNorm(n_neurons, relu),
                 Dense(n_neurons, n_classes),
                 softmax)  

   Performance is usually better also if we, instead of training on the entire 
   dataset at once, divide the training data into "minibatches" and update 
   the network weights on each minibatch separately.
   First define the following function:

   .. code-block:: julia

      using StatsBase: sample

      function create_minibatches(xtrain, ytrain; batch_size=32, n_batch=10)
          minibatches = Tuple[]
          for i in 1:n_batch
              randinds = sample(1:size(xtrain, 2), batch_size)
              push!(minibatches, (xtrain[:, randinds], ytrain[:,randinds]))
          end
          return minibatches
      end

   and then create the minibatches by calling the function.  

   You will not need to manually loop over the minibatches, simply pass 
   the ``minibatches`` vector of tuples to the ``Flux.train!`` function. 
   Does this make a difference?

   .. solution:: 

      .. code-block:: julia

         function create_minibatches(xtrain, ytrain; batch_size=32, n_batch=10)
             minibatches = Tuple[]
             for i in 1:n_batch
                 randinds = sample(1:size(xtrain, 2), batch_size)
                 push!(minibatches, (xtrain[:, randinds], ytrain[:,randinds]))
             end
             return minibatches
         end
   
         n_features, n_classes, n_neurons = 4, 3, 10
         model = Chain(
                 Dense(n_features, n_neurons),
                 BatchNorm(n_neurons, relu),
                 Dense(n_neurons, n_classes),
                 softmax)

         opt_state = Flux.setup(Adam(), model)
   
         minibatches = create_minibatches(xtrain, ytrain)
         for i in 1:100
             # train on minibatches
             Flux.train!((m,x,y) -> loss(m, x, y), model, minibatches, opt_state);
         end
   
         accuracy(xtrain, ytrain)
         # 0.9849624060150376
         accuracy(xtest, ytest)
         # 0.9850746268656716
   
         predicted_species = Flux.onecold(model(xtest), ["Adelie", "Gentoo", "Chinstrap"])
         true_species = Flux.onecold(ytest, ["Adelie", "Gentoo", "Chinstrap"])
         ConfusionMatrix()(predicted_species, true_species)
   
      .. figure:: img/confusion_matrix.png
         :scale: 40 %

      Much better!

.. exercise:: More improvements

   **Exercise: Hyperparameter Tuning**
      
   Experiment with different hyperparameters of the model and the training process. 

   .. code-block:: julia

      # Try different batch sizes in the minibatch creation.
      minibatches = create_minibatches(xtrain, ytrain, batch_size=64, n_batch=10)

      # Experiment with different learning rates for the ADAM optimizer.
      opt_state = Flux.setup(Adam(0.05), model)

      # Change the number of neurons in the hidden layer of the model.
      model = Chain(
         Dense(n_features, 20, relu),
         Dense(20, n_classes),
         softmax
      )

      # The solution will depend on the specific hyperparameters chosen.

   **Exercise: Feature Engineering**
      
   Consider doing some feature engineering on your input data.

   .. code-block:: julia

      # Try normalizing or standardizing the input features.
      xtrain = (xtrain .- mean(xtrain, dims=2)) ./ std(xtrain, dims=2)
      xtest = (xtest .- mean(xtest, dims=2)) ./ std(xtest, dims=2)

   **Exercise: Different Model Architectures**
      
   Experiment with different model architectures.

   .. code-block:: julia

      # Try adding more layers to your model.
      model = Chain(
         Dense(n_features, n_neurons, relu),
         Dense(n_neurons, n_neurons, relu),
         Dense(n_neurons, n_classes),
         softmax
      )

   Remember to experiment and see how these changes affect your model's performance! 😊

See also
--------

-  Many interesting datasets are available in Julia through the 
   `RDatasets <https://github.com/JuliaStats/RDatasets.jl>`_ package.
   For instance:

   .. code-block:: julia

      Pkg.add("RDatasets")
      using RDatasets
      # load a couple of datasets
      iris = dataset("datasets", "iris")
      neuro = dataset("boot", "neuro")

- `"The Future of Machine Learning and why it looks a lot like Julia" by Logan Kilpatrick <https://towardsdatascience.com/the-future-of-machine-learning-and-why-it-looks-a-lot-like-julia-a0e26b51f6a6>`_
- `Deep Learning with Flux - A 60 Minute Blitz <http://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/>`__
- `Deep Convolutional Generative Adversarial Network (DCGAN) <http://fluxml.ai/Flux.jl/stable/tutorials/2021-10-08-dcgan-mnist/>`__

Neuromorphic | Probabilistic learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - https://darsnack.github.io/SpikingNN.jl/dev/
   - https://turinglang.org/v0.24/tutorials/
   - Nordic Neuromorphs | NorN Discord Community – https://discord.gg/5Qq6yX5

Quantum
^^^^^^^

   - https://juliapackages.com/c/quantum-mechanics
   - Swedish Quantum Society | SQS – https://swedishquantumsociety.vercel.app/
