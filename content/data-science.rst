.. _data_science:

Data science and machine learning
=================================

.. questions::

   - Can I use Julia for machine learning?
     
.. instructor-note::

   - 20 min teaching
   - 30 min exercises


Working with data
-----------------

Via Data Formats and Dataframes lesson, we explored a Julian approach
to manipulating and visualization of data.

Julia is a good language to use for data science problems as
it will perform well and alleviate the need to translate
computationally demanding parts to another language.

Here we will learn and clustering, classification, machine learning and deep learning (toy example). Use penguin data.machine learning.

Download a dataset
^^^^^^^^^^^^^^^^^^

We start by downloading a dataset containing measurements 
of characteristic features of different penguin species.


.. figure:: img/lter_penguins.png
   :align: center

   Artwork by @allison_horst

To obtain the data we simply add the PalmerPenguins package.

.. code-block:: julia

   Pkg.add("PalmerPenguins")
   using PalmerPenguins



Dataframes
^^^^^^^^^^

We will use `DataFrames.jl <https://dataframes.juliadata.org/stable/>`_ 
package function here to  analyze the penguins dataset, but first we need to install it:

.. code-block:: julia

   Pkg.add("DataFrames")
   using DataFrames

.. todo:: Dataframes

   We now create a dataframe containing the PalmerPenguins dataset.
   
   .. code-block:: julia
   
      using PalmerPenguins
      table = PalmerPenguins.load()
      df = DataFrame(table)
   
      # the raw data can be loaded by
      #tableraw = PalmerPenguins.load(; raw = true)
   
      first(df, 5)
   
   .. code-block:: text
   
      344×7 DataFrame
       Row │ species    island     bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g  sex     
           │ String     String     Float64?        Float64?       Int64?             Int64?       String? 
      ─────┼──────────────────────────────────────────────────────────────────────────────────────────────
         1 │ Adelie   Torgersen            39.1           18.7                181         3750  male
         2 │ Adelie   Torgersen            39.5           17.4                186         3800  female
         3 │ Adelie   Torgersen            40.3           18.0                195         3250  female
         4 │ Adelie   Torgersen       missing        missing              missing      missing  missing 
         5 │ Adelie   Torgersen            36.7           19.3                193         3450  female
   
   
   Summary statistics can be displayed with the ``describe`` function:
   
   .. code-block:: julia
   
      describe(df)
   
   .. code-block:: text
   
      7×7 DataFrame
       Row │ variable           mean     min     median  max        nmissing  eltype                  
           │ Symbol             Union…   Any     Union…  Any        Int64     Type                    
      ─────┼──────────────────────────────────────────────────────────────────────────────────────────
         1 │ species                     Adelie          Gentoo            0  String
         2 │ island                      Biscoe          Torgersen         0  String
         3 │ bill_length_mm     43.9219  32.1    44.45   59.6              2  Union{Missing, Float64}
         4 │ bill_depth_mm      17.1512  13.1    17.3    21.5              2  Union{Missing, Float64}
         5 │ flipper_length_mm  200.915  172     197.0   231               2  Union{Missing, Int64}
         6 │ body_mass_g        4201.75  2700    4050.0  6300              2  Union{Missing, Int64}
         7 │ sex                         female          male             11  Union{Missing, String}

   As it was done in the Data Formats and Dataframes lesson, we can
   
   .. code-block:: julia
   
      dropmissing!(df)
   
The main features we are interested in for each penguin observation are 
`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` and `body_mass_g`.
What the first three features mean is illustrated in the picture below.

.. figure:: img/culmen_depth.png
   :align: center

   Artwork by @allison_horst


Machine learning in Julia
-------------------------

Despite being a relatively new language, Julia already has a strong and rapidly expanding 
ecosystem of libraries for machine learning and deep learning. A fundamental advantage of Julia for ML 
is that it solves the two-language problem - there is no need for different languages for the 
user-facing framework and the backend heavy-lifting (like for most other DL frameworks).

A particular focus in the Julia approach to ML is `"scientific machine learning" (SciML) <https://sciml.ai/>`_ 
(a.k.a. physics-informed learning), i.e. machine learning which incorporates scientific models into 
the learning process instead of relying only on data. The core principle of SciML is `differentiable 
programming` - the ability to automatically differentiate any code and thus incorporate it into 
Flux (predictive) models.

However, Julia is still behind frameworks like PyTorch and Tensorflow/Keras in terms of documentation and API design.

Traditional machine learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Julia has packages for traditional (non-deep) machine learning:

- `ScikitLearn.jl <https://scikitlearnjl.readthedocs.io/en/latest/>`_ is a port of the popular Python package.
- `MLJ.jl <https://alan-turing-institute.github.io/MLJ.jl/dev/>`_ provides a common interface 
  and meta-algorithms for selecting, tuning, evaluating, composing and comparing over 150 machine learning models.


- `Machine Learning · Julia Packagesl <https://juliapackages.com/c/machine-learning/>`_: This is a website that lists various Julia packages related to machine learning, such as MLJ.jl, Knet.jl, TensorFlow.jl, DiffEqFlux.jl, FastAI.jl, ScikitLearn.jl, and many more. 
  You can browse the packages by their popularity, alphabetical order, or update date. Each package has a brief description and a link to its GitHub repository.
- `AI · Julia Packages <https://www.juliapackages.com/c/ai>`_: This is another website that lists Julia packages related to artificial intelligence, such as Flux.jl, 
  AlphaZero.jl, BrainFlow.jl, NeuralNetDiffEq.jl, Transformers.jl, MXNet.jl, and more. You can also sort the packages by different criteria and see their details.
- `Julia Libraries · Top Julia Machine Learning Libraries - Analytics Vidhya <https://www.analyticsvidhya.com/blog/2021/05/top-julia-machine-learning-libraries/>`_: This is 
  an article that discusses some useful Julia libraries for machine learning and deep learning applications, such as computer vision and natural language processing. 


We will use a few utility functions from ``MLJ.jl`` in our deep learning 
exercise below, so we will need to add it to our environment:

.. code-block:: julia

   using Pkg
   Pkg.add("MLJ")

Deep learning
^^^^^^^^^^^^^

`Flux.jl <https://fluxml.ai/>`_ comes "batteries-included" with many useful tools 
built in, but also enables the user to write own Julia code for DL components.

- Flux has relatively few explicit APIs for features like regularisation or embeddings. 
- All of Flux is straightforward Julia code and it can be worth to inspect and extend it if needed.
- Flux works well with other Julia libraries, like dataframes, images and differential equation solvers.
  One can build complex data processing pipelines that integrate Flux models.

To install Flux:

.. code-block:: julia

   using Pkg
   Pkg.add("Flux")


.. todo:: Training a deep neural network to classify penguins

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
      loss(x, y) = Flux.crossentropy(model(x), y)

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

   We now define an anonymous callback function to pass into the training function 
   to monitor the progress, select the standard ADAM optimizer, and extract the parameters 
   of the model:

   .. code-block:: julia

      callback = () -> @show(loss(xtrain, ytrain))
      opt = ADAM()
      θ = Flux.params(model)

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
      for i in 1:10
          Flux.train!(loss, θ, [(xtrain, ytrain)], opt, cb = Flux.throttle(callback, 1))
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


.. todo:: Create a custom plotting function

   Convert the final ``scatter`` plot in the type-along section "Visualizing the Penguin dataset"
   and convert it into a ``create_scatterplot`` function: 
   
   - The function should take as arguments a dataframe and two column symbols. 
   - Use the ``minimum()`` and ``maximum()`` functions to automatically set the x-range of the plot 
     using the ``xlim = (xmin, xmax)`` argument to ``scatter()``.
   - If you have time, try grouping the data by ``:island`` or ``:sex`` instead of ``:species`` 
     (keep in mind that you may need to adjust the number of marker symbols and colors).
   - If you have more time, play around with the plot appearance using ``theme()`` and the marker symbols and colors.

   .. solution::

      .. code-block:: julia

         function create_scatterplot(df, col1, col2, groupby)
             xmin, xmax = minimum(df[:, col1]), maximum(df[:, col1])
             # markers and colors to use for the groups
             markers = [:circle :ltriangle :star5 :rect :diamond :hexagon]
             colors = [:magenta :springgreen :blue :coral2 :gold3 :purple]
             # number of unique groups can't be larger than the number of colors/markers
             ngroups = length(unique(df[:, groupby]))
             @assert ngroups <= length(colors)
         
             scatter(df[!, col1],
                     df[!, col2],
                     xlabel = col1,
                     ylabel = col2,
                     xlim = (xmin, xmax),
                     group = df[!, groupby],
                     marker = markers[:, 1:ngroups],
                     color = colors[:, 1:ngroups],
                     markersize = 5,
                     alpha = 0.8
                     )
         end    

         create_scatterplot(df, :bill_length_mm, :body_mass_g, :sex)
         create_scatterplot(df, :flipper_length_mm, :body_mass_g, :island)  


.. _DLexercise:

.. todo:: Improve the deep learning model

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

      function create_minibatches(xtrain, ytrain, batch_size=32, n_batch=10)
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

         function create_minibatches(xtrain, ytrain, batch_size=32, n_batch=10)
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
   
         callback = () -> @show(loss(xtrain, ytrain))
         opt = ADAM()
         θ = Flux.params(model)
   
         minibatches = create_minibatches(xtrain, ytrain)
         for i in 1:100
             # train on minibatches
             Flux.train!(loss, θ, minibatches, opt, cb = Flux.throttle(callback, 1));
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
- `Deep Learning with Flux - A 60 Minute Blitz <https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html>`__
