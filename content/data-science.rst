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

In this episode, we will go through a *Clustering* example. Clustering is an
unsupervised learning technique, used to discover "groups" of datapoints (called
*clusters*) such that:

- Points belonging to the same cluster are *similar*, in the sense that they
  share common properties
- Points in different clusters are not similar

Being an unsupervised method, it doesn't need labeled examples, it aims to
uncover hidden structure in the data.
It is commonly used in a number of situations, including customer segmentation,
geospatial analysis, biomedical data analysis, even time series analysis in some
cases.

In Julia, the `Clustering.jl
<https://juliastats.org/Clustering.jl/stable/index.html>`_ package implements a
few common clustering algorithms, including *k-means*, density based clustering
and more, with most algorithms expecting data in a matrix of shape ``(features x
observations)``.

In this particular example, we will use (a sample of) the published dataset of
the `New York City yellow taxi rides <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>`_.
This data is published monthly by an agency of the municipality of New York and
includes data such as pickup and dropoff locations (lat/lon or an "ID" that can
be mapped to a location) and timestamps, trip length, airport fare surcharge,
fare mount, tips, and more. This dataset is very big (a few GBs, 12 million rows
for January 2015), thus we prepared two subsets for local exploration: one
having `100k rows
<https://github.com/ENCCS/julia-for-hpda/raw/refs/heads/fix-dl-notebooks/content/data/taxi_100k.parquet>`_
and one having `50k rows
<https://github.com/ENCCS/julia-for-hpda/raw/refs/heads/fix-dl-notebooks/content/data/taxi_50k.parquet>`_.
In our tests, the clustering should be able to run in just a few seconds even
with 100k rows. Please download the datasets now! The idea is to try to uncover
patterns in the data, e.g. correlation between location and tipping/fare,
whether clusters correspond to specific location, and so on. To do this, we will
use the *k-means* clustering algorithm. The objective is to partition the
dataset into ``k`` clusters by minimising the distance between points and their
assigned cluster centre. The algorithm is implemented as follows:

#. Choose the number of clusters ``k``. This can be done either by the user based
   on preliminary knowledge of the data (e.g. number of cell populations in
   biomedical data) or with some heuristic (e.g. elbow method)
#. Initialise ``k`` cluster centres randomly
#. Assign each point to the nearest centre
#. Update each centre as the mean of the assigned points
#. Repeat 3 and 4 until convergence

The function to minimise is:

.. math::

    \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2

where:

- :math:`C_k` is a cluster
- :math:`\mu_k` is its centroid

There are several other clustering techniques, such as *k-medoids*, DBSCAN and
hierarchical clustering.
Now, let us try to apply this to our example dataset!

.. type-along:: Clustering example of NYC yellow taxi rides

    .. code-block:: julia

        using DataFrames, Parquet2

        df = DataFrame(Parquet2.Dataset("taxi_100k.parquet"); copycols=true)

        dropmissing!(df)

        # Remove extreme outliers
        df = filter(row -> -74.1 < row[:pickup_longitude] < -73.7 && 40.6 < row[:pickup_latitude] < 40.9, df)


    We can now select the features we're interested in, such as geographical information (latitude and longitude), distance and fare:

    .. code-block:: julia

        features = select(df, [
            :pickup_longitude,
            :pickup_latitude,
            :fare_amount,
            :trip_distance,
        ])

    Moreover, we need to scale the data (we're computing distance between clusters,
    so we don't want larger fields to be overrepresented):

    .. code-block:: julia

        using Statistics

        X = Matrix(features)

        μ = mean(X, dims=1)
        σ = std(X, dims=1)

        X_scaled = ((X .- μ) ./ σ)'

    Now we can go for the actual clustering:

    .. code-block:: julia

        using Clustering

        k = 6
        result = kmeans(X_scaled, k)

        clusters = result.assignments

    Visualisation (plot pickup location, colour by cluster):

    .. code-block:: julia


        using Plots

        scatter(
            df.pickup_longitude,
            df.pickup_latitude,
            marker_z = clusters,
            ms = 2,
            alpha = 0.5,
            legend = false,
            xlabel = "Longitude",
            ylabel = "Latitude",
            title = "Taxi Clusters (Location + Price + Distance)"
        )

    You should hopefully get something similar:

    .. figure:: img/taxi_clusters.png
        :align: center

.. exercise:: Exercises

    .. exercise:: Exercise1: Spatial clustering only

        Cluster using only longitude and latitude and compare the results with
        multi-feature clustering.

        .. solution::

            .. code-block:: julia

                X_geo = Matrix(select(df, [:pickup_longitude, :pickup_latitude]))'

                result_geo = kmeans(X_geo, 6)

    .. exercise:: Exercise 2: Add time features

        Try adding the tip amount to the features and see if people in certain areas tip more!

        .. solution::

            .. code-block:: julia

                features = select(df, [
                    :pickup_longitude,
                    :pickup_latitude,
                    :fare_amount,
                    :tip_amount
                ])
                # ... carry on as before

    .. exercise:: Exercise 3: Geospatial plotting

        Let us now try to get a nicer plot by plotting the clusters on the land
        surface of the city. To do so, we use the ``GeoMakie.jl
        <https://geo.makie.org/stable/>`_ package, which is suitable for
        geospatial data plotting, and `NaturalEarth.jl
        <https://juliageo.org/NaturalEarth.jl/stable/>`_ which is a proxy to the
        `Natural Earth <https://www.naturalearthdata.com/>`_ dataset, containing
        the landmass (and other features) of the whole planet.

        .. solution::

            .. code-block:: julia


                using NaturalEarth, GeoJSON, CairoMakie, GeoMakie

                geo = NaturalEarth.naturalearth("land", 10)

                lon_min, lon_max = -74.3, -73.65
                lat_min, lat_max = 40.45, 40.95

                fig = Figure(resolution=(900,700), backgroundcolor=:lightblue);
                ax = Axis(fig[1,1]);

                for feature in geo.features
                    geom = feature.geometry

                    if geom isa GeoJSON.Polygon
                        for ring in geom.coordinates
                            xs = getindex.(ring, 1)
                            ys = getindex.(ring, 2)

                            if maximum(xs) > lon_min && minimum(xs) < lon_max
                                poly!(ax, xs, ys, color=:lightgray)
                            end
                        end

                    elseif geom isa GeoJSON.MultiPolygon
                        for poly_coords in geom.coordinates
                            for ring in poly_coords
                                xs = getindex.(ring, 1)
                                ys = getindex.(ring, 2)

                                if maximum(xs) > lon_min && minimum(xs) < lon_max
                                    poly!(ax, xs, ys, color=:lightgray)
                                end
                            end
                        end
                    end
                end
                Makie.scatter!(ax, df.pickup_longitude, df.pickup_latitude, color = clusters)
                Makie.xlims!(ax, lon_min, lon_max)
                Makie.ylims!(ax, lat_min, lat_max)

                fig

            .. figure:: img/taxi_geoplotting.png


Furthermore, we also have a couple of notebooks that deal with clustering and
classification problems. The first one deals with another example of
clustering of housing prices in California and follows a similar structure,
although it uses a different plotting package. The second one shows different
ways to perform a classification task on the Iris dataset from a previous
episode, building on our findings to try to predict the three species using a
variety of traditional machine learning techniques such as Lasso, Support Vector
Machines (SVM), Ridge regression and more. They can be run in VS Code or through
the JupyterLab interface.

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
tools without special abstractions. Moreover, a `Training API
<https://lux.csail.mit.edu/stable/api/Lux/utilities#Training-API>`_ is available
to hide some of the boilerplate if needed.

Generally speaking, Lux shines when integration with SciML is needed, or any
time neural networks are embedded in larger scientific computing applications,
and new developments tend to land in Lux first. Conversely, Flux is the more mature,
general purpose deep learning framework. From a Python perspective, Flux feels
more like PyTorch/Keras, whereas Lux is more akin to Jax/Flax.

To install Flux:

.. code-block:: julia

   using Pkg
   Pkg.add("Flux")

Whereas for Lux:

.. code-block:: julia

    using Pkg
    Pkg.add("Lux")




.. exercise:: Training a deep neural network to classify penguins

   To train a model we need four things:

   - A collection of data points that will be provided to the objective
     function.
   - A objective (cost or loss) function, that evaluates how well a model
     is doing given some input data.
   - The definition of a model and access to its trainable parameters.
   - An optimiser that will update the model parameters appropriately.

   In this case, we will train a simple neural network to be able to classify
   the four species of penguins from the dataset above based on their anatomical
   features (bill length, depth, etc.).

   First we load the data:

   .. code-block:: julia

      using MLJ: partition, ConfusionMatrix
      using DataFrames
      using PalmerPenguins
      using OneHotArrays

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
      ytrain = OneHotArrays.onehotbatch(ytrain, ["Adelie", "Gentoo", "Chinstrap"])
      ytest = OneHotArrays.onehotbatch(ytest, ["Adelie", "Gentoo", "Chinstrap"])

      # count penguin classes to see if it's balanced
      sum(ytrain, dims=2)
      sum(ytest, dims=2)



   Next up is the loss function which will be minimized during the training.
   We also define another function which will give us the accuracy of the model:

   .. tabs::

      .. group-tab:: Flux

        .. code-block:: julia

            using Flux
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

            predicted_species = OneHotArrays.onecold(model(xtest), ["Adelie", "Gentoo", "Chinstrap"])
            true_species = OneHotArrays.onecold(ytest, ["Adelie", "Gentoo", "Chinstrap"])
            ConfusionMatrix()(predicted_species, true_species)

        The model can be serialised for later inference using JLD2: 

        .. code-block:: julia 

            @save "model_trained.jld2" model_state=Flux.state(model)
            @load "model_trained.jld2" model_state 
            model = model(); # So the definition has to be available 
            Flux.loadmodel!(model, model_state);

      .. group-tab:: Lux

          .. code-block:: julia

              using Lux, Random, Optimisers
              # cross-entropy loss function
              loss = Lux.CrossEntropyLoss()
              # accuracy function
              # onecold (inverse of onehot) gives back the original representation
              function accuracy(model, ps, st, x, y)
                return sum(OneHotArrays.onecold(first(model(x, ps, st))) .== OneHotArrays.onecold(y)) / size(y,2)
              end

          `model` above is our neural network, so we can go on and create it!

          .. code-block:: julia

              n_features, n_classes, n_neurons = 4, 3, 10
              model = Lux.Chain(
                  Dense(n_features => n_neurons,sigmoid),
                  Dense(n_neurons => n_classes),
                  x -> softmax(x)
              )

          We can now set up the actual training infrastructure. For this case, we'll use the Lux Training API:

          .. code-block:: julia

              rng = Random.default_rng()
              ps, st = Lux.setup(rng, model)
              opt = Optimisers.Adam(0.01)

              train_state = Lux.Training.TrainState(model, ps, st, opt)

          In this case, we used an Adam optimiser. We can give a look at the initial predictions (i.e. before training)

          .. code-block:: julia

              accuracy(model, ps, st, xtrain, ytrain)
              accuracy(model, ps, st, xtest, ytest)

          Now we can start the training loop!

          .. code-block:: julia

              for epoch in 1:100
                _, l, _, train_state = Lux.Training.single_train_step!(AutoZygote(), loss, (xtrain, ytrain), train_state)
                if epoch % 10 == 0
                  println("Epoch $epoch - Loss $l")
                end
              end

              # check final accuracy
              accuracy(xtrain, ytrain)
              accuracy(xtest, ytest)

        The performance of the model is probably somewhat underwhelming, but you will
        fix that in an exercise below!

        We finally create a confusion matrix to quantify the performance of the model:

        .. code-block:: julia

            predicted_species = OneHotArrays.onecold(first(model(xtest, ps, st)), ["Adelie", "Gentoo", "Chinstrap"])
            true_species = OneHotArrays.onecold(ytest, ["Adelie", "Gentoo", "Chinstrap"])
            ConfusionMatrix()(predicted_species, true_species)

        The model parameters can be saved using JLD2:

        .. code-block:: julia 

            @save "trained_model.jld2" ps st 
            @load "trained_model.jld2" ps st

            y, st = model(x, ps, st)


Exercises
---------

.. exercise:: Improve the deep learning model

   Improve the performance of the neural network we trained above!
   The network is not improving much because of the large numerical
   range of the input features (from around 15 to around 6000) combined
   with the fact that we use a ``sigmoid`` activation function. A standard
   method in machine learning is to normalize features by "batch
   normalization". Replace the network definition with the following and
   see if the performance improves:

   .. tabs::

    .. group-tab:: Flux

        .. code-block:: julia

          n_features, n_classes, n_neurons = 4, 3, 10
          model = Chain(
                    Dense(n_features, n_neurons),
                    BatchNorm(n_neurons, relu),
                    Dense(n_neurons, n_classes),
                    softmax)

    .. group-tab:: Lux

        .. code-block:: julia

            n_features, n_classes, n_neurons = 4, 3, 10
            model = Chain(
                Dense(n_features => n_neurons),
                BatchNorm(n_neurons, relu),
                Dense(n_neurons => n_classes),
                softmax
            )


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

      .. tabs::

        .. group-tab:: Flux

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

        .. group-tab:: Lux

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
                Dense(n_features => n_neurons),
                BatchNorm(n_neurons, relu),
                Dense(n_neurons => n_classes),
                softmax
            )

            rng = Random.default_rng()
            ps, st = Lux.setup(rng, model)

            # ----------------------
            # Optimizer + TrainState
            # ----------------------
            opt = Adam(0.01)

            tstate = Lux.Training.TrainState(model, ps, st, opt)

            minibatches = create_minibatches(xtrain, ytrain)

            for epoch in 1:epochs
                for batch in minibatches
                    _, l, _, tstate = Training.single_train_step!(
                        AutoZygote(),
                        loss,
                        batch,
                        tstate
                    )
                end
            end

            accuracy(Lux.testmode(model), ps, st, xtrain, ytrain)
            # 0.9849624060150376
            accuracy(Lux.testmode(model), ps, st, xtest, ytest)
            # 0.9850746268656716

            predicted_species = OneHotArrays.onecold(first(model(xtest, ps, st)), ["Adelie", "Gentoo", "Chinstrap"])
            true_species = OneHotArrays.onecold(ytest, ["Adelie", "Gentoo", "Chinstrap"])
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

- `Deep Learning with Flux - A 60 Minute Blitz <http://fluxml.ai/Flux.jl/stable/tutorials/2020-09-15-deep-learning-flux/>`__
- `Deep Convolutional Generative Adversarial Network (DCGAN) <http://fluxml.ai/Flux.jl/stable/tutorials/2021-10-08-dcgan-mnist/>`__
- `Lux tutorials <https://lux.csail.mit.edu/stable/tutorials/>`_
- `Lux + Reactant <https://www.youtube.com/watch?v=bLNH8L6Zubg>`_
- `Deep learning with Flux (online book) <https://neroblackstone.github.io/D2lJulia/README.html>`_

Neuromorphic | Probabilistic learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   - https://darsnack.github.io/SpikingNN.jl/dev/
   - https://turinglang.org/v0.24/tutorials/
   - Nordic Neuromorphs | NorN Discord Community – https://discord.gg/5Qq6yX5
