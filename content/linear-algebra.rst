.. _linear_algebra:

Linear algebra
=================================

.. questions::

   - How can I perform vector and matrix operations in Julia?
   - Can I easily use Julia for typical linear algebra tasks?
     
.. instructor-note::

   - 40 min teaching
   - 20 min exercises


Loading a dataset
^^^^^^^^^^^^^^^^^

We start by downoading Fisher's iris dataset. This dataset contains
measurements from 3 different species of iris: setosa, versicolor and
virginica with 50 datapoints of each species. There are four
measurements for datapoint, namely sepal length, sepal width, petal
length and petal width (in centimeters).

.. figure:: img/iris_resize.jpg
   :align: center

   Image by David Iliff.

To obtain the data we use the RDatasets package:

.. code-block:: julia

   using DataFrames, LinearAlgebra, Statistics, RDatasets, Plots
   df = dataset("datasets", "iris")


Principal Component Analysis (PCA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will now illustrate how PCA can be performed on the iris
dataset. First extract the first four columns (the features described
above) as well as the labels separately:

.. code-block:: julia

   Xdf = df[:,1:4]
   X = Matrix(Xdf)
   y = df[:,5]

Firt we center the data by substracting the mean and in addition we
normalize by dividing by the standard deviation:

.. code-block:: julia

   m = mean(X, dims=1)
   r = size(X)[1]
   X = X - ones(r,1)*m
   s = ones(1, 4)./std(X, dims=1)
   X = X.*s

Now compute the covariance matrix together with its eigenvectors and eigenvalues:

.. code-block:: julia

   M = transpose(X)*X
   P = eigvecs(M)
   E = eigvals(M)

.. code-block:: text

   4-element Vector{Float64}:
      3.08651062786422
     21.866774460125956
    136.19054024874245
    434.8561746632673

We see that the first eigenvalue is quite a bit smaller than the for
instance the last one. Our data lies approximately in a 3-dimensional
subspace. Most of the variance in the dataset happens in this subspace.

The basis of eigenvectors we got is orthogonal and normalized:

.. code-block:: julia

    transpose(P)*P
		
We may perform dimensionality reduction by projecting the data to this subspace: 

.. code-block:: julia

    # projection of dataset onto orthonormal basis of eigenvectors
    # the three with largest eigenvalues
    Xp = X*P[:,2:4]

    # This following results in three least important directions, interesting comparison
    # Xp = X*P[:,1:3]

Plotting the result:

.. code-block:: julia

   setosa = Xp'[:,y.=="setosa"]
   versicolor = Xp'[:,y.=="versicolor"]
   virginica = Xp'[:,y.=="virginica"]

   p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
   scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
   scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)

   plt = plot!(p,xlabel="PC1",ylabel="PC2",zlabel="PC3")

   display(plt)

.. figure:: img/iris_scatter_plot.png
   :align: center

   Scatter plot of the projected data.
