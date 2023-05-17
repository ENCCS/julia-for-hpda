.. _linear_algebra:

Linear algebra
==============

.. questions::

   - How can I perform vector and matrix operations in Julia?
   - Can I easily use Julia for typical linear algebra tasks?
     
.. instructor-note::

   - 40 min teaching
   - 20 min exercises


Loading a dataset
^^^^^^^^^^^^^^^^^

We start by downoading Fisher's iris dataset. This dataset contains
measurements from 3 different species of the plant iris: setosa, versicolor and
virginica with 50 datapoints of each species. There are four
measurements for datapoint, namely sepal length, sepal width, petal
length and petal width (in centimeters).

.. figure:: img/iris_resize.jpg
   :align: center

   Image of iris by David Iliff.

To obtain the data we use the RDatasets package:

.. code-block:: julia

   using DataFrames, LinearAlgebra, Statistics, RDatasets, Plots
   df = dataset("datasets", "iris")

Matrix and vector operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall that matrices and vectors may be defined as follows:

.. code-block:: julia

   # define some column vectors
   v1 = [1.0, 2.0, 3.0]
   v2 = v1.^2

   # combine column vectors into 3x3 matrix
   A = [v1 v2 [7.0, 6.0, 5.0]]

   # another way to make matrices
   M = [5 -3 2;15 -9 6;10 -6 4]

.. code-block:: text

   julia> A
   3×3 Matrix{Float64}:
    1.0  1.0  7.0
    2.0  4.0  6.0
    3.0  9.0  5.0

   julia> M
   3×3 Matrix{Int64}:
     5  -3  2
    15  -9  6
    10  -6  4

.. code-block:: julia

   # vector addition and scaling
   v1 + v2
   v1 - 0.5*v2

   B = [v3 v2 v1]

   # matrix vector multiplication
   A*v1

   # matrix multiplication
   A*B
   A^5

.. code-block:: text

   julia>  v1+v2
   3-element Vector{Float64}:
     2.0
     6.0
    12.0

   julia> v1 - 0.5*v2
   3-element Vector{Float64}:
     0.5
     0.0
    -1.5

   julia> A*B
   3×3 Matrix{Float64}:
    44.0  68.0  24.0
    44.0  72.0  28.0
    48.0  84.0  36.0

Standard operations such as rank, determinant, trace, matrix multiplication,
transpose, matrix inverse, identity operator, eigenvalues, eigen vectors and so on:
.. code-block:: julia

   # rank of matrix
   rank(A) # full rank 3

   # determinant
   det(A) # 16

   # lower rank matrix
   C = [v1 v2 v1+0.66*v2]

   rank(C) # rank 2

   # 6x6 matrix
   D = [A A;A A]
   rank(D) # 3
   det(D) # 0

   # trace
   tr(A) # 10

   # eigen vectors and eigenvalues
   eigen(A)

   # identity operator (does not build identity matrix)
   I
   A*I # A
   I*D # D

   # matrix inverse
   inv(A)
   inv(A)*A # identity matrix
   A*inv(A) # identity matrix

   # solving linear systems of equations
   u = A*v1
   # solve A*x = u with least squares
   A \ u # v1
   # solve in another way
   inv(A)*u # v1

   # matrix must have full rank
   inv(C) # ERROR: SingularException(3)

   # nilpotent matrix M from above
   rank(M) # 1
   M*M # zero matrix

   # transpose
   transpose(A)
   A' # transpose of real matrix
   # complex matrix
   E = (A+im*A)
   E' # Hermitian conjugate

   # dot product
   dot(v1, v2) # 36
   v1'*v2 # 36

   # cross product of 3-vectors
   cross(v1, v2)
   dot(cross(v1, v2), v1) # 0 (orthogonal)


.. code-block:: text

   julia> eigen(A)
   Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
   values:
   3-element Vector{Float64}:
    -3.250962397052609
    -0.3615511210246384
    13.61251351807725
   vectors:
   3×3 Matrix{Float64}:
    -0.821765  -0.96124   -0.440897
    -0.211254   0.228475  -0.539484
     0.529221   0.154329  -0.717333

TODO:

  * QR factorization, diagonalization or similar, change of basis
  * random matrices
  * Sparse operations (with random examples?)

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
