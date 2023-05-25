.. _linear_algebra:

Linear algebra
==============

.. questions::

   - How can I create vectors and matrices in Julia?
   - How can I perform vector and matrix operations in Julia?
     
.. instructor-note::

   - 40 min teaching
   - 20 min exercises

List comprehension and vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can create vectors in a simple way similar to Python.

.. code-block:: julia

   # list comprehension
   [i^2 for i in range(1,40)] # 40-element Vector

   # conditional list comprehension
   [i^2 for i in range(1,40) if i%5==0] # 8-element Vector

   f(x,y)=x*y # f (generic function with 1 method)
   A = [1,2,3,4]
   B = [2,3,4,5]
   f.(A, B) # 2,6,12,20

   # another way
   for x in zip(A,B)
       println(x[1]*x[2])
   end

Vectorization is done with the dot syntax similar to Matlab.

.. code-block:: julia

   # vectorization
   A.^2 # [1,4,9,16]
   A .+ B
   A + B == A .+ B # true

   sin(A)
   # ERROR: MethodError: no method matching sin(::Vector{Int64})

   sin.(A) # 4-element Vector

   # vectorize everywhere
   @. sin(A) + cos(A)
   @. A+A^2-sin(A)*sin(B)

.. code-block:: text
   julia> @. A+A^2-sin(A)*sin(B)

   4-element Vector{Float64}:
     1.2348525987657073
     5.871679939797543
    12.106799974237582
    19.27428371612359

An example where vectorization, random vectors and Plot are used:

.. code-block:: julia
   using Plots

   x = range(0, 10, length=100)
   # vector has length 100
   # from 0 to 10 in 99 steps of size 10/99=0.101...

   y = sin.(x)
   y_noisy = @. sin(x) + 0.1*randn() # normally distributed noise

   plt = plot(x, y, label="sin(x)")
   plot!(x, y_noisy, seriestype=:scatter, label="data")

   # to save figure in file
   # savefig("sine_with_noise.png")

   diaplay(plt)

Adding elements to existing arrays (appending arrays).

.. code-block:: julia

   # pushing elements to vector
   U = [1,2,3,4]
   push!(U, 55) # [1,2,3,4,55]
   pop!(U) # 55
   U # [1,2,3,4]

   # Array of type Any
   U = []
   push!(U, 5) # [5]
   u = [1,2,3]
   push!(U, u) # [5, [1,2,3]]

   # references
   u = [1,2,3,4]
   v = u
   v[2] = 33
   v # [1,33,3,4]
   u # [1,33,3,4]

   # using copy
   u = [1,2,3,4]
   v = copy(u)
   v[2] = 33
   v # [1,33,3,4]
   u # [1,2,3,4]

   # curiosity: push! stores a reference to the object pushed, not a copy
   u[2] = 77
   U # [5, [1,77,3]]

   # Can use copy if want other behavior
   U = []
   push!(U, 5) # [5]
   u = [1,2,3]
   push!(U, u) # [5, copy(u)]
   u[2] = 77
   U # is still [5, [1,2,3]]
   # however
   v = U[2]
   v[2] = 77
   U # [5, [1,77,3]]

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

Timing
^^^^^^

Some examples of timing and benchmarking.

.. code-block:: julia

   function my_product(A, B)
       for x in zip(A,B)
           push!(C, x[1]*x[2])
       end
   C
   end

   A = randn(10^8)
   B = randn(10^8)
   C = Float64[]

   @time my_product(A, B);
   @time A.*B;

   tic = time()
   C = my_product(A, B)
   toc = time()
   println(toc - tic)

.. code-block:: julia

   4.496966 seconds (100.01 M allocations: 1.563 GiB, 31.38% gc time, 0.21% compilation time)
   0.195021 seconds (4 allocations: 762.940 MiB, 1.21% gc time)
   3.4010000228881836

.. questions::

   - What does @time do? Why is there a relatively large difference
     above between manual timing and timing with @time?

Loading a dataset
^^^^^^^^^^^^^^^^^

To prepare our illustration of PCA, we start by downoading Fisher's
iris dataset. This dataset contains measurements from 3 different
species of the plant iris: setosa, versicolor and virginica with 50
datapoints of each species. There are four measurements for datapoint,
namely sepal length, sepal width, petal length and petal width (in
centimeters).

.. figure:: img/iris_resize.jpg
   :align: center

   Image of iris by David Iliff.

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


   plt = plot(setosa[1,:],setosa[2,:],setosa[3,:], seriestype=:scatter, label="setosa")
   plot!(versicolor[1,:],versicolor[2,:],versicolor[3,:], seriestype=:scatter, label="versicolor")
   plot!(virginica[1,:],virginica[2,:],virginica[3,:], seriestype=:scatter, label="virginica")
   plot!(xlabel="PC1", ylabel="PC2", zlabel="PC3")

   display(plt)

.. figure:: img/iris_scatter_plot.png
   :align: center

   Scatter plot of the projected data.

TODO:
  * QR factorization?
  * random matrices
  * Sparse operations (with random examples)
  * Compare execution time with sparse matrix computations and normal
  * Plot histograms of different distributions from random library
  * Make some excersizes on these themes
