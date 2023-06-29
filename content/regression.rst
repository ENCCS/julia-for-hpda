.. _regression:

Regression, time-series prediction and analysis
================================================

.. questions::

   - How can I perform regression in Julia?
   - How can I perform time-series analysis and prediciton in Julia?

.. instructor-note::

   - 120 min teaching
   - 60 min exercises

Linear regression with synthetic data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin with some simple examples of linear regression on generated data. For the models we will use the package GLM (Generlized Linear Models).

.. code-block:: julia

   using Plots, GLM, DataFrames

   X = Vector(range(0, 10, length=20))
   y = 5*X .+ 3.4
   y_noisy = @. 5*X + 3.4 + randn()

   plt = plot(X, y, label="linear")
   plot!(X, y_noisy, seriestype=:scatter, label="data")

   display(plt)

.. figure:: img/linear_synth_1.png
   :align: center

.. code-block:: julia

   df = DataFrame(cX=X, cy=y_noisy)
   lm1 = fit(LinearModel, @formula(cy ~ cX), df)

   # alternative syntax
   # lm(@formula(cy ~ cX), df)
   # glm(@formula(cy ~ cX), df, Normal(), IdentityLink())
   # lm(@formula(cy ~ cX), df)

.. code-block:: text

   StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

   cy ~ 1 + cX

   Coefficients:
   ───────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)  Lower 95%  Upper 95%
   ───────────────────────────────────────────────────────────────────────
   Intercept)  3.46467   0.448322    7.73    <1e-06    2.52278    4.40656
   cX           5.05127   0.0766497  65.90    <1e-22    4.89024    5.21231
   ───────────────────────────────────────────────────────────────────────

.. code-block:: julia

   # note the formula argument
   fit(LinearModel, @formula(cX ~ cy), df) # modelling line with slope 1/5 and intercept -3.4/5

Plotting the result.

.. code-block:: julia

   y_pred = predic(lm1)

   # alternative: do it explicitly
   # coeffs = coeftable(lm1).cols[1] # intercept and slope
   # y_pred = coeffs[1] + coeffs[2]*X

   plot!(X, y_pred, label="predicted")

   display(plt)

.. figure:: img/linear_synth_2.png
   :align: center

   Image of linear model prediction. The example shown has intercept 2.9 and slope 5.1 (the result depends on random added noise).

Multivariate linear models are very similar.

.. code-block:: julia

   using Plots, GLM, DataFrames

   n = 4
   C = randn(n+1,1)
   X = rand(100,n)

   y = X*C[2:end] .+ C[1]
   y_noisy = y .+ 0.01*randn(100,1)

   df = DataFrame(cX1=X[:,1], cX2=X[:,2], cX3=X[:,3], cX4=X[:,4], cy=y_noisy[:,1])


   lm2 = lm(@formula(cy ~ cX1+cX2+cX3+cX4), df)

   println(lm2)
   println()
   print(C)

.. code-block:: text

   cy ~ 1 + cX1 + cX2 + cX3 + cX4

   Coefficients:
   ───────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error        t  Pr(>|t|)  Lower 95%  Upper 95%
   ───────────────────────────────────────────────────────────────────────────
   (Intercept)  -1.02879   0.0035902   -286.55    <1e-99  -1.03592   -1.02166
   cX1          -0.935462  0.0034155   -273.89    <1e-99  -0.942242  -0.928681
   cX2           0.183037  0.00345387    52.99    <1e-71   0.17618    0.189894
   cX3          -0.737696  0.00390208  -189.05    <1e-99  -0.745443  -0.729949
   cX4          -1.59192   0.00327437  -486.18    <1e-99  -1.59842   -1.58542
   ───────────────────────────────────────────────────────────────────────────

   [-1.022984643687018; -0.9366244594383493; 0.18095529608948402; -0.7396860440808664; -1.595858344253308;;]

It is straight forward to incorporate linear models with basis functions, that is to model a function as a linear combination of given functions such polynomials or trigonometric functions.

.. code-block:: julia

   using Plots, GLM, DataFrames

   # try this polynomial
   X = range(-6, 6, length=40)
   y = X.^5 .- 34*X.^3 .+ 225*X
   y_noisy = y .+ randn(40,)

   # model sensitive to noise
   # if more noise, need more points (keep noise down for clarity in graph)

   plt = plot(X, y, label="polynomial")
   plot!(X, y_noisy, seriestype=:scatter, label="data")

   display(plt)

   df = DataFrame(cX=X, cy=y_noisy)

   lm1 = lm(@formula(cy ~ cX^5 + cX^4 + cX^3 + cX^2 + cX + 1), df)

.. code-block:: text

   StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

   cy ~ 1 + :(cX ^ 5) + :(cX ^ 4) + :(cX ^ 3) + :(cX ^ 2) + cX

   Coefficients:
   ───────────────────────────────────────────────────────────────────────────────────────
                        Coef.   Std. Error         t  Pr(>|t|)     Lower 95%     Upper 95%
   ───────────────────────────────────────────────────────────────────────────────────────
   (Intercept)   -0.0354375    0.343821        -0.10    0.9185   -0.734166      0.663291
   cX ^ 5         1.00118      0.000551333   1815.92    <1e-85    1.00006       1.0023
   cX ^ 4        -0.000992084  0.00169158      -0.59    0.5614   -0.00442979    0.00244563
   cX ^ 3       -34.054        0.0236797    -1438.11    <1e-82  -34.1021      -34.0058
   cX ^ 2         0.0230557    0.0571179        0.40    0.6890   -0.0930219     0.139133
   cX           225.511        0.226822       994.22    <1e-76  225.05        225.972
   ───────────────────────────────────────────────────────────────────────────────────────

.. figure:: img/linear_basis_1.png
   :align: center

   Fitting a polynomial to data.

.. code-block:: julia

   # try a cosine combination
   X = range(-6, 6, length=100)
   y = cos.(X) .+ cos.(2*X)
   y_noisy = y .+ 0.1*randn(100,)

   plt = plot(X, y, label="waveform")
   plot!(X, y_noisy, seriestype=:scatter, label="data")

   display(plt)

   df = DataFrame(X=X, y=y_noisy)

   lm1 = lm(@formula(y ~ 1 + cos(X) + cos(2*X) + cos(3*X) + cos(4*X)), df)

.. code-block:: text

StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

   y ~ 1 + :(cos(X)) + :(cos(2X)) + :(cos(3X)) + :(cos(4X))

   Coefficients:
   ────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error      t  Pr(>|t|)    Lower 95%  Upper 95%
   ────────────────────────────────────────────────────────────────────────────
   (Intercept)   0.0130408   0.0108222   1.21    0.2312  -0.00844393  0.0345256
   cos(X)        0.981561    0.015653   62.71    <1e-78   0.950486    1.01264
   cos(2X)       0.984984    0.0156219  63.05    <1e-78   0.953971    1.016
   cos(3X)      -0.0135547   0.015573   -0.87    0.3863  -0.044471    0.0173616
   cos(4X)       0.0148532   0.0155105   0.96    0.3407  -0.015939    0.0456454
   ────────────────────────────────────────────────────────────────────────────

.. figure:: img/linear_basis_2.png
   :align: center

   Fitting trigonomtric functions to data.

Note the similarity to Fourier analysis. Let's see how you do the Fourier transform of the data in the last example.




Loading data
^^^^^^^^^^^^

We will now have a look at a climate data set containing daily mean
temperature, humidity, wind speed and mean pressure at a location in
Dehli India over a period of several years.

.. code-block:: julia

   using DataFrames, CSV, DataFrames, Plots

   df_train = CSV.read("C:/Users/davidek/julia_kurser/DailyDelhiClimateTrain.csv", DataFrame)
   df_test = CSV.read("C:/Users/davidek/julia_kurser/DailyDelhiClimateTest.csv", DataFrame)
   df_train

   M = [df_train.meantemp df_train.humidity df_train.wind_speed df_train.meanpressure]
   plottitles = ["meantemp" "humidity" "wind_speed" "meanpressure"]
   plotylabels =  ["C°" "g/m^3" "km/h" "hPa"]
   # color=[1 2 3 4] gives default colors
   plot(M, layout=(4,1), color=[1 2 3 4], legend=false, title=plottitles, xlabel="time (days)", ylabel=plotylabels, size=(800,800))

.. figure:: img/climate_plots_first.png
   :align: center

   Plots of measurements.

TODO:

Linear regression
^^^^^^^^^^^^^^^^^

  * Linear regression real data
  * Linear regression basis functions (polynomial and cos, sin)
  * Link to Fourier Analysis
  * Use cos, sin or something as basis functions for climate data

Non-linear regression
^^^^^^^^^^^^^^^^^^^^^

  * Climate data
  * other data set?

Some standard time-series models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  * Linear models (including with dummy variables)
  * Autoregression
  * etc.