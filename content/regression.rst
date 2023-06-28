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
   y_noisy = @. 5*X .+ 3.4 + randn()

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
   # given slope 1/5 and intercept -3.4/5
   fit(LinearModel, @formula(cX ~ cy), df)

Plotting the result.

.. code-block:: julia

   y_pred = predic(lm1)

   # explicitly
   # coeffs = coeftable(lm1).cols[1] # intercept and slope
   # y_pred = coeffs[1] + coeffs[2]*X

.. figure:: img/linear_synth_2.png
   :align: center

   Image of linear model prediction. The example shown has intercept 2.9 and slope 5.1 (the result depends on random added noise).


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



Linear regression
^^^^^^^^^^^^^^^^^

Ideas:

  * Exlplain simple linear regression
  * Use cos, sin or something as basis functions for climate data

Non-linear regression
^^^^^^^^^^^^^^^^^^^^^

Some standard time-series models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

