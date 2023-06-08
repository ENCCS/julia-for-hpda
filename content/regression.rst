.. _regression:

Regression, time-series prediction and analysis
================================================

.. questions::

   - How can I perform regression in Julia?
   - How can I perform time-series analysis and prediciton in Julia?

.. instructor-note::

   - 120 min teaching
   - 60 min exercises

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

