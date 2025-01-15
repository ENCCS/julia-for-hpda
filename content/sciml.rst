.. _sciml:

Scientific Machine Learning
===========================

.. questions::

   - What is the SciML Julia package and how to use it?
   - How can I mix physical pre-knowledge and data to solve modelling problems in Julia?

.. instructor-note::

   - 35 min teaching
   - 15 min exercises

.. callout::

   The code in this lession is written for Julia v1.11.2.

A modelling problem
-------------------

In this session we will have a look at a physical modelling problem
and investigate how Machine Learning can be used in combination with
classical problems.

Consider a spherical object moving in a viscous fluid under the influence
of a drag force and gravity. We will consider this problem in 2 dimensions.

.. code-block:: julia

   using LinearAlgebra, Statistics, Random
   using ComponentArrays, Lux, Zygote, Plots
   using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
   using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches

   rng = Random.default_rng()

   function dynamics!(du, u, p, t)
      m = 1.0
      g = 10.0

      du[1] = -((u[1]^2 + u[3]^2)^0.5)*u[1] + 2*u[3]
      du[3] = -((u[1]^2 + u[3]^2)^0.5)*u[3] - m*g - 2*u[1]
      du[2] = u[1]
      du[4] = u[3]
   end

   tspan = (0.0, 1.0)

   sc = 1.0
   deltat = 0.1

   u0 = sc * rand(4)
   inits_g = [rand(4) for ii in range(1,6)]

   prob = ODEProblem(dynamics!, u0, tspan)

   prbs = [ODEProblem(dynamics!, ui, tspan) for ui in inits_g]

   times = Vector(0:deltat:1.0)

   sols = [Array(solve(prb, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.1)) for prb in prbs]

   Xs = hcat(sols...)

   # for axis equal; aspect_ratio = :equal
   scatter(Xs[2,:], Xs[4,:], alpha = 0.75, color = :green, label = ["True Data" nothing])

   .. figure:: img/solutions_1.png
   :align: center

   Solutions to an initial value problem.