Motivation
==========

.. questions::

   - What is the two-language problem?
   - How performant is Julia?
   - What is composability?
   - What will we learn and not learn in this lesson?
   - How does Julia achieve both speed and dynamism as a programming language?
   - What is the two-language problem and how does Julia solve it?
   - What is an example of composability in Julia and why is it useful for HPDA applications?
   - What are some of the drawbacks and workarounds of Julia, such as time to first plot, ecosystem maturity, package evolution, and memory footprint?
   - What are some of the topics that we will cover in this project, such as multithreading, multiprocessing, MPI, and GPU computing?
   - What are some of the topics that we will not cover in this project, such as interoperability with other languages and specific scientific packages?

.. instructor-note::

   - 15 min teaching

.. code-block:: rst

  Julia is a programming language that combines the best of both worlds: the speed and efficiency of compiled languages like C/C++ and Fortran, and the expressiveness and productivity of interpreted languages like Python and R. Julia is especially suited for high-performance data analysis (HPDA), which involves processing large volumes of data and performing complex computations on them. HPDA applications can benefit from Julia's features, such as:

  - **Multiple dispatch**: Julia allows functions to have different behaviors depending on the types of their arguments, enabling generic and flexible code that can handle various data structures and algorithms.
  - **Metaprogramming**: Julia has a powerful macro system that can manipulate code as data, allowing for code generation and transformation at compile time or run time.
  - **Parallelism**: Julia supports various forms of parallelism, such as distributed computing, multi-threading, coroutines, and GPU computing, making it easy to scale up and speed up HPDA applications.
  - **Interoperability**: Julia can seamlessly call functions and libraries written in other languages, such as Python, R, C/C++, and Fortran, allowing for reuse of existing code and integration with other tools.

  In this project, we will explore how to use Julia for HPDA applications in different domains, such as scientific computing, machine learning, bioinformatics, and social network analysis. We will learn how to write efficient and elegant Julia code, how to leverage existing packages and libraries, how to parallelize and optimize our code, and how to benchmark and compare our results with other languages. We will also discuss the challenges and limitations of Julia, such as time to first plot, ecosystem maturity, package evolution, and memory footprint. By the end of this project, we hope to demonstrate that Julia is a powerful and practical language for HPDA.