Instructor's guide
==================

Prerequisites
-------------

This lesson material assumes some familiarity with Julia's syntax. The necessary
prerequisites are covered in the
`introductory Julia lesson <https://enccs.github.io/julia-intro/>`__ which should
be required reading for participants prior to attending a workshop.
In addition, before the workshop the participants need to install the required packages as explained in
`Installing packages <https://enccs.github.io/julia-for-hpda/setup/>`__.

Mode of teaching
----------------

This lesson uses the VSCode environment as it is the preferred IDE by Julia developers and
has a powerful language extension for Julia. 

The prerequisite
`introductory Julia lesson <https://enccs.github.io/julia-intro/>`__
has an initial hands-on episode "Special features of Julia" where
only the Julia REPL is used. This makes learners comfortable with the REPL
before moving on to the more complicated environment in VSCode.


Demonstrations, type-alongs and exercises
-----------------------------------------

The instructor walks through the material and demonstrates all the coding found 
outside the special type-along boxes. It's important to not rush and to clearly 
explain what is being written. It should be clear that learners are not expected 
to type-along during these sessions. The instructor can either type things out or 
copy-paste from the code blocks.

Some episodes have type-along sections demarkated with light-green boxes with a keyboard
emoji. It should be clearly explained that learners are expected to type-along during 
these sessions. Here's it's better for the instructor to type things out rather than 
copy-pasting everything, although larger code blocks should be copy-pasted to avoid 
error-prone and boring typing.

Each episode ends with one or more exercises. Learners should be given plenty of 
time to work on these. Recommended timings are provided at the top of each episode.


Possibly confusing points
-------------------------

- To enable learners to copy-paste from code blocks to install and manage packages, 
  the lesson adheres to the convention of using the ``Pkg`` API (e.g. 
  ``using Pkg ; Pkg.add("some-package")``. This is explained in the "Developing in Julia" episode 
  of the `introductory Julia lesson <https://enccs.github.io/julia-intro/>`__ but needs to be
  explained very carefully to avoid confusion.

Schedule
--------

The following schedule was used for a workshop in October 2023. As a prerequisite, one day was spent
on the `introductory Julia lesson <https://enccs.github.io/julia-intro/>`__ according to the
`schedule <https://enccs.github.io/julia-intro/guide/#suggested-schedule-for-1-day-workshop/>`__.
The next three days were scheduled as follows:

**Day 1**

+-------------+--------------------------------------------+
|  Time       | Section                                    |
+=============+============================================+
| 9:00-9:10   | Welcome                                    |
+-------------+--------------------------------------------+
| 9:10-9:25   | Motivation (Julia for data analysis)       |
+-------------+--------------------------------------------+
| 9:25-9:45   | Data formats and data frames               |
+-------------+--------------------------------------------+
| 9:45-10:00  | Break                                      |
+-------------+--------------------------------------------+
| 10:00-10:45 | Data formats and data frames               |
+-------------+--------------------------------------------+
| 10:45-11:00 | Break                                      |
+-------------+--------------------------------------------+
| 11:00-12:00 | Linear algebra                             |
+-------------+--------------------------------------------+


**Day 2**

+-------------+--------------------------------------------+
|  Time       | Section                                    | 
+=============+============================================+
| 9:00-9:45   | Data science and machine learning 1        |
+-------------+--------------------------------------------+
| 9:45-10:00  | Break                                      |
+-------------+--------------------------------------------+
| 10:00-10:45 | Data science and machine learning 2        |
+-------------+--------------------------------------------+
| 10:45-11:00 | Break                                      |
+-------------+--------------------------------------------+
| 11:00-12:00 | Data science and machine learning 3        |
+-------------+--------------------------------------------+


**Day 3**

+-------------+--------------------------------------------+
|  Time       | Section                                    | 
+=============+============================================+
| 9:00-9:45   | Linear regression                          |
+-------------+--------------------------------------------+
| 9:45-10:00  | Break                                      |
+-------------+--------------------------------------------+
| 10:00-10:45 | Fourier techniques, non-linear regression  |
+-------------+--------------------------------------------+
| 10:45-11:00 | Break                                      |
+-------------+--------------------------------------------+
| 11:00-11:45 | Non-linear regression continued            |
+-------------+--------------------------------------------+
| 11:45-12:00 | Conclusions and outlook                    |
+-------------+--------------------------------------------+


Future improvements of the lesson
---------------------------------

- Provide a Project.toml file in a repository for participants to download
  and instantiate in project environment before workshop starts.
