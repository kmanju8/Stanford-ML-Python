# Machine Learning by Andrew Ng Programming Exercises: Pythonized
Stanford University Machine Learning Course taught by Andrew Ng on Coursera

My own implementations of the course exercises in Python, hence my choice to host online.
Boiler plate code rewritten also rewritten.
Several reasons to not use original Octave/Matlab files:

* Python is by far more useful and widely used than Oct/Mat. Aim is to gain more experience with python data libs.
* Original course assignments do to much legwork for students. Just filling in the gaps as told is insufficient in thoroughly understanding the material in my opinion.
* More responsibility in interpreting results. Where the original assignments offer expected values with interpreations for the results, reimplimenting naturally results in different results, requiring more thought in understanding results and potential improvements (closer to life).

Derivations in course also rather lacking. Would like to include additional derivations, where possible, but low priority.

File format in general aims to mimic that of the course exercises.
Exact changes differ from section to section; these are detailed within relevant folders.


Actual submissions were not made; it did not appear a good use of time to reimplement the code for submission.
So, whilst I cannot guarantee an identical match to the answers, where possible answer matched those given in the original assignments.
Sanity checks through plotting or similar additionally show that even if not in 100% agreement with original answers, solutions are practically sufficient. 

Initially numpy was used in order to learn more about the library. Being well versed with applied-maths, realized I am functionally able to work with numpy without any difficulty,
at minimum in the instance of this project. But in general there appeared little educational benefit to continue further.
From Ex3.1 (In which Multi-classification took an hour to train for a basic digit identifier) switch to pytorch was made, to utilize GPU instead, as well as learn the library as well.
As it turned out, my laptop's graphics card is too basic to support pytorch, so the intention is to use cloud services to run ML programs there. Google Cloud Platform is currently
my platform of choice, though subject to change due to the limited trial.

Long term goals:

*Application to other non-assigned problems.
*Final impressions and reflections on course.