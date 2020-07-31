# Setting up the environment

1. Run `python -m venv env`. Your `python` will need to be version 3.7+.
2. Run `source env/bin/activate` to activate the environment.
3. Run `pip install -r requirements.txt` to install requirements (Pandas).

# General guidelines

Code in `data_generation.py` should not be modified, but you can call those functions in whatever way you want. The data has two versions, a smaller data version 1 and a larger data version 2. You can control which one you access by using the static variables `DATA_VERSION_1` and `DATA_VERSION_2`

`data_fitting.py` has been provided as an example of what some of the solutions may involve. This file is intended only as a starting point and should be extended or corrected as necessary.

More tasks are presented here than can be reasonably solved in the suggested time limit (5 hours). It is up to you to prioritize which tasks to take to completion. However, we plan to discuss the possible solutions and approaches to many of the tasks outlined here, so even if you do not implement a solution to a particular task, it is worth considering how one might go about it.

Throughout these tasks, we are more interested in the speed, efficiency, and implementation quality of the computational approach than in the strength of the analysis. Approximate solutions should be considered, but the trade off between accuracy, generalizability, and efficiency should be considered. You will not be judged on your test coverage, but we do expect minimal testing as needed to accomplish your task.

Feel free to install other Python packages to help you complete the tasks (e.g. sklearn, scipy, etc.) as well as use the various features of the Python standard library.

Please commit your code as you go so that we can get a sense of your development process. You can collect your analysis results in whatever way you want (e.g. a separate write-up)

# Tasks:

1. Using data version 1, fit a linear model predicting the values of feature set C from linear combinations of features from set A and set B. Evaluate and comment on the R^2 of this model.

2. Imagine that this data (data version 1) was being generated in a live fashion, one timestamp at a time, as real time passes. At each timestamp, generate a linear model similar to task 1 using all the data up to that timestamp.

3. Fit a linear model in the same way as task 1 using data version 2. If possible, total memory usage should be kept under 1GB (or under 6 GB in a less advanced solution), and data generation + computation time should be under 10 minutes.

4. Perform task 3 using data version 2, but include an L2 penalty in the regression (i.e. ridge regression). Perform this regression for a range of L2 penalties and choose a method for selecting the "best" penalty hyperparameter. If the live fit (one for every timestamp) is not feasible, an L2 regression on the entire data set is still valuable. Additionally, a researcher has requested the ability to be able to "quickly" run any new arbitrary L2 penalty and calculate an R^2 as a trial (where quickly is <1 minute) without having to leave the Python process running between trials.

5. A researcher has hypothesized a non-linear relationship between the targets and feature A_f009 from data version 2. The proposed behavior is that when A_f009 is in the largest 1% of its distribition, we need a different linear model. Extract the feature set at the timestamps where A_f009 is above its 99% percentile of values and fit a linear model to these data points that predicts feature set C.
