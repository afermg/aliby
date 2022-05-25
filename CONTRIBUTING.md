# Contributing

We focus our work on python 3.7 due to the current neural network being developed on tensorflow 1. In the near future we will migrate the networ to pytorch to support more recent versions of all packages.

## Issues
All issues are managed within the gitlab [ repository ](https://git.ecdf.ed.ac.uk/swain-lab/aliby/aliby/-/issues), if you don't have an account on the University of Edinburgh's gitlab instance and would like to submit issues please get in touch with [Prof. Peter Swain](mailto:peter.swain@ed.ac.uk ).

## Branching
* master: very sparingly and only for changes that need to be made in both
 versions as I will be merging changes from master into the development
 branches frequently
 
Branching cheat-sheet:
```git
git branch my_branch # Create a new branch called branch_name from master
git branch my_branch another_branch #Branch from another_branch, not master
git checkout -b my_branch # Create my_branch and switch to it

# Merge changes from master into your branch
git pull #get any remote changes in master
git checkout my_branch
git merge master

# Merge changes from your branch into another branch
git checkout another_branch
git merge my_branch #check the doc for --no-ff option, you might want to use it
```

## Data aggregation

ALIBY has been tested by a few research groups, but we welcome new data sources for the models and pipeline to be as general as possible. Please get in touch with [ us ](mailto:peter.swain@ed.ac.uk ) if you are interested in testing it on your data.
