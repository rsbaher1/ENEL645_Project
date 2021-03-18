ENEL645_Project - Winter 2021
======

Project Desription
------

TODO

Contribution
------
1. Clone this repo and checkout your branch

   * To create a branch, pull from master and use your name as the branch name.
   * on your machine run `git pull` before `git checkout <branch_name>`
   * Remember to ensure you are inside the directory of the Git project.

2. Open `config.py` in the root of the project.
   
   This file contains all the configurable parameters for the project. Once development of the code is complete (i.e. running with no errors), 
this should be the only file to edit for training/running the model.

   The variable `MODEL` should be assigned either `1` or `2`, anything else will result in an error.
   *  `MODEL = 1` means model 1 in defined in `CNN.py` will be trained.
   *  `MODEL = 2` means model 2 in defined in `CNN.py` will be trained.
   
   The variable `DATA_AUGMENTATION` should be assigned either `True` or `False`, anything else will result in an error.
   *  `DATA_AUGMENTATION = True` means that Data augmentation will be applied to the dataset before training.
   *  `DATA_AUGMENTATION = False` means that Data augmentation will **_NOT_** be applied to the dataset before training.
   *  *__Note:__ later on we can experiment with different data augmentation techniques if we have time.*
   
   The comments in the file explain the other variables.
   
3. When you want to update the repository run the following commands
   1. `git pull`
   2. `git add <file_name> <file_name>`
   3. `git commit -m "<summarize the changes you made>"`
   4. `git push`
   5. Got to github and greate a pull request from your branch to the master branch.
   
      Please explain in more detail what changes you made in the description of the pull request
      
4. *_REMINDER:_ Make sure to update your branch every time the master branch is updated to avoid any merge issues.*

   Below are the steps to update your branch:
   1. Stash your changes if you made any on your branch
   2. Create a pull request from master to your branch and merge
   3. Run: `git pull` and you should see the changes 
   4. Unstash to reapply the changes you made
