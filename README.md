# Tree-Decision
Tree Decision with automatic and phased building.

Creating a class object: `DecisionTree(df, tar_var, tar_val, mode, criterion, max_branches, min_objs)`.

  * `df` – object of class DataFrame from Pandas;

  * `tar_var` - target variable;

  * `tar_val` - value of target variable;
  
  * `mode` - build mode (`auto` or `phased`);
  
  * `criterion` - build criterion (`chi2` - Chi-squared; `entropy` - Information gain; `gini` - Gini impurity);
  
  * `max_branches` - maximum number of branches in length;
  
  * `criterion` - minimum number of objects in a branch.
  
  Class Methods:
  
  `build()` - building a decision tree.
  `draw()` - drawing a decision tree graph.
  
  ![alt text](https://sun9-12.userapi.com/impg/GQ7PrtBeTn4WD4tpJtLFMXO5dTNMomQUtKdxPg/PPYW8lZn-H8.jpg?size=2560x1023&quality=96&sign=de6ed80e0c120b36f0dea0680524a9e0&type=album)
