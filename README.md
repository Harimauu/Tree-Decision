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
  
 ## Class Methods:
  
`build()` - building a decision tree.
  
`draw()` - drawing a decision tree graph.
  
![alt text](https://sun9-12.userapi.com/impg/GQ7PrtBeTn4WD4tpJtLFMXO5dTNMomQUtKdxPg/PPYW8lZn-H8.jpg?size=2560x1023&quality=96&sign=de6ed80e0c120b36f0dea0680524a9e0&type=album)

`get_probs(df)` - getting probabilities to plot ROC Curve.

`build_roc(y_vals, probs)` - plotting a Roc Curve. Also method returns AUC, optimal FPR and TPR.

  * `y_vals` – target variable values for objects;

  * `probs` - probabilities to plot ROC Curve.

![alt text](https://sun9-72.userapi.com/impg/4UEpH-Bm9HLWn5zF2-7o7AmnjLq1gnK-Q4SBUg/cLyTRcpQdYI.jpg?size=394x278&quality=96&sign=289e6aff91f0eb76d9195577a861a8fd&type=album)

## Example:

```py
    df = pd.read_csv("heart.csv", sep = ";")
    tar_var = 'Heart'
    tar_val = 'yes'
    dec_tree = DecisionTree(df, tar_var, 'yes', 'auto', 'gini', 10, 1)
    dec_tree.build()
    dec_tree.draw()
    y_vals = df[tar_var].tolist()
    probs = dec_tree.get_probs(df)
    values = dec_tree.build_roc(y_vals, probs)
    print(f"AUC: {values[0]}")
```
