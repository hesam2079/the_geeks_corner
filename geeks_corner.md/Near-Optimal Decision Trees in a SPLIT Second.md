
	Varun Babbar ...
	42nd international conference on machine learning, Vancouver, Canada, 2025
## 1. Abstract

This paper introduces **SPLIT**, **LicketySPLIT**, and **RESPLIT**, a family of decision tree algorithms that achieve near-optimal performance with significantly reduced runtime compared to existing optimal methods. By combining greedy splitting beyond a shallow "lookahead" depth with dynamic programming and branch-and-bound, these algorithms offer a scalable way to construct sparse and accurate trees while enabling efficient approximation of the Rashomon set of near-optimal models. The proposed methods are shown to be orders of magnitude faster than state-of-the-art optimal decision tree algorithms, with negligible loss in accuracy or sparsity.

---

## 2. Conclusion

The paper presents **SPLIT**, **LicketySPLIT**, and **RESPLIT**, which bridge the gap between fast greedy decision tree algorithms and slower but more accurate optimal ones. These methods provide a scalable solution for learning interpretable, sparse trees while also enabling efficient exploration of the Rashomon set. The key contribution lies in their ability to maintain high performance while drastically reducing computational cost. Future work includes extending the framework to multiclass classification, regression, and real-time applications requiring rapid model updates.

---

## 3. Literature Review & Key Concepts

### Related Works:
| Term | Meaning |
|------|---------|
| **CART (Breiman, 1984)** | Greedy algorithm for building binary decision trees using Gini impurity as a split criterion. |
| **GOSDT (McTavish et al., 2022)** | Optimal decision tree algorithm using branch-and-bound with theoretical guarantees on sparsity and accuracy. |
| **Murtree (Demirovic et al., 2022)** | Dynamic programming-based method for finding optimal decision trees under constraints on leaf count. |
| **DL8.5 (Aglin et al., 2020)** | Branch-and-bound algorithm for optimal decision trees with pruning techniques for efficiency. |
| **Rashomon Set (Breiman, 2001)** | Set of all models with nearly identical predictive performance, used for model interpretability and robustness analysis. |

### Definitions and Concepts:

| Concept | One-Line Explanation |
|--------|-----------------------|
| **Lookahead Depth** | Maximum depth up to which the algorithm optimizes splits; beyond this, splits are chosen greedily. |
| **Branch-and-Bound** | Optimization technique that explores the space of possible trees by pruning branches with suboptimal bounds. |
| **Sparsity Penalty ($\lambda$)** | Regularization parameter penalizing the number of leaves in the tree. |
| **Greedy Splitting** | Choosing the best feature at each node without considering future splits. |
| **Dynamic Programming Tree Search** | Efficient search strategy that reuses computation across overlapping subproblems in tree construction. |
| **Rashomon Approximation** | Estimating the set of near-optimal decision trees for model robustness and interpretability. |
| **0-1 Loss** | Classification error metric where prediction is either correct (0) or incorrect (1). |
| **Subproblem Caching** | Storing intermediate results to avoid recomputation and reduce runtime. |

---

## 4. Main Idea

The main idea is to use a hybrid approach: optimize splits using branch-and-bound only up to a shallow lookahead depth and apply greedy decisions beyond it, resulting in a scalable yet near-optimal decision tree learner.

---

## 5. Algorithms

### Algorithm 1: `get_bounds(D, dl, d)`
```python
def get_bounds(D, dl, d):
    if d == dl:
        Tg = Greedy(D, d - dl, λ)
        lb = ub = λ * #leaves(Tg) + error(Tg)
    else:
        lb = 2λ
        ub = λ + min(positive_ratio, negative_ratio)
    return lb, ub
```

### Algorithm 2: **SPLIT**
```python
def SPLIT(D, dl, d_budget, λ):
    if d_budget == dl:
        return Greedy(D, d_budget - dl, λ)
    else:
        for each feature f in F:
            D_left = D(f), D_right = D(not f)
            lb_left, ub_left = get_bounds(D_left, dl, d_budget - 1)
            lb_right, ub_right = get_bounds(D_right, dl, d_budget - 1)
            total_ub = ub_left + ub_right
            if total_ub < best_ub:
                best_split = f
        left_tree = SPLIT(D_left, dl, d_budget - 1, λ)
        right_tree = SPLIT(D_right, dl, d_budget - 1, λ)
        return Node(feature=best_split, left=left_tree, right=right_tree)
```

### Algorithm 3: **LicketySPLIT**
```python
def LicketySPLIT(D, dl, d_budget, λ):
    if d_budget == dl:
        return Greedy(D, d_budget - dl, λ)
    else:
        best_gain = -inf
        for each feature f in F:
            gain = compute_gain(f, D)
            if gain > best_gain:
                best_gain = gain
                best_split = f
        D_left = D(f), D_right = D(not f)
        left_tree = LicketySPLIT(D_left, dl, d_budget - 1, λ)
        right_tree = LicketySPLIT(D_right, dl, d_budget - 1, λ)
        return Node(feature=best_split, left=left_tree, right=right_tree)
```

### Algorithm 4: **Greedy Baseline**
```python
def Greedy(D, d, λ):
    if d == 0:
        return Leaf(majority_label(D))
    else:
        best_gain = -inf
        for each feature f in F:
            gain = compute_gain(f, D)
            if gain > best_gain:
                best_gain = gain
                best_split = f
        D_left = D(f), D_right = D(not f)
        left_tree = Greedy(D_left, d - 1, λ)
        right_tree = Greedy(D_right, d - 1, λ)
        return Node(feature=best_split, left=left_tree, right=right_tree)
```

### Algorithm 5: **RESPLIT (Rashomon Set Approximation)**
```python
def RESPLIT(D, dl, d_budget, λ, k):
    trees = []
    for _ in range(k):
        tree = SPLIT(D, dl, d_budget, λ)
        trees.append(tree)
        λ += ε  # perturb λ slightly to encourage diversity
    return trees
```

### Algorithm 6: **Tree Enumeration via RESPLIT Indexing**
```python
def RESPLIT_indexing(resplit_obj):
    hash_map = {}
    tcount, pc = RESPLIT_set_count(resplit_obj)
    start = 0
    for i in range(len(pc)):
        if i > 0:
            start = pc[i - 1] + 1
        end = pc[i]
        prefix_tree = resplit_obj.prefix_list[i]
        for local_idx in range(end - start):
            subtree = locate_subtree(prefix_tree, local_idx)
            global_idx = start + local_idx
            hash_map[global_idx] = subtree
    return hash_map
```

---
