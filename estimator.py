# estimator.py
import json
import joblib
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Sequence


class SplitType(IntEnum):
    numerical = 0
    categorical = 1


@dataclass
class Node:
    left: int
    right: int
    parent: int
    split_idx: int
    split_cond: float
    default_left: bool
    split_type: SplitType
    categories: List[int]


class Tree:
    """Single XGBoost tree with a simple predict_row method."""

    def __init__(self, nodes: Sequence[Node]):
        self.nodes = nodes

    def is_leaf(self, nid: int) -> bool:
        return self.nodes[nid].left == -1

    def predict_row(self, x: np.ndarray) -> float:
        """
        Traverse the tree for a single sample x (1D numpy array),
        returning the leaf value (raw score contribution from this tree).

        Assumes only numerical splits (no categorical splits).
        """
        nid = 0  # start at root
        while True:
            node = self.nodes[nid]

            # Leaf node: the leaf value is stored in split_cond
            if self.is_leaf(nid):
                return float(node.split_cond)

            # Numerical split
            feat_idx = node.split_idx
            val = x[feat_idx]

            if np.isnan(val):
                # Missing value: go to default child
                nid = node.left if node.default_left else node.right
            else:
                # Standard numerical decision
                if val < node.split_cond:
                    nid = node.left
                else:
                    nid = node.right


class XGBJsonModel:
    """
    Minimal XGBoost JSON model parser + predictor.

    Supports:
    - binary:logistic objective
    - numeric features only
    """

    def __init__(self, model_dict: dict):
        learner = model_dict["learner"]
        learner_params = learner["learner_model_param"]

        # base_score is stored as a JSON string, e.g. "[0.5]"
        base_score_list = json.loads(learner_params["base_score"])
        # Binary classification => one group
        self.base_score = float(base_score_list[0])

        # Info about trees
        gbm = learner["gradient_booster"]["model"]
        self.tree_info = gbm["tree_info"]  # group index per tree (all 0 for binary)
        model_shape = gbm["gbtree_model_param"]
        num_trees = int(model_shape["num_trees"])
        j_trees = gbm["trees"]
        assert len(j_trees) == num_trees

        trees: List[Tree] = []
        for i in range(num_trees):
            tree = j_trees[i]
            tree_id = int(tree["id"])
            assert tree_id == i

            left_children = tree["left_children"]
            right_children = tree["right_children"]
            parents = tree["parents"]
            split_conditions = tree["split_conditions"]
            split_indices = tree["split_indices"]

            # default_left and split_type can be bytes or ints, normalize to ints
            def to_ints(data):
                return [int(v) for v in data]

            default_left = to_ints(tree["default_left"])
            split_types = to_ints(tree["split_type"])

            # categorical storage (we'll parse but not actually use, since
            # you have only continuous features)
            cat_segments = tree["categories_segments"]
            cat_sizes = tree["categories_sizes"]
            cat_nodes = tree["categories_nodes"]
            cats = tree["categories"]

            node_categories: List[List[int]] = []
            cat_cnt = 0
            last_cat_node = cat_nodes[cat_cnt] if cat_nodes else -1

            for node_id in range(len(left_children)):
                if node_id == last_cat_node:
                    beg = cat_segments[cat_cnt]
                    size = cat_sizes[cat_cnt]
                    end = beg + size
                    node_cats = cats[beg:end]
                    node_categories.append(node_cats)
                    cat_cnt += 1
                    if cat_cnt == len(cat_nodes):
                        last_cat_node = -1
                    else:
                        last_cat_node = cat_nodes[cat_cnt]
                else:
                    node_categories.append([])  # numerical or leaf

            nodes: List[Node] = []
            for node_id in range(len(left_children)):
                nodes.append(
                    Node(
                        left=left_children[node_id],
                        right=right_children[node_id],
                        parent=parents[node_id],
                        split_idx=split_indices[node_id],
                        split_cond=split_conditions[node_id],
                        default_left=(default_left[node_id] == 1),
                        split_type=SplitType(split_types[node_id]),
                        categories=node_categories[node_id],
                    )
                )

            trees.append(Tree(nodes))

        self.trees = trees

    def predict_proba(self, X):
        """
        X: pandas DataFrame or numpy array with the same feature order as training.
        Returns: array of shape (n_samples, 2) with [P(class 0), P(class 1)].
        """
        X_arr = np.asarray(X, dtype=float)
        n_samples = X_arr.shape[0]

        # start from base_score for all samples
        margins = np.full(n_samples, self.base_score, dtype=float)

        for tree in self.trees:
            for i in range(n_samples):
                margins[i] += tree.predict_row(X_arr[i])

        # binary:logistic -> sigmoid
        probs_pos = 1.0 / (1.0 + np.exp(-margins))
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])


def estimator(model_path="logit_model_1.joblib"):
    """
    Load either:
    - a legacy joblib model if path ends with .joblib
    - an XGBoost JSON model (xgb_model.json) and wrap it as XGBJsonModel
    """
    if model_path.endswith(".json"):
        with open(model_path, "r") as f:
            model_dict = json.load(f)
        return XGBJsonModel(model_dict)
    else:
        # fall back to old behavior
        return joblib.load(model_path)