from typing import Iterable, Callable

from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations, chain
import numpy as np

from data.datasets.ebnerd.metrics import (
    intralist_diversity,
    check_key_in_all_nested_dicts,
    compute_combinations,
    get_keys_in_dict,
)

### IntralistDiversity
class IntralistDiversity:
    """
    A class for calculating the intralist diversity metric for recommendations in a recommendation system, as proposed
    by Smyth and McClave in 2001. This metric assesses the diversity within a list of recommendations by computing the
    average pairwise distance between all items in the recommendation list.

    Examples:
        >>> div = IntralistDiversity()
        >>> R = np.array([
                ['item1', 'item2'],
                ['item2', 'item3'],
                ['item3', 'item4']
            ])
        >>> lookup_dict = {
                'item1': {'vector': [0.1, 0.2]},
                'item2': {'vector': [0.2, 0.3]},
                'item3': {'vector': [0.3, 0.4]},
                'item4': {'vector': [0.4, 0.5]}
            }
        >>> lookup_key = 'vector'
        >>> pairwise_distance_function = cosine_distances
        >>> div(R, lookup_dict, lookup_key, pairwise_distance_function)
            array([0.00772212, 0.00153965, 0.00048792])
        >>> div._candidate_diversity(list(lookup_dict), 2, lookup_dict, lookup_key)
            (0.0004879239129211843, 0.02219758592259058)
    """

    def __init__(self) -> None:
        self.name = "intralist_diversity"

    def __call__(
        self,
        R: np.ndarray[np.ndarray[str]],
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
        pairwise_distance_function: Callable = cosine_distances,
    ) -> np.ndarray[float]:
        """
        Calculates the diversity score for each subset of recommendations in `R` using the provided `lookup_dict`
        to find the document vectors and a `pairwise_distance_function` to calculate the diversity. The diversity is
        calculated as the average pairwise distance between all items within each subset of recommendations.

        Args:
            R (np.ndarray[np.ndarray[str]]): A numpy array of numpy arrays, where each inner array contains the IDs
                (as the lookup value in 'lookup_dict') of items for which the diversity score will be calculated.
            lookup_dict (dict[str, dict[str, any]]): A nested dictionary where each key is an item ID and the value is
                another dictionary containing item attributes, including the document vectors identified by `lookup_key`.
            lookup_key (str): The key within the nested dictionaries of `lookup_dict` that corresponds to the document
                vector of each item.
            pairwise_distance_function (Callable, optional): A function that takes two arrays of vectors and returns a
                distance matrix. Defaults to cosine_distances, which measures the cosine distance between vectors.

        Returns:
            np.ndarray[float]: An array of floating-point numbers representing the diversity score for each subset of
                recommendations in `R`.
        """
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        diversity_scores = []
        for sample in R:
            ids = get_keys_in_dict(sample, lookup_dict)
            if len(ids) == 0:
                divesity_score = np.nan
            else:
                document_vectors = np.array(
                    [lookup_dict[id].get(lookup_key) for id in ids]
                )
                divesity_score = intralist_diversity(
                    document_vectors,
                    pairwise_distance_function=pairwise_distance_function,
                )
            diversity_scores.append(divesity_score)
        return np.asarray(diversity_scores)

    def _candidate_diversity(
        self,
        R: np.ndarray[str],
        n_recommendations: int,
        lookup_dict: dict[str, dict[str, any]],
        lookup_key: str,
        pairwise_distance_function: Callable = cosine_distances,
        max_number_combinations: int = 20000,
        seed: int = None,
    ):
        """
        Estimates the minimum and maximum diversity scores for candidate recommendations.

        Args:
            R (np.ndarray[str]): An array of item IDs from which to generate recommendation combinations.
            n_recommendations (int): The number of recommendations per combination to evaluate.
            lookup_dict (dict[str, dict[str, any]]): A dictionary mapping item IDs to their attributes, including the
                vectors identified by `lookup_key` used for calculating diversity.
            lookup_key (str): The key within the attribute dictionaries of `lookup_dict` corresponding to the item
                vectors used in diversity calculations.
            pairwise_distance_function (Callable, optional): A function to calculate the pairwise distance between item
                vectors. Defaults to `cosine_distances`.
            max_number_combinations (int, optional): The maximum number of combinations to explicitly evaluate for
                diversity before switching to random sampling. Defaults to 20000.
            seed (int, optional): A seed for the random number generator to ensure reproducible results when sampling
                combinations. Defaults to None.

        Returns:
            tuple[float, float]: The minimum and maximum diversity scores among the evaluated combinations of
            recommendations.
        """
        #
        check_key_in_all_nested_dicts(lookup_dict, lookup_key)
        R = get_keys_in_dict(R, lookup_dict)
        n_items = len(R)
        if n_recommendations > n_items:
            raise ValueError(
                f"'n_recommendations' cannot exceed the number of items in R (items in candidate list). {n_recommendations} > {n_items}"
            )
        n_combinations = compute_combinations(n_items, n_recommendations)
        # Choose whether to compute or estimate the min-max diversity based on number of combinations to compute:
        if n_combinations > max_number_combinations:
            np.random.seed(seed)
            aids_iterable = chain(
                np.random.choice(R, n_recommendations, replace=False)
                for _ in range(max_number_combinations)
            )
        else:
            aids_iterable = combinations(R, n_recommendations)

        diversity_scores = self.__call__(
            aids_iterable,
            lookup_dict=lookup_dict,
            lookup_key=lookup_key,
            pairwise_distance_function=pairwise_distance_function,
        )
        return diversity_scores.min(), diversity_scores.max()

