from math import comb
from typing import Callable

from sklearn.metrics.pairwise import cosine_distances
import numpy as np



def intralist_diversity(
    R: np.ndarray[np.ndarray],
    pairwise_distance_function: Callable = cosine_distances,
) -> float:
    """Calculate the intra-list diversity of a recommendation list.

    This function implements the method described by Smyth and McClave (2001) to
    measure the diversity within a recommendation list. It calculates the average
    pairwise distance between all items in the list.

    Formula:
        Diversity(R) = ( sum_{i∈R} sum_{j∈R_{i}} dist(i, j) )  / ( |R|(|R|-1) )

    where `R` is the recommendation list, and `dist` represents the pairwise distance function used.

    Args:
        R (np.ndarray[np.ndarray]): A 2D numpy array where each row represents a recommendation.
            This array should be either array-like or a sparse matrix, with shape (n_samples_X, n_features).
        pairwise_distance_function (Callable, optional): A function to compute pairwise distance
            between samples. Defaults to `cosine_distances`.

    Returns:
        float: The calculated diversity score. If the recommendation list contains less than or
            equal to one item, NaN is returned to signify an undefined diversity score.

    References:
        Smyth, B., McClave, P. (2001). Similarity vs. Diversity. In: Aha, D.W., Watson, I. (eds)
        Case-Based Reasoning Research and Development. ICCBR 2001. Lecture Notes in Computer Science(),
        vol 2080. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-44593-5_25

    Examples:
        >>> R1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> print(intralist_diversity(R1))
            0.022588438516842262
        >>> print(intralist_diversity(np.array([[0.1, 0.2], [0.1, 0.2]])))
            1.1102230246251565e-16
    """
    R_n = R.shape[0]  # number of recommendations
    if R_n <= 1:
        # Less than or equal to 1 recommendations in recommendation list
        diversity = np.nan
    else:
        pairwise_distances = pairwise_distance_function(R, R)
        diversity = np.sum(pairwise_distances) / (R_n * (R_n - 1))
    return diversity


def check_key_in_all_nested_dicts(dictionary: dict, key: str) -> None:
    """
    Checks if the given key is present in all nested dictionaries within the main dictionary.
    Raises a ValueError if the key is not found in any of the nested dictionaries.

    Args:
        dictionary (dict): The dictionary containing nested dictionaries to check.
        key (str): The key to look for in all nested dictionaries.

    Raises:
        ValueError: If the key is not present in any of the nested dictionaries.

    Example:
        >>> nested_dict = {
                "101": {"name": "Alice", "age": 30},
                "102": {"name": "Bob", "age": 25},
            }
        >>> check_key_in_all_nested_dicts(nested_dict, "age")
        # No error is raised
        >>> check_key_in_all_nested_dicts(nested_dict, "salary")
        # Raises ValueError: 'salary is not present in all nested dictionaries.'
    """
    for dict_key, sub_dict in dictionary.items():
        if not isinstance(sub_dict, dict) or key not in sub_dict:
            raise ValueError(
                f"'{key}' is not present in '{dict_key}' nested dictionary."
            )

def compute_combinations(n: int, r: int) -> int:
    """Compute Combinations where order does not matter (without replacement)

    Source: https://www.statskingdom.com/combinations-calculator.html
    Args:
        n (int): number of items
        r (int): number of items being chosen at a time
    Returns:
        int: number of possible combinations

    Formula:
    * nCr = n! / ( (n - r)! * r! )

    Assume the following:
    * we sample without replacement of items
    * order of the outcomes does NOT matter
    """
    return comb(n, r)

def get_keys_in_dict(id_list: any, dictionary: dict) -> list[any]:
    """
    Returns a list of IDs from id_list that are keys in the dictionary.
    Args:
        id_list (List[Any]): List of IDs to check against the dictionary.
        dictionary (Dict[Any, Any]): Dictionary where keys are checked against the IDs.

    Returns:
        List[Any]: List of IDs that are also keys in the dictionary.

    Examples:
        >>> get_keys_in_dict(['a', 'b', 'c'], {'a': 1, 'c': 3, 'd': 4})
            ['a', 'c']
    """
    return [id_ for id_ in id_list if id_ in dictionary]