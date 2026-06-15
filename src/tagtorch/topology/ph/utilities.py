"""
Utilities and convenience methods for computing persistent homology.
"""

import gudhi

def compute_ph_alpha(
        points,
        maxdim=1,
    ):
    """
    Computes the persistence homology of a point cloud using the alpha complex.
    Parameters:
        - points: A numpy array of shape (n_samples, n_features) representing the point cloud
        - maxdim: The maximum homology dimension to compute homology (default: 1)
    Returns:
        - A list of persistence tuples (dimension, (birth, death)) for each homology dimension
    """
    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.prune_above_dimension(maxdim+1)
    return simplex_tree.persistence()


def print_first_n_characters(obj, n=1000):
    """
    Utility function to print the first n characters of an object (e.g., list, array).
    """
    characters = str(obj)
    print(characters[:n])


def default_params():
    """
    Parameter schemas and defaults for TDA/vectorization methods.
    """
    return {
        'homology_dimensions': [0,1],
        'persistence_image': {'resolution': [20,20], 'sigma': 0.1},
        'persistence_landscape': {'n_layers': 5}
    }    