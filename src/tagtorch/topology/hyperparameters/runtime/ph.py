"""
Tools to automatically select the largest argument which a **persistent homology solver**
can accept, subject to the requirement that the function runs within a
specified time limit.
"""

import matplotlib.pyplot as plt
import numpy as np
import oat_python as oat
from ripser import ripser
import gudhi
import concurrent.futures
import sys
import time
from .parameter_search import run_with_timeout_process, max_argument_for_runtime_with_binary_search
from .param_search import max_argument_for_runtime_with_linear_search
from .ph_subprocess import _ripser_eps
from tagtorch.topology.ph.utilities import compute_ph_alpha


# ===============================================
# LOW-LEVEL HELPER (WRAPPWER) FUNCTIONS 
# ===============================================

def _ripser_eps(points, max_homology_dimension):
    """
    Returns a function that computes the persistence homology of
    the given points using ripser with specified dimension and epsilon thresholds.
    """
    def func(eps):
        try:
            print(f"[_ripser_eps] Calling ripser with maxdim={max_homology_dimension}, {points.shape[0]} points, thresh={eps}")
            result = ripser(points, maxdim=max_homology_dimension, thresh=eps)
            print(f"[_ripser_eps] ripser returned successfully: {str(result)[:50]}")
            return True # result
        except Exception as e:
            print(f"[_ripser_eps] Exception: {e}")
            raise
    return func

# ===============================================
# LINEAR SEARCH
# ===============================================

def max_ph_epsilon_for_runtime_with_linear_search(
        points, 
        max_homology_dimension, 
        ph_method,
        max_runtime=5,
        epsilons=None,
    ):
    """
    Uses linear search to find the maximum epsilon value for which the persistence homology computation runs within the specified runtime.
    Parameters:
    - points: The point cloud data, formatted as a NumPy array of shape (num_points, ambient_dimension).
    - max_homology_dimension: The maximum homology dimension to compute.
    - ph_method: The method to compute persistence homology (e.g., 'ripser').
    - max_runtime: The maximum allowed runtime in seconds.
    - epsilons: A list of epsilon values to test. The list should be in sorted order.
    Returns:
    - The maximum epsilon value for which the persistence homology computation runs within the specified runtime.
    """

    if ph_method == 'ripser':
        computation_method = _ripser_eps(points, max_homology_dimension)
    elif ph_method == "alpha":
        return None
    else:
        raise ValueError(f"Unknown ph_method: {ph_method}")

    max_param        =  oat.dissimilarity.enclosing_radius_for_points(points)
    epsilons         =  np.linspace(0, max_param, num=20) if epsilons is None else epsilons

    return max_argument_for_runtime_with_linear_search(
        function_to_time    =   computation_method,
        max_runtime         =   max_runtime,
        parameter_values    =   epsilons
    )


# ===============================================
# SOLVER SPECIFIC
# ===============================================


def max_ripser_percent_for_runtime_with_binary_search(points, max_homology_dimension, max_runtime=5, parameter_values=None, mode="hard"):
    """
    Uses binary search to find the maximum epsilon value for which the ripser persistence homology computation runs within the specified runtime.
    Parameters:
    - points: The point cloud data, formatted as a NumPy array of shape (num_points, ambient_dimension).
    - max_homology_dimension: The maximum homology dimension to compute.
    - max_runtime: The maximum allowed runtime in seconds.
    - parameter_values: A list of epsilon values to test. The list should be in sorted order.
    - mode: "hard" or "soft" timeout mode for subprocesses.
    Returns:
    - The maximum epsilon value for which the persistence homology computation runs within the specified runtime.
    """
    computation_method = _ripser_eps(points, max_homology_dimension)
    max_param        =  oat.dissimilarity.enclosing_radius_for_points(points)
    parameter_values  = np.linspace(0, max_param, num=20) if parameter_values is None else parameter_values

    return max_argument_for_runtime_with_binary_search(
        function_to_time    =   computation_method,
        max_runtime         =   max_runtime,
        parameter_sets      =   parameter_values,
        mode                =   mode,
    )


# ===============================================
# SIMULATION ON STANDARD NORMAL DISTRIBUTION
# ===============================================

def get_ph_cutoff_curves_for_standard_normal_with_max_runtime(
        max_runtime=3,
        point_cloud_sizes = np.arange(100, 2000, 5)
    ):
    """
    Computes the maximum epsilon values for which the persistence homology computation runs within
    the specified runtime for a standard normal distribution.

    This function checks

    - point clouds in dimensions 2, 3, and 4
    - homology in dimensions 0, 1, and 2
    - VR and Alpha complexes

    Returns:
        A dictionary `d` of form 
        ```
        d[point_cloud_dimension][method][homology_dimension][point_cloud_size] = max_epsilon_value
        ```
        where `method` is either 'vr' or 'alpha', and `max_epsilon_value` is the largest epsilon for which the PH computation runs within `max_runtime` seconds.
    """
    point_cloud_dimensions = [2, 3, 4]
    homology_dimensions = [0, 1, 2]
    methods = ['vr', 'alpha']
    max_eps = None
    results = {}
    for dim in point_cloud_dimensions:
        results[dim] = {}
        for method in methods:
            results[dim][method] = {}
            for hom_dim in homology_dimensions:
                results[dim][method][hom_dim] = {}
                for counter, size in enumerate(point_cloud_sizes):
                    if counter > 0:
                        if max_eps is None:
                            results[dim][method][hom_dim][size] = None
                            print(f"  Skipping size {size} due to previous timeout.")
                            continue
                    print(f"Testing dimension={dim}, method={method}, homology_dimension={hom_dim}, size={size}")
                    points = np.random.randn(size, dim)  # Standard normal distribution
                    if method == 'vr':
                        max_eps = max_ph_epsilon_for_runtime_with_linear_search(points, hom_dim, ph_method='ripser', max_runtime=max_runtime)
                    else:
                        max_eps = max_ph_epsilon_for_runtime_with_linear_search(points, hom_dim, ph_method='alpha', max_runtime=max_runtime)
                    results[dim][method][hom_dim][size] = max_eps
                    print(f"  Max epsilon for runtime {max_runtime}s: {max_eps}")
    return results

# ===============================================
# PLOTTING FUNCTIONS
# ===============================================

def plot_ph_cutoff_curves(curves_dict):
    """
    Plots the output of get_ph_cutoff_curves_for_standard_normal_with_max_runtime.
    Creates a single figure with subplots (one for each point cloud dimension),
    each with multiple curves (one for each method and homology dimension).
    All subplots share the same axis limits.
    Args:
        curves_dict: output of get_ph_cutoff_curves_for_standard_normal_with_max_runtime
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    dims = sorted(curves_dict.keys())
    n_dims = len(dims)
    # Find global axis limits
    all_sizes = []
    all_epsilons = []
    for dim in dims:
        for method in curves_dict[dim]:
            for hom_dim in curves_dict[dim][method]:
                sizes = sorted(curves_dict[dim][method][hom_dim].keys())
                epsilons = [curves_dict[dim][method][hom_dim][size] for size in sizes]
                all_sizes.extend(sizes)
                all_epsilons.extend(epsilons)
    x_min, x_max = min(all_sizes), max(all_sizes)
    epsilons_filtered = [e for e in all_epsilons if e is not None]
    y_min, y_max = min(epsilons_filtered), max(epsilons_filtered)
    fig, axes = plt.subplots(1, n_dims, figsize=(7*n_dims, 6), squeeze=False)
    for ax_idx, dim in enumerate(dims):
        ax = axes[0, ax_idx]
        color_idx = 0
        for method in curves_dict[dim]:
            for i, hom_dim in enumerate(curves_dict[dim][method]):
                sizes = sorted(curves_dict[dim][method][hom_dim].keys())
                max_epsilons = [curves_dict[dim][method][hom_dim][size] for size in sizes]
                label = f"{method.upper()} $H_{hom_dim}$"
                ax.plot(sizes, max_epsilons, color=colors[color_idx % len(colors)], linestyle=linestyles[i % len(linestyles)], marker='o', label=label)
                color_idx += 1
        ax.set_title(f"Point Cloud Dimension {dim}")
        ax.set_xlabel("Point Cloud Size")
        ax.set_ylabel("Max Epsilon (runtime cutoff)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

# ===============================================
# TOP-LEVEL TOOLS
# ===============================================

def estimate_distance_thresholds_and_vectorize_point_clouds(point_clouds, runtime=5, vr_thresholds=None, vectorizer=None):
    """
    For a list of point clouds, estimate the largest threshold for PH computation in dims 0,1,2 (using ripser),
    then compute PH for each method (ripser, alpha), vectorize, and return the vectors.
    Only thresholds for ripser are estimated (alpha does not use threshold).
    Args:
        point_clouds: list of np.ndarray, each shape (N, d)
        runtime: max allowed runtime per PH computation (seconds)
        vr_thresholds: list of thresholds to test (if None, will use np.linspace)
        vectorizer: callable that takes PH diagrams and returns a vector (if None, returns diagrams)
    Returns:
        dict with keys 'ripser', 'alpha', each value is a list of vectors (or diagrams if no vectorizer)
    """
    # Use the average cloud for threshold estimation
    avg_cloud = point_clouds[len(point_clouds)//2]  # median cloud as representative
    if vr_thresholds is None:
        max_val = np.median( [oat.dissimilarity.enclosing_radius_for_points(pc) for pc in point_clouds] )
        vr_thresholds = np.linspace(0.0, max_val, 40)
    ph_results = {'ripser': [], 'alpha': []}
    ripser_thresholds = {}
    # Estimate thresholds for each dimension (0,1,2)
    for dim in [0,1,2]:
        best_thresh = max_argument_for_runtime_with_linear_search(
            _ripser_eps(avg_cloud, dim),
            max_runtime=runtime,
            parameter_values=vr_thresholds
        )
        ripser_thresholds[dim] = best_thresh
    # Compute PH for each cloud and method
    for pc in point_clouds:
        # Ripser PH (using estimated thresholds)
        ripser_diagrams = []
        for dim in [0,1,2]:
            res = ripser(pc, maxdim=dim, thresh=ripser_thresholds[dim])
            ripser_diagrams.append(res['dgms'])
        # Get the persistence diagram for each dimension
        ripser_diagrams = [dgm[ dim ] for dgms in ripser_diagrams for dim, dgm in enumerate(dgms) if dim < len(dgm) ]
        if vectorizer:
            ph_results['ripser'].append(vectorizer(ripser_diagrams))
        else:
            ph_results['ripser'].append(ripser_diagrams)
        # Alpha PH (if available)
        alpha_complex = gudhi.AlphaComplex(points=pc)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.persistence()
        alpha_diagrams = []
        for dim in [0,1,2]:
            dgm = simplex_tree.persistence_intervals_in_dimension(dim)
            alpha_diagrams.append(dgm)
        if vectorizer:
            ph_results['alpha'].append(vectorizer(alpha_diagrams))
        else:
            ph_results['alpha'].append(alpha_diagrams)
    return ph_results








# ===============================================
# TESTS
# ===============================================



def test_max_ph_epsilon_for_runtime_with_linear_search():
    points              =   np.random.rand(5000, 2)  # Example point cloud
    print("Testing VR")
    max_eps             =   max_ph_epsilon_for_runtime_with_linear_search(
                                points, 
                                max_homology_dimension=1, 
                                ph_method='ripser', 
                                max_runtime=1,
                            )
    print(f"Maximum epsilon value that runs within 1 second (RIPSER): {max_eps}")

    print("Testing ALPHA")
    max_eps             =   max_ph_epsilon_for_runtime_with_linear_search(
                                points, 
                                max_homology_dimension=1, 
                                ph_method='alpha', 
                                max_runtime=1,
                            )
    print(f"Maximum epsilon value that runs within 1 second (ALPHA): {max_eps}")


def test_estimate_distance_thresholds_and_vectorize_point_clouds():
    point_clouds = [np.random.rand(1000, 2) for _ in range(5)]
    results = estimate_distance_thresholds_and_vectorize_point_clouds(point_clouds, runtime=1)
    print("Estimated PH results for 5 point clouds (ripser and alpha):")
    for method in results:
        print(f"Method: {method}")
        for i, res in enumerate(results[method]):
            print(f"  Cloud {i}: {res[:5]}...")  # Print first 5 features/diagrams

