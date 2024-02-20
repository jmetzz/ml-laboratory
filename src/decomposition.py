import numpy as np
from matplotlib import pyplot as plt


def pca(instances, n_components=None) -> np.ndarray:
    """
    Principal Component Analysis (PCA) is a dimensionality reduction technique
    that identifies the directions (principal components)
    that maximize the variance in the data.

    It involves a few key steps which can be efficiently handled with linear algebra.
    1. standardize the data
    2. compute the covariance matrix
    3. compute the eigenvectors and eigenvalues of teh covariance matrix
    4. sort the eigenvectors by decreasing eigenvalues
    5. project the data onto the principal (selected) components

    Keep in mind PCA is a linear dimensionality reduction method.

    Some assumptions:
    * the data is standardized, since PCA is affected by scale. Thus,
    one need to scale the features in the data before applying PCA.
    Standardization can be achieved with mean of 0 and a standard deviation of 1.

    * variables need to have a linear relationship

    Args:
        instances (np.ndarray): The dataset, with instances as rows and features as columns.
        n_components (int): The number of principal components to return.
            If None, return all components

    Returns:
        np.ndarray representing the first n principal components.
    """
    # 1. standardize the data
    x_std = (instances - np.mean(instances, axis=0)) / np.std(instances, axis=0)

    # 2. compute the covariance matrix
    cov_matrix = np.cov(x_std.T)

    # 3. compute the eigenvectors and eigenvalues of teh covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. sort the eigenvectors by decreasing eigenvalues
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    # 5. project the data onto the principal (selected) components
    # Select the first n_components eigenvectors (principal components)
    n_components = n_components or len(eigenvectors)
    principal_components = eigenvectors[:, :n_components]
    x_pca = np.dot(x_std, principal_components)
    return x_pca


def tsne_gradient_descent(P, initial_Y, learning_rate=200.0, n_iterations=1000, momentum=0.8, learning_rate_decay=0.95):
    """
    Perform gradient descent to minimize the KL divergence in t-SNE.
    (t-distributed stochastic neighbor embedding)

    This is an implementation of an over-simplified version of t-SNE.
    It won't be as efficient or robust as the implementations available
    in libraries like scikit-learn, and is provided for conceptual understanding only.

    Non-linear and non-deterministic method for dimensionality reduction used for
    visualization of high-dimensional datasets.

    t-SNE is a nonlinear technique that preserves local distances between points,
    making it excellent for visualizing clusters or groups in the data.

    High-level overview of the t-SNE process:

    1. Compute pairwise affinities of points in the high-dimensional space
    with a Gaussian distribution centered on each point. Perplexity, a key hyperparameter,
    influences the variance of the Gaussians and, hence, the scale of local neighborhoods.

    2. Define a similar distribution in the low-dimensional space but using a
    Student’s t-distribution to compute affinities between points.
    The t-distribution has heavier tails than a Gaussian, allowing distant points in
    the high-dimensional space to be modeled more accurately as distant in
    the low-dimensional map.

    3. Minimize the Kullback-Leibler (KL) divergence between the two distributions
    with respect to the positions of points in the low-dimensional space. This step is
    usually done using gradient descent. The KL divergence measures how one
    probability distribution diverges from a second, expected probability distribution.


    Args:
        P (np.ndarray): Pairwise affinities in the high-dimensional space, shape (n_samples, n_samples).
        initial_Y (np.ndarray): Initial low-dimensional representations, shape (n_samples, n_components).
        learning_rate (float): The learning rate for gradient descent.
        n_iterations (int): Number of iterations to run the gradient descent.
        momentum (float): Momentum term to smooth updates.
        learning_rate_decay (float): The rate of learning rate decay per iteration.


    Returns:
        np.ndarray: Optimized low-dimensional representations.

    """
    Y = initial_Y.copy()
    Y_velocity = np.zeros(Y.shape)  # Initialize velocities for momentum

    for _ in range(n_iterations):
        Q = compute_low_dim_affinities(Y)
        PQ_diff = P - Q

        # Gradient computation
        grads = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            grads[i, :] = 4 * np.dot(PQ_diff[i, :] + PQ_diff[:, i], (Y[i, :] - Y))

        # Apply momentum
        Y_velocity = momentum * Y_velocity - learning_rate * grads
        # Update Y
        Y += Y_velocity

        # Apply adaptive learning rate
        learning_rate *= learning_rate_decay

        # Optional: Implement early exaggeration phase here

    return Y


def compute_pairwise_affinities(X, perplexity=30.0):
    """
    Compute pairwise affinities in high-dimensional space using a Gaussian distribution.

    Args:
        X (np.ndarray): The high-dimensional data, shape (n_samples, n_features).
        perplexity (float): The perplexity controls the balance between local and global aspects of the data.

    Returns:
        np.ndarray: Pairwise affinities of shape (n_samples, n_samples).
    """
    n_samples, _ = X.shape
    # Euclidean distances squared
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Perplexity to variance
    P = np.zeros((n_samples, n_samples))
    beta = np.ones((n_samples, 1))
    logU = np.log(perplexity)

    # Compute affinities with binary search for precision
    for i in range(n_samples):
        # Diagonal elements are zero
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n_samples]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Binary search for the optimal sigma
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > 1e-5 and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n_samples]))] = thisP

    # Symmetrize and normalize
    P = (P + P.T) / (2 * n_samples)
    return P


def Hbeta(D, beta):
    """
    Compute the entropy and probability for a given precision (1/variance).

    Args:
        D (np.ndarray): Squared distances for a particular data point.
        beta (float): Precision parameter for the Gaussian distribution.

    Returns:
        H (float): The entropy of the distribution.
        P (np.ndarray): The probabilities for each data point.
    """
    P = np.exp(-D * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def compute_low_dim_affinities(Q):
    """
    Compute pairwise affinities in low-dimensional space using a Student-t distribution.

    In the low-dimensional space, we use a Student-t distribution to compute the pairwise affinities,
    which helps mitigate the "crowding problem" by allowing repulsion between distant clusters.

    Args:
        Q (np.ndarray): Low-dimensional representations, shape (n_samples, n_components).

    Returns:
        np.ndarray: Pairwise affinities, shape (n_samples, n_samples).
    """
    (n_samples, _) = Q.shape
    sum_Q = np.sum(np.square(Q), 1)
    Q_D = np.add(np.add(-2 * np.dot(Q, Q.T), sum_Q).T, sum_Q)
    Q = 1 / (1 + Q_D)
    np.fill_diagonal(Q, 0)  # Set diagonal to zero
    Q /= np.sum(Q)  # Normalize
    return Q


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    # Generate a synthetic dataset
    n_samples = 100

    # Generate some data
    X = np.random.rand(n_samples, 5)
    X_transformed = pca(X, n_components=2)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)

    # t-SNE example
    # Generate a synthetic dataset
    # Create two clusters in high-dimensional space
    n_features = 50  # High-dimensional space
    cluster_1 = np.random.normal(loc=-2.5, scale=1.0, size=(n_samples // 2, n_features))
    cluster_2 = np.random.normal(loc=2.5, scale=1.0, size=(n_samples // 2, n_features))
    X_high_dim = np.vstack([cluster_1, cluster_2])

    n_components = 3  # Target low-dimensional space
    # Random initialization of low-dimensional space
    Y_init = np.random.normal(loc=0.0, scale=0.01, size=(n_samples, n_components))

    pairwise_affinity = compute_pairwise_affinities(X_high_dim, perplexity=30.0)
    Y_low_dim = tsne_gradient_descent(
        pairwise_affinity, Y_init, learning_rate=200.0, n_iterations=10, momentum=0.8, learning_rate_decay=0.95
    )

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="3d")

    ax.scatter(Y_low_dim[:, 0], Y_low_dim[:, 1], Y_low_dim[:, 2], alpha=0.6)
    ax.set_title("3D t-SNE visualization of synthetic dataset")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.show()
