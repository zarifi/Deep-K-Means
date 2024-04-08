import numpy as np
from sklearn.cluster import KMeans
# from libKMCUDA import kmeans_cuda
# from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms, squared_norm
from numpy.random import RandomState
import time


# def k_means_gpu_cuml(weight_vector, n_clusters, seed=int(time.time()), gpu_id=0):
# 	"""
# 		Perform K-Means clustering on the GPU using RAPIDS cuML library.

# 		Parameters:
# 		- weight_vector: The input weight vectors to cluster.
# 		- n_clusters: The number of clusters to form.
# 		- seed: A seed for initializing the centers.
# 		- gpu_id: ID of the GPU to use.

# 		Returns:
# 		- centers and labels
# 		"""
# 	# Perform clustering
# 	cuml_kmeans = cuKMeans(n_clusters=n_clusters, init="k-means++", random_state=seed)
# 	cuml_kmeans.fit(weight_vector)
# 	labels = cuml_kmeans.labels_
# 	centers = cuml_kmeans.cluster_centers_

# 	return centers, labels

def initialize_centroids(X, n_clusters, random_state=None):
    """
    Initialize centroids using the 'k-means++' method from scikit-learn's KMeans.

    Parameters:
    - X: Data array of shape (n_samples, n_features)
    - n_clusters: The number of centroids to initialize.
    - random_state: Seed or random state for reproducibility.

    Returns:
    - centroids: Initialized centroids of shape (n_clusters, n_features).
    """
    # Create a KMeans instance with n_init=1 and max_iter=1 for initialization purposes
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=1, random_state=random_state)
    
    # Dummy fit to initialize centroids
    kmeans.fit(X)
    
    # The initialized centroids
    centroids = kmeans.cluster_centers_
    
    return centroids

def k_means_cpu(weight_vector, n_clusters, seed=int(time.time())):

	kmeans_result = KMeans(n_clusters=n_clusters, init='k-means++', random_state = seed).fit(weight_vector)
	labels = kmeans_result.labels_
	centers = kmeans_result.cluster_centers_
	weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
	for v in range(weight_vector.shape[0]):
		weight_vector_compress[v, :] = centers[labels[v], :]
	# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
	return weight_vector_compress

def k_means_gpu(weight_vector, n_clusters, verbosity=0, seed=int(time.time()), gpu_id=7):

	if n_clusters == 1:

		mean_sample = np.mean(weight_vector, axis=0)

		weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

		return weight_vector

	elif weight_vector.shape[0] == n_clusters:

		return weight_vector

	elif weight_vector.shape[1] == 1:

		return k_means_cpu(weight_vector, n_clusters, seed=seed)

	else:
		# print('n_clusters', n_clusters)
		# print('weight_vector.shape',weight_vector.shape)
		# print('kmeans++ init start')
		# init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector, n_clusters=n_clusters, x_squared_norms=row_norms(weight_vector, squared=True), random_state=RandomState(seed))
		init_centers = initialize_centroids(X=weight_vector, n_clusters=n_clusters, random_state=RandomState(seed))
		# # print('kmeans++ init finished')
		# # print('init_centers.shape',init_centers.shape)
		centers, labels = k_means_gpu_cuml(weight_vector, n_clusters, seed, gpu_id)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)

		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
		weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
		for v in range(weight_vector.shape[0]):
			weight_vector_compress[v, :] = centers[labels[v], :]
		# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
		return weight_vector_compress

# def k_means_gpu_sparsity(weight_vector, n_clusters, ratio=0.5, verbosity=0, seed=int(time.time()), gpu_id=0):

# 	# print(n_clusters)
# 	if ratio == 0:

# 		return k_means_gpu(weight_vector=weight_vector, n_clusters=n_clusters, seed=seed, gpu_id=gpu_id)

# 	if ratio == 1:

# 		if n_clusters == 1:

# 			mean_sample = np.mean(weight_vector, axis=0)

# 			weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

# 			return weight_vector

# 		elif weight_vector.shape[0] == n_clusters:

# 			return weight_vector

# 		else:
# 			# mean_sample = np.mean(weight_vector, axis=0)
# 			weight_vector_1_mean = np.mean(weight_vector, axis=0)

# 			weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
# 			for v in weight_vector.shape[0]:
# 				weight_vector_compress[v, :] = weight_vector_1_mean

# 			return weight_vector_compress

# 	else:

# 		if n_clusters == 1:

# 			mean_sample = np.mean(weight_vector, axis=0)

# 			weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))

# 			return weight_vector

# 		elif weight_vector.shape[0] == n_clusters:

# 			return weight_vector

# 		elif weight_vector.shape[1] == 1:

# 			return k_means_sparsity(weight_vector, n_clusters, ratio, seed=seed)

# 		else:
# 			# print('n_clusters', n_clusters)
# 			# print('weight_vector.shape',weight_vector.shape)
# 			# print('kmeans++ init start')
# 			num_samples = weight_vector.shape[0]
# 			mean_sample = np.mean(weight_vector, axis=0)

# 			center_cluster_index = np.argsort(np.linalg.norm(weight_vector - mean_sample, axis=1))[
# 								   :int(num_samples * ratio)]
# 			# weight_vector_1 = weight_vector[min_index, :]
# 			weight_vector_1_mean = np.mean(weight_vector[center_cluster_index, :], axis=0)

# 			remaining_cluster_index = np.asarray([i for i in np.arange(num_samples) if i not in center_cluster_index])

# 			weight_vector_train = weight_vector[remaining_cluster_index, :]
# 			# weight_vector_train = [element for i, element in enumerate(weight_vector) if i not in min_index]
# 			# weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))
# 			init_centers = initialize_centroids(X=weight_vector_train, n_clusters=n_clusters - 1, random_state=RandomState(seed))
# 			# init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector_train, n_clusters=n_clusters - 1,
# 			# 												x_squared_norms=row_norms(weight_vector_train,
# 			# 																		  squared=True),
# 			# 												random_state=RandomState(seed))
# 			# # print('kmeans++ init finished')
# 			# # print('init_centers.shape',init_centers.shape)
# 			centers, labels = k_means_gpu_cuml(weight_vector_train, n_clusters - 1, seed, gpu_id)
# 			# centers, labels = kmeans_cuda(samples=weight_vector_train, clusters=n_clusters - 1, init=init_centers,
# 			# 							  yinyang_t=0,
# 			# 							  seed=seed, device=gpu_id, verbosity=verbosity)
# 			# print(np.unique(labels, axis=0).shape[0]+1)
# 			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
# 			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
# 			# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
# 			weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)
# 			for v in center_cluster_index:
# 				weight_vector_compress[v, :] = weight_vector_1_mean

# 			for i, v in enumerate(remaining_cluster_index):
# 				weight_vector_compress[v, :] = centers[labels[i], :]
# 			# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
# 			# print(np.unique(weight_vector_compress, axis=0).shape[0])
# 			# print(n_clusters, '\n')
# 			# assert np.unique(weight_vector_compress, axis=0).shape[0]==n_clusters, "cluster number mismatch"
# 			return weight_vector_compress

def k_means_sparsity(weight_vector, n_clusters, ratio, seed=int(time.time())):

	num_samples = weight_vector.shape[0]
	mean_sample = np.mean(weight_vector, axis=0)

	center_cluster_index = np.argsort(np.linalg.norm(weight_vector - mean_sample, axis=1))[:int(num_samples * ratio)]
	# weight_vector_1 = weight_vector[min_index, :]
	weight_vector_1_mean = np.mean(weight_vector[center_cluster_index, :], axis=0)

	remaining_cluster_index = np.asarray([i for i in np.arange(num_samples) if i not in center_cluster_index])

	weight_vector_train = weight_vector[remaining_cluster_index, :]
	# weight_vector_train = [element for i, element in enumerate(weight_vector) if i not in min_index]
	# weight_vector = np.tile(mean_sample, (weight_vector.shape[0], 1))
	# init_centers = sklearn.cluster.k_means_._k_init(X=weight_vector_train, n_clusters=n_clusters-1,
	# 												x_squared_norms=row_norms(weight_vector_train, squared=True),
	# 												random_state=RandomState(seed))
	# # # print('kmeans++ init finished')
	# # # print('init_centers.shape',init_centers.shape)
	# centers, labels = kmeans_cuda(samples=weight_vector_train, clusters=n_clusters-1, init=init_centers, yinyang_t=0,
	# 							  seed=seed, device=gpu_id, verbosity=verbosity)
	# print(np.unique(labels, axis=0).shape[0]+1)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="k-means++", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="random", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	# centers, labels = kmeans_cuda(samples = weight_vector, clusters = n_clusters, init="afk-mc2", yinyang_t=0, seed=seed, device=gpu_id, verbosity=verbosity)
	kmeans_result = KMeans(n_clusters=n_clusters, init='k-means++',
						   random_state=seed).fit(weight_vector_train)
	labels = kmeans_result.labels_
	centers = kmeans_result.cluster_centers_
	weight_vector_compress = np.zeros((weight_vector.shape[0], weight_vector.shape[1]), dtype=np.float32)

	for i, v in enumerate(remaining_cluster_index):
		weight_vector_compress[v, :] = centers[labels[i], :]

	for v in center_cluster_index:
		weight_vector_compress[v, :] = weight_vector_1_mean
	# for i, v in enumerate(remaining_cluster_index):
	# 	weight_vector_compress[v, :] = centers[labels[i], :]
	# weight_compress = np.reshape(weight_vector_compress, (filters_num, filters_channel, filters_size, filters_size))
	# print(np.unique(weight_vector_compress, axis=0).shape[0])
	# print(n_clusters, '\n')
	# assert np.unique(weight_vector_compress, axis=0).shape[0]==n_clusters, "cluster number mismatch"
	return weight_vector_compress
