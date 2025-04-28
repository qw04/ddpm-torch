import numpy as np
from time import time
import torch


# improved precision and recall github code
def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000):
    state = dict()
    num_samples = ref_features.shape[0]
   
    ref_manifold = ManifoldEstimator(ref_features, row_batch_size, col_batch_size, nhood_sizes) 
    eval_manifold = ManifoldEstimator(eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_samples)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state


class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3]):
        """Estimate the manifold of given feature vectors.
        
            Args:
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] = self.pairwise_distances(row_batch, col_batch)
    
            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)[:, self.nhood_sizes]

    def evaluate(self, eval_features):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = self.pairwise_distances(feature_batch, ref_batch)

            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

        return batch_predictions


    def pairwise_distances(self, U, V):
        U = torch.tensor(U, dtype=torch.float32)
        V = torch.tensor(V, dtype=torch.float32)
        
        norm_u = torch.sum(U**2, dim=1)
        norm_v = torch.sum(V**2, dim=1)
        
        norm_u = norm_u.view(-1, 1)
        norm_v = norm_v.view(-1, 1)

        D = torch.clamp(norm_u - 2 * torch.matmul(U, V.T) + norm_v, min=0.0)
        return D
    
if __name__ == "__main__":
    # Example usage of the knn_precision_recall_features function
    ref_features = np.random.rand(2000, 750)  # Replace with actual reference features
    eval_features = np.random.rand(2000, 750)  # Replace with actual evaluation features

    state = knn_precision_recall_features(ref_features, eval_features)
    print("reference features:", ref_features)
    print("evaluation features:", eval_features)

    print("Precision:", state['precision'])
    print("Recall:", state['recall'])