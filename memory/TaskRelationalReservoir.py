import torch
import numpy as np

from memory._Memory import Memory
from sklearn.mixture import BayesianGaussianMixture


class TaskRelationalReservoir(Memory):
    def __init__(self, args):
        super().__init__(args)
        self.clusterer = None

    def save_reservoir(self):
        # Save the reservoir to a local file
        torch.save(self.clusterer.means_, f"{self.logger.log_dir}/reservoir_cluster_means.torch")
        torch.save(self.clusterer.covariances_, f"{self.logger.log_dir}/reservoir_cluster_covs.torch")
        print(f"=> Saving reservoir. Label distribution: {torch.unique(self.buffer['labels'], return_counts=True)}")

    def epoch_update(self, logger, task_counter, model=None):
        # Fit the GMM over both old and new reservoir
        with torch.no_grad():
            device = f"cuda:{self.args.devices[0]}" if self.args.accelerator == "gpu" else "cpu"
            combined_images = torch.vstack([self.buffer["images"], self.current_task_data["images"]])
            combined_labels = torch.vstack([self.buffer["labels"], self.current_task_data["labels"]])
            combined_embeddings = model.dynamics_func.to(device).sample_embeddings(combined_images.unsqueeze(1))

        # Fit Bayesian Gaussian Mixture model over the reservoir, taking their cluster assignments as pseudo-labels
        if self.clusterer is None:
            self.clusterer = BayesianGaussianMixture(n_components=1, max_iter=1000, n_init=3, random_state=self.args.seed)
            self.clusterer = self.clusterer.fit(combined_embeddings.detach().cpu().numpy())

        else:
            """ Refit with n+1 components with previous components """
            # Save the previous weights
            weights1 = self.clusterer.weights_
            means1 = self.clusterer.means_
            covariances1 = self.clusterer.covariances_

            # Rebuild the GMM, adding on another component
            self.clusterer = BayesianGaussianMixture(
                n_components=np.unique(self.buffer["labels"].detach().cpu().numpy()).shape[0] + 1,
                max_iter=1000, n_init=3, weight_concentration_prior=0.01, random_state=self.args.seed
            )

            new_component_mean = combined_embeddings[self.buffer["images"].shape[0]:][
                np.random.choice(self.current_task_data["images"].shape[0])].detach().cpu().numpy()
            new_component_covariance = np.mean(covariances1, axis=0)  # Use the average of the existing covariances

            self.clusterer.weights_ = np.concatenate([weights1 * (1 - 1e-1), [1e-1]])
            self.clusterer.means_ = np.vstack([means1, new_component_mean])
            self.clusterer.covariances_ = np.concatenate([covariances1, [new_component_covariance]])
            self.clusterer.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.clusterer.covariances_))

            # Do the fit
            self.clusterer = self.clusterer.fit(combined_embeddings.detach().cpu().numpy())

            """ Detect inactive components, throw them out """
            # Step 2: Detect inactive components (weights close to zero)
            threshold = 0.05
            active_indices = np.where(self.clusterer.weights_ > threshold)[0]

            # Step 3: Extract parameters of active components
            active_weights = self.clusterer.weights_[active_indices]
            active_means = self.clusterer.means_[active_indices]
            active_covariances = self.clusterer.covariances_[active_indices]

            # Normalize the weights of the active components
            active_weights /= active_weights.sum()

            # Step 4: Initialize a new BayesianGaussianMixture with the number of active components
            self.clusterer = BayesianGaussianMixture(
                n_components=len(active_indices), max_iter=1000, n_init=3,
                weight_concentration_prior=0.01, random_state=self.args.seed
            )

            self.clusterer.weights_ = active_weights
            self.clusterer.means_ = active_means
            self.clusterer.covariances_ = active_covariances
            self.clusterer.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(active_covariances))
            self.clusterer = self.clusterer.fit(combined_embeddings.detach().cpu().numpy())

        # Get cluster assignments to fit GMM
        assignments = self.clusterer.predict(combined_embeddings.detach().cpu().numpy())

        # Save means as torch Tensor
        self.cluster_means = torch.Tensor(self.clusterer.means_).to(self.args.devices[0])

        # Get the number of clusters with a significant representation
        cluster_labels, cluster_counts = np.unique(assignments, return_counts=True)

        # For each cluster predicted, get an equal representation in the reservoir*
        # * making sure the cluster size is decent enough
        indices = []
        for label in cluster_labels:
            cluster_indices = np.where(assignments == label)[0]

            if cluster_indices.shape[0] < self.args.memory_samples // cluster_labels.shape[0]:
                indices.append(cluster_indices)
            else:
                indices.append(np.random.choice(cluster_indices, size=self.args.memory_samples // cluster_labels.shape[0], replace=False))
        indices = np.concatenate(indices)

        # Using these indices, build the new reservoir
        self.buffer["images"] = combined_images[indices]
        self.buffer["labels"] = combined_labels[indices]
        self.buffer["embeddings"] = combined_embeddings[indices]
        self.buffer["cluster_assignments"] = assignments[indices]

        # Wipe the accumulated last-task
        self.current_task_indices = None
        self.current_task_data["images"] = None
        self.current_task_data["labels"] = None

        # Print out the current distribution to stdout
        with open(f"{logger.log_dir}/log.txt", 'a') as f:
            f.write(f"=> GMM Component Weights: {self.clusterer.weights_}\n")
            f.write(f"=> Distribution of labels: {np.unique(self.buffer['labels'].detach().cpu().numpy(), return_counts=True)}\n")
            f.write(f"=> Distribution of cluster assignments: {np.unique(self.buffer['cluster_assignments'], return_counts=True)}\n")

    def batch_update(self, images, labels, task_counter):
        # Initialize vectors at first batch
        if self.age == 0:
            self.buffer["images"] = images[0].unsqueeze(0)
            self.buffer["labels"] = labels[0].unsqueeze(0)
            self.age += 1

        # Just add to buffer if buffer is not filled
        elif self.args.memory_samples - self.age > 0:
            stopping_point = self.args.memory_samples - self.age
            self.buffer["images"] = torch.vstack((self.buffer["images"], images[:stopping_point]))
            self.buffer["labels"] = torch.vstack((self.buffer["labels"], labels[:stopping_point]))
            self.age += self.args.batch_size

        # If current task buffer is not initialized, just assign values
        if self.current_task_data["images"] is None:
            self.current_task_data["images"] = images
            self.current_task_data["labels"] = labels

        elif self.current_task_data["images"].shape[0] < self.args.memory_samples // torch.unique(self.buffer["labels"]).shape[0]:
            self.current_task_data["images"] = torch.vstack((self.current_task_data["images"], images))
            self.current_task_data["labels"] = torch.vstack((self.current_task_data["labels"], labels))

        self.age += self.args.batch_size

    def get_batch(self):
        # Select random indices from the reservoir
        sample_indices = np.random.choice(range(self.buffer["images"].shape[0]), self.args.batch_size // 2, replace=False)

        # Get the corresponding data
        images = self.buffer["images"][sample_indices]
        labels = self.buffer["labels"][sample_indices]
        batch_assignments = self.buffer["cluster_assignments"][sample_indices]
        buffer_assignments = self.buffer["cluster_assignments"]

        # Get the domains for the image based on the prototype it is assigned to
        domains = []
        for pid in batch_assignments:
            domain_indices = np.where(buffer_assignments == pid)[0]
            domains.append(self.buffer["images"][np.random.choice(domain_indices, self.args.domain_size, replace=True)])

        # Stack the domains
        domains = torch.stack(domains)
        return images, domains, labels
