import torch
import torch.nn.functional as F


class _BaseEdgeFilter(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, graph):
        edge_similarity, edge_suspicious_score = self.compute_edge_statistics(graph)
        edge_mask = edge_similarity >= self.threshold
        node_suspicious_score = self._aggregate_node_scores(
            graph.num_nodes,
            graph.edge_index,
            edge_suspicious_score,
            graph.x.device,
            graph.x.dtype,
        )
        node_mask = node_suspicious_score <= (1.0 - self.threshold)
        return {
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "node_score": node_suspicious_score,
            "edge_score": edge_suspicious_score,
            "edge_similarity": edge_similarity,
        }

    def compute_edge_statistics(self, graph):
        raise NotImplementedError

    def _aggregate_node_scores(self, num_nodes, edge_index, edge_scores, device, dtype):
        node_scores = torch.zeros(num_nodes, device=device, dtype=dtype)
        node_counts = torch.zeros(num_nodes, device=device, dtype=dtype)
        if edge_scores.numel() == 0:
            return node_scores

        src, dst = edge_index
        node_scores.scatter_add_(0, src, edge_scores)
        node_scores.scatter_add_(0, dst, edge_scores)
        ones = torch.ones_like(edge_scores, dtype=dtype)
        node_counts.scatter_add_(0, src, ones)
        node_counts.scatter_add_(0, dst, ones)

        nonzero = node_counts > 0
        node_scores[nonzero] = node_scores[nonzero] / node_counts[nonzero]
        return node_scores


class OriginalFilter(_BaseEdgeFilter):
    def __init__(self, threshold):
        super().__init__(threshold=threshold)
        self.mode = "original"

    def compute_edge_statistics(self, graph):
        edge_index = graph.edge_index
        if edge_index.numel() == 0:
            empty = torch.empty(0, device=graph.x.device, dtype=graph.x.dtype)
            return empty, empty

        edge_similarity = F.cosine_similarity(
            graph.x[edge_index[0]],
            graph.x[edge_index[1]],
            dim=1,
        )
        edge_suspicious_score = 1.0 - edge_similarity
        return edge_similarity, edge_suspicious_score


class NeighborSimilarityFilter(_BaseEdgeFilter):
    def __init__(self, threshold, w1=0.5, w2=0.5):
        super().__init__(threshold=threshold)
        self.mode = "neighbor_similarity"
        self.w1 = w1
        self.w2 = w2

    def compute_edge_statistics(self, graph):
        edge_index = graph.edge_index
        if edge_index.numel() == 0:
            empty = torch.empty(0, device=graph.x.device, dtype=graph.x.dtype)
            return empty, empty

        h1 = self._aggregate_neighbors(graph.num_nodes, edge_index, graph.x)
        h2 = self._aggregate_neighbors(graph.num_nodes, edge_index, h1)

        sim_1 = F.cosine_similarity(h1[edge_index[0]], h1[edge_index[1]], dim=1, eps=1e-12)
        sim_2 = F.cosine_similarity(h2[edge_index[0]], h2[edge_index[1]], dim=1, eps=1e-12)
        edge_similarity = self.w1 * sim_1 + self.w2 * sim_2
        edge_suspicious_score = 1.0 - edge_similarity
        return edge_similarity, edge_suspicious_score

    def _aggregate_neighbors(self, num_nodes, edge_index, features):
        aggregated = torch.zeros_like(features)
        aggregated.index_add_(0, edge_index[1], features[edge_index[0]])
        return aggregated


class HybridFilter(_BaseEdgeFilter):
    def __init__(self, threshold, w1=0.5, w2=0.5, alpha=0.5):
        super().__init__(threshold=threshold)
        self.mode = "hybrid"
        self.original_filter = OriginalFilter(threshold=threshold)
        self.neighbor_filter = NeighborSimilarityFilter(threshold=threshold, w1=w1, w2=w2)
        self.alpha = alpha

    def compute_edge_statistics(self, graph):
        original_similarity, original_edge_score = self.original_filter.compute_edge_statistics(graph)
        neighbor_similarity, neighbor_edge_score = self.neighbor_filter.compute_edge_statistics(graph)

        edge_suspicious_score = self.alpha * original_edge_score + (1.0 - self.alpha) * neighbor_edge_score
        edge_similarity = 1.0 - edge_suspicious_score
        return edge_similarity, edge_suspicious_score
