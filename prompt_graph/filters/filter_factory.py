from .neighbor_similarity_filter import HybridFilter, NeighborSimilarityFilter, OriginalFilter


def build_filter(args):
    filter_mode = getattr(args, "filter_mode", "original")
    threshold = getattr(args, "pt_threshold", 0.0)
    sim1_weight = getattr(args, "filter_sim1_weight", 0.5)
    sim2_weight = getattr(args, "filter_sim2_weight", 0.5)
    hybrid_alpha = getattr(args, "filter_hybrid_alpha", 0.5)

    if filter_mode == "original":
        return OriginalFilter(threshold=threshold)
    if filter_mode == "neighbor_similarity":
        return NeighborSimilarityFilter(threshold=threshold, w1=sim1_weight, w2=sim2_weight)
    if filter_mode == "hybrid":
        return HybridFilter(
            threshold=threshold,
            w1=sim1_weight,
            w2=sim2_weight,
            alpha=hybrid_alpha,
        )
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")
