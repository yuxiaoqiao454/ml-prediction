# registry for clustering methods
CLUSTERING_REGISTRY = {}

def register_clustering(name):
    """Decorator to register a clustering function."""
    def decorator(fn):
        CLUSTERING_REGISTRY[name] = fn
        return fn
    return decorator
