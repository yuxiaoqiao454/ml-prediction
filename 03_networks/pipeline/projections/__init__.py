# registry for projection methods
PROJECTION_REGISTRY = {}

def register_projection(name):
    """Decorator to register a projection function."""
    def decorator(fn):
        PROJECTION_REGISTRY[name] = fn
        return fn
    return decorator


# === Import and register ===
from .cosine import run_cosine_projection
from .count_based import run_count_projection
# from .count_shared import run_count_shared_projection
# from .jaccard import run_jaccard_projection
# engagement_count can be added later

# manual registration (for clarity)
PROJECTION_REGISTRY["cosine"] = run_cosine_projection
PROJECTION_REGISTRY["count"] = run_count_projection
# PROJECTION_REGISTRY["count_shared"] = run_count_shared_projection
# PROJECTION_REGISTRY["jaccard"] = run_jaccard_projection
