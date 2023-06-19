from xmodaler.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone
"""

def build_backbone(cfg, name):
    embeddings = BACKBONE_REGISTRY.get(name)(cfg)
    return embeddings