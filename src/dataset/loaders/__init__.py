from .gmhp7k import GMHP7kLoader
from .hocon34k import HOCON34kLoader
from .detox import DetoxLoader
from .jigsaw import JigsawLoader
from .wikipedia_attacks import WikipediaAttacksLoader
from .wikipedia_politeness import WikipediaPolitenessLoader
from .gutefrage import GutefragLoader

# Registry — ordered dict determines load order in UnifiedCorpusDataset
ALL_LOADERS = {
    "gmhp7k":               GMHP7kLoader,
    "hocon34k":             HOCON34kLoader,
    "detox":                DetoxLoader,
    "jigsaw":               JigsawLoader,
    "wikipedia_attacks":    WikipediaAttacksLoader,
    "wikipedia_politeness": WikipediaPolitenessLoader,
    "gutefrage":            GutefragLoader,
}

__all__ = [
    "GMHP7kLoader",
    "HOCON34kLoader",
    "DetoxLoader",
    "JigsawLoader",
    "WikipediaAttacksLoader",
    "WikipediaPolitenessLoader",
    "GutefragLoader",
    "ALL_LOADERS",
]
