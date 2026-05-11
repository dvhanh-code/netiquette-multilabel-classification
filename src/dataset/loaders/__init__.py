from .gmhp7k import GMHP7kLoader
from .hocon34k import HOCON34kLoader
from .detox import DetoxLoader
from .jigsaw import JigsawLoader
from .wikipedia_attacks import WikipediaAttacksLoader
from .gutefrage import GutefragLoader
from .rp_mod import RPModLoader

# WikipediaPolitenessLoader is kept as an archived optional loader but is
# intentionally excluded from ALL_LOADERS — it targets a politeness/spectrum
# framework that is outside the 4-label harmful-language schema.
from .wikipedia_politeness import WikipediaPolitenessLoader  # noqa: F401 (archived)

# Registry — ordered dict determines load order in UnifiedCorpusDataset
ALL_LOADERS = {
    "gmhp7k":            GMHP7kLoader,
    "hocon34k":          HOCON34kLoader,
    "detox":             DetoxLoader,
    "jigsaw":            JigsawLoader,
    "wikipedia_attacks": WikipediaAttacksLoader,
    "gutefrage":         GutefragLoader,
    "rp_mod":            RPModLoader,
}

__all__ = [
    "GMHP7kLoader",
    "HOCON34kLoader",
    "DetoxLoader",
    "JigsawLoader",
    "WikipediaAttacksLoader",
    "GutefragLoader",
    "RPModLoader",
    "ALL_LOADERS",
    # Archived optional loader — not part of the main pipeline:
    "WikipediaPolitenessLoader",
]
