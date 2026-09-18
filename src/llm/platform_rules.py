"""
src/llm/platform_rules.py
--------------------------
Platform-specific rule injection for the prompt template system.

Each platform preset customizes which behaviors are tolerated or emphasized,
allowing the same trained LLM to act as a configurable content moderator.
This addresses RQ4: flexible adaptation to platform-specific netiquette.

Usage:
    rules = get_platform_rules("news_comments")
    prompt = builder.build_system_prompt(platform_rules=rules)

    # Or build custom rules:
    rules = PlatformRules(
        platform_name="My Forum",
        additional_guidelines=["No profanity, even mild."],
        emphasis_labels=["toxic"],
    )
"""

from typing import Dict, List, Optional
from src.llm.prompts import PlatformRules

# ─────────────────────────────────────────────────────────────────────────────
# Platform presets
# ─────────────────────────────────────────────────────────────────────────────

_PRESETS: Dict[str, PlatformRules] = {
    # Strict: zero tolerance. Typical for children's platforms or regulated media.
    "strict": PlatformRules(
        platform_name="Strict Community Forum",
        additional_guidelines=[
            "Any profanity, even mild, should be classified as toxic.",
            "Sarcastic or ironic insults are still insults — intent does not exempt.",
            "Borderline content should be classified as present (err on the side of safety).",
            "Even indirect or implied threats should be classified as threat.",
        ],
        emphasis_labels=["toxic", "insult"],
    ),

    # Lenient: allows debate culture, strong opinions, hyperbole.
    "debate_forum": PlatformRules(
        platform_name="Political Debate Forum",
        additional_guidelines=[
            "Strong political criticism and sharp rhetoric are acceptable.",
            "Only classify as toxic if language is personally attacking or gratuitously offensive.",
            "Hyperbolic expressions of frustration ('das ist eine Katastrophe') are NOT toxic.",
            "Policy criticism targeting political groups is NOT hate_speech unless it targets a protected characteristic.",
            "Require explicit evidence of threat intent — conditional statements about political consequences are NOT threat.",
        ],
        emphasis_labels=["hate_speech", "threat"],
    ),

    # News comment sections: semi-formal, political topics, moderate tolerance.
    "news_comments": PlatformRules(
        platform_name="News Comment Section",
        additional_guidelines=[
            "Sharp political opinions and critical commentary are permitted.",
            "Classify as toxic if language is gratuitously offensive or derails civil discourse.",
            "Discriminatory statements about political or ethnic groups should be classified as hate_speech.",
            "Focus on content that would embarrass a reputable news outlet.",
        ],
        emphasis_labels=["hate_speech", "toxic"],
    ),

    # Social media: short-form, high volume, high base rate of toxicity.
    "social_media": PlatformRules(
        platform_name="Social Media Platform",
        additional_guidelines=[
            "Short-form content: even brief insults count.",
            "Coded language and dogwhistles that imply group-based hatred should be classified as hate_speech.",
            "Implied threats ('watch your back') should be classified as threat.",
            "Consider context of harassment campaigns: repeated mild insults escalate.",
        ],
        emphasis_labels=["hate_speech", "insult", "threat"],
    ),

    # Academic / research context: maximize recall for annotation purposes.
    "annotation": PlatformRules(
        platform_name="Research Annotation",
        additional_guidelines=[
            "Err toward including borderline cases (maximize recall).",
            "When unsure, classify as present with low confidence (0.5-0.6).",
            "Provide detailed reasoning for all non-trivial cases.",
        ],
        emphasis_labels=None,
    ),
}


def get_platform_rules(preset_name: str) -> PlatformRules:
    """
    Return a pre-built PlatformRules for a named platform preset.

    Available presets:
        strict, debate_forum, news_comments, social_media, annotation
    """
    if preset_name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(
            f"Unknown platform preset {preset_name!r}. Available: {available}"
        )
    return _PRESETS[preset_name]


def list_presets() -> List[str]:
    return sorted(_PRESETS.keys())


def custom_platform_rules(
    platform_name: str,
    guidelines: List[str],
    emphasis_labels: Optional[List[str]] = None,
) -> PlatformRules:
    """Build a one-off PlatformRules from a list of guideline strings."""
    return PlatformRules(
        platform_name=platform_name,
        additional_guidelines=guidelines,
        emphasis_labels=emphasis_labels,
    )