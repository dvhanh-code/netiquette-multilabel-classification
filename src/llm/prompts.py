"""
src/llm/prompts.py
------------------
Prompt template system for multilabel netiquette classification.

Design principles:
  - Label definitions are precise enough to distinguish overlapping categories
    (toxic vs insult, hate_speech vs insult, threat vs hate_speech).
  - Platform rules are injected as a named block so the model treats them as
    authoritative overrides, not suggestions.
  - Few-shot examples cover all four quadrants: clean, single-label, multi-label,
    and the hardest-to-distinguish pairs.
  - Chain-of-thought (CoT) mode adds an explicit reasoning instruction before
    the structured output and is evaluated as a separate experimental condition.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LABELS = ["hate_speech", "toxic", "threat", "insult"]


@dataclass
class LabelDefinition:
    name: str
    description: str
    positive_examples: List[str]
    negative_examples: List[str]


@dataclass
class PlatformRules:
    platform_name: str
    additional_guidelines: List[str] = field(default_factory=list)
    # If set, the prompt emphasizes these labels above others.
    emphasis_labels: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Default label definitions
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LABEL_DEFINITIONS: Dict[str, LabelDefinition] = {
    "hate_speech": LabelDefinition(
        name="hate_speech",
        description=(
            "Content that attacks, dehumanizes, or calls for discrimination against a "
            "person or group based on a PROTECTED CHARACTERISTIC: ethnicity, religion, "
            "national origin, gender, sexual orientation, or disability. "
            "Key markers: slurs targeting a group, dehumanizing comparisons, advocacy "
            "for exclusion/violence based solely on identity. "
            "Distinguished from 'insult': hate_speech targets an identity group, "
            "insult targets an individual personally."
        ),
        positive_examples=[
            "Diese Ausländer gehören alle abgeschoben — sie sind keine echten Menschen.",
            "Frauen sollten kein Wahlrecht haben, sie sind dazu nicht fähig.",
        ],
        negative_examples=[
            "Ich stimme der Einwanderungspolitik dieser Regierung nicht zu.",
            "Er ist ein kompletter Idiot.",  # insult, not hate_speech
        ],
    ),
    "toxic": LabelDefinition(
        name="toxic",
        description=(
            "Rude, hostile, or unreasonably aggressive content that degrades the quality "
            "of civil discourse. Includes profanity used aggressively, inflammatory "
            "rhetoric, condescension, and content a reasonable person would find "
            "deeply offensive — even without targeting a protected group or a specific "
            "individual. Toxicity is the broadest category: hate_speech and insult are "
            "always also toxic, but not vice versa."
        ),
        positive_examples=[
            "Das ist der größte Blödsinn, den ich je gehört habe! Hört auf zu reden.",
            "Was für ein totaler Schwachsinn! Ihr wisst alle nichts.",
        ],
        negative_examples=[
            "Das Argument überzeugt mich nicht.",
            "Ich bin anderer Meinung, aber ich verstehe deinen Standpunkt.",
        ],
    ),
    "threat": LabelDefinition(
        name="threat",
        description=(
            "Explicit or strongly implied statements of intent to cause harm, injury, "
            "or damage to a person, group, or their property. Includes direct threats "
            "('ich werde dir etwas antun'), conditional threats ('wenn du X machst, "
            "passiert dir Y'), and intimidation statements that imply physical or serious "
            "social consequences. "
            "NOT a threat: general predictions ('du wirst scheitern'), hyperbolic "
            "expressions of frustration ('ich könnte dich umbringen' said colloquially), "
            "or news reports describing third-party threats."
        ),
        positive_examples=[
            "Wenn du das nochmal schreibst, werde ich dafür sorgen, dass du es bereust.",
            "Pass auf, wo du alleine herumläufst.",
        ],
        negative_examples=[
            "Das wird Konsequenzen haben.",
            "Du wirst schon sehen, was passiert, wenn du weiter lügst.",
        ],
    ),
    "insult": LabelDefinition(
        name="insult",
        description=(
            "Direct personal attacks on a specific individual's intelligence, character, "
            "appearance, competence, or worth. Includes name-calling, mockery, and "
            "derogatory statements aimed at one person. "
            "Distinguished from hate_speech: insult targets the individual, not a group. "
            "Distinguished from toxic: insult is always also toxic, but a comment can "
            "be toxic (broadly offensive) without a specific personal attack."
        ),
        positive_examples=[
            "Du bist doch komplett bescheuert und verstehst gar nichts.",
            "Was für ein inkompetenter Vollidiot bist du eigentlich?",
        ],
        negative_examples=[
            "Dein Argument ist schwach.",
            "Diese Ausländer sind Idioten.",  # insult to a group → hate_speech, not insult
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot examples
# ─────────────────────────────────────────────────────────────────────────────
# 8 examples covering: clean, single-label (each), and multi-label cases.

_FEW_SHOT_EXAMPLES = [
    {
        "text": "Das finde ich zwar interessant, aber ich stimme nicht zu.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.02,
            "toxic": 0,       "toxic_confidence": 0.03,
            "threat": 0,      "threat_confidence": 0.01,
            "insult": 0,      "insult_confidence": 0.02,
            "reasoning": "Polite disagreement, no offensive content whatsoever.",
        },
    },
    {
        "text": "Diese Ausländer nehmen uns alle Arbeitsplätze weg — sie gehören abgeschoben!",
        "output": {
            "hate_speech": 1, "hate_speech_confidence": 0.96,
            "toxic": 1,       "toxic_confidence": 0.89,
            "threat": 0,      "threat_confidence": 0.04,
            "insult": 0,      "insult_confidence": 0.08,
            "reasoning": (
                "Attacks immigrants as a group based on national origin (hate_speech). "
                "The hostile call for deportation is also toxic."
            ),
        },
    },
    {
        "text": "Du bist doch komplett bescheuert! Komm mal auf den Boden der Tatsachen.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.03,
            "toxic": 1,       "toxic_confidence": 0.92,
            "threat": 0,      "threat_confidence": 0.01,
            "insult": 1,      "insult_confidence": 0.90,
            "reasoning": (
                "Direct personal attack on the individual's intelligence (insult). "
                "The aggressive tone also qualifies as toxic."
            ),
        },
    },
    {
        "text": "Wenn du nochmal so einen Unsinn schreibst, wirst du es bereuen.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.03,
            "toxic": 1,       "toxic_confidence": 0.84,
            "threat": 1,      "threat_confidence": 0.88,
            "insult": 0,      "insult_confidence": 0.11,
            "reasoning": (
                "Conditional threat implying unspecified negative consequences (threat). "
                "Hostile tone qualifies as toxic."
            ),
        },
    },
    {
        "text": "Interessante Perspektive! Ich habe andere Erfahrungen gemacht, aber das ist eine spannende Diskussion.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.01,
            "toxic": 0,       "toxic_confidence": 0.01,
            "threat": 0,      "threat_confidence": 0.01,
            "insult": 0,      "insult_confidence": 0.01,
            "reasoning": "Positive, constructive engagement.",
        },
    },
    {
        "text": "Was für ein kompletter Vollidiot! Leute wie du sollten sich schämen.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.06,
            "toxic": 1,       "toxic_confidence": 0.96,
            "threat": 0,      "threat_confidence": 0.02,
            "insult": 1,      "insult_confidence": 0.95,
            "reasoning": "Strong personal insult targeting the individual's competence and character.",
        },
    },
    {
        "text": "Pass auf, wo du nachts alleine bist. Leute wie du verschwinden manchmal einfach.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.12,
            "toxic": 1,       "toxic_confidence": 0.91,
            "threat": 1,      "threat_confidence": 0.94,
            "insult": 0,      "insult_confidence": 0.10,
            "reasoning": (
                "Strongly implied physical threat through intimidating language about "
                "the person's safety (threat + toxic)."
            ),
        },
    },
    {
        "text": "Die Politik dieser Partei ist völlig falsch und schadet dem Land.",
        "output": {
            "hate_speech": 0, "hate_speech_confidence": 0.04,
            "toxic": 0,       "toxic_confidence": 0.07,
            "threat": 0,      "threat_confidence": 0.01,
            "insult": 0,      "insult_confidence": 0.05,
            "reasoning": "Critical political opinion expressed without offensive language.",
        },
    },
]

_JSON_SCHEMA = """\
{
  "hate_speech": <integer: 0 or 1>,
  "hate_speech_confidence": <float in [0.0, 1.0]>,
  "toxic": <integer: 0 or 1>,
  "toxic_confidence": <float in [0.0, 1.0]>,
  "threat": <integer: 0 or 1>,
  "threat_confidence": <float in [0.0, 1.0]>,
  "insult": <integer: 0 or 1>,
  "insult_confidence": <float in [0.0, 1.0]>,
  "reasoning": "<string: brief explanation, max 2 sentences>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds system and user prompts for multilabel netiquette classification.

    Args:
        label_definitions:
            Override the default label definitions. Useful for ablations.
        use_cot:
            If True, instructs the model to briefly analyze before outputting JSON.
        use_few_shot:
            Include few-shot examples in the system prompt.
        include_reasoning:
            Include a "reasoning" field in the required JSON output.
    """

    def __init__(
        self,
        label_definitions: Optional[Dict[str, LabelDefinition]] = None,
        use_cot: bool = False,
        use_few_shot: bool = True,
        include_reasoning: bool = True,
    ):
        self.label_defs = label_definitions or DEFAULT_LABEL_DEFINITIONS
        self.use_cot = use_cot
        self.use_few_shot = use_few_shot
        self.include_reasoning = include_reasoning

    def build_system_prompt(
        self,
        platform_rules: Optional[PlatformRules] = None,
    ) -> str:
        parts: List[str] = []

        parts += [
            "You are an expert content moderator specializing in German-language online comments.",
            "Your task: classify a given comment across four harm categories (multilabel — a comment can belong to multiple).",
            "",
        ]

        # Label definitions
        parts.append("## Label Definitions\n")
        for label in LABELS:
            defn = self.label_defs[label]
            parts.append(f"### {label.upper()}")
            parts.append(defn.description)
            parts.append("")

        # Platform-specific overrides
        if platform_rules:
            parts.append("## Platform-Specific Guidelines")
            parts.append(f"**Platform:** {platform_rules.platform_name}")
            for g in platform_rules.additional_guidelines:
                parts.append(f"- {g}")
            if platform_rules.emphasis_labels:
                labels_str = ", ".join(platform_rules.emphasis_labels)
                parts.append(
                    f"- Pay special attention to: {labels_str}"
                )
            parts.append("")

        # Universal rules
        parts += [
            "## Classification Rules",
            "- Each label is an independent binary decision (0=absent, 1=present).",
            "- A comment may have 0, 1, or multiple labels simultaneously.",
            "- `confidence` reflects certainty: 0.5 = maximum uncertainty, 0.95+ = very certain.",
            "- Satire or irony that QUOTES harmful speech to CRITIQUE it is generally NOT classified as hate_speech.",
            "- News reports describing third-party threats are NOT classified as threat.",
            "- Assess only the text as written; do not infer unwritten context.",
            "- `toxic` is the broadest category: hate_speech and insult are always also toxic.",
            "",
        ]

        if self.use_cot:
            parts += [
                "## Reasoning Process",
                "Before writing the JSON, briefly reason through each label in 1-2 sentences.",
                "Then output the JSON object.",
                "",
            ]

        # Output schema
        schema = _JSON_SCHEMA
        if not self.include_reasoning:
            schema = schema.replace(
                '  "reasoning": "<string: brief explanation, max 2 sentences>"\n', ""
            )

        parts += [
            "## Output Format",
            "Return ONLY a valid JSON object. No markdown fences, no extra text.",
            schema,
            "",
        ]

        # Few-shot examples
        if self.use_few_shot:
            parts.append("## Examples\n")
            for i, ex in enumerate(_FEW_SHOT_EXAMPLES, 1):
                output = dict(ex["output"])
                if not self.include_reasoning:
                    output.pop("reasoning", None)
                parts.append(f"Example {i}:")
                parts.append(f'Input: "{ex["text"]}"')
                parts.append(f"Output: {json.dumps(output, ensure_ascii=False)}")
                parts.append("")

        return "\n".join(parts)

    def build_user_prompt(self, text: str) -> str:
        return f'Classify the following German comment:\n\nInput: "{text}"\nOutput:'

    @property
    def expected_output_keys(self) -> List[str]:
        keys = []
        for label in LABELS:
            keys.append(label)
            keys.append(f"{label}_confidence")
        if self.include_reasoning:
            keys.append("reasoning")
        return keys