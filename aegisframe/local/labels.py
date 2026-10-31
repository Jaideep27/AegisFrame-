"""v0.7 label vocabulary, shared by the data engine (reframebench/datagen_v2) and the runtime.

Definitions of every label are in docs/labeling-guideline.md. Categories describe what kind of
fact changes and who would rely on the result, never a country, a document brand or a language.
"""

from __future__ import annotations

DECISION = ["ALLOW", "REVIEW", "BLOCK"]

INTENT = [
    # harmful when the output could pass as genuine
    "alter_fact",          # change a fact on an existing record
    "remove_marking",      # erase SAMPLE / VOID / COPY / watermark / cancellation stamp
    "forge_auth",          # add or copy a signature, seal, stamp, QR or hologram
    "fabricate_record",    # make a realistic record from text alone, with specific facts
    # benign or grey
    "own_correction",      # claims a fact on their own record is wrong (grey)
    "marked_sample",       # output carries an enforced SAMPLE / SPECIMEN marking
    "fictional_template",  # placeholders or obviously fictional entities
    "redact",              # hide / blur / black out information
    "translate",           # translate the record's text, facts unchanged
    "enhance",             # restore, sharpen, straighten, crop, recolour; no fact change
    "extract",             # read, summarise, explain, answer a question; no image output
    "annotate",            # highlight, circle, add notes or arrows clearly separate from the record
    "accessibility",       # larger text, high contrast, reformat for screen readers
    "photo_edit",          # ordinary photo edit with no record involved
]
HARMFUL_INTENTS = frozenset(INTENT[:4])

FACT_CLASS = ["none", "monetary", "identity", "credential_result", "validity_date", "authorisation",
              "provenance_marking", "quantity_other"]

FRAUD_PATTERN = ["none", "income_inflation", "balance_inflation", "payment_proof", "amount_deflation",
                 "expense_padding", "credential_inflation", "result_flip", "identity_swap", "validity_extension",
                 "date_shift", "authorisation_forgery", "marking_removal", "record_fabrication"]

RELIANCE_PARTY = ["none", "lender_landlord", "employer", "merchant_counterparty", "education", "government",
                  "insurer", "platform_public", "court_police", "healthcare"]

LEGITIMATE_NEED = ["none", "privacy", "accessibility", "translation", "information", "presentation",
                   "training_sample", "design_template", "personal_photo", "own_record_fix"]

# the reason the user gives (the story), whatever their real intent
PURPOSE = ["none", "research_dataset", "model_testing", "education_demo", "legal_dispute", "error_correction",
           "ownership", "authority_role", "urgency", "emotional", "roleplay_fiction", "translation_cover",
           "business", "other", "jailbreak_persona", "prior_compliance", "ai_staff", "issuer_permission",
           "replacement_copy", "restoration", "hypothetical", "minimising", "prank_meme", "template_bait",
           "grandma_roleplay"]

DISTINGUISHABLE = ["yes", "no", "na"]

# Persuasive Adversarial Prompt taxonomy (Zeng et al. 2024), 40 techniques
PAP = ["evidence_based", "logical_appeal", "expert_endorsement", "non_expert_testimonial", "authority_endorsement",
       "social_proof", "injunctive_norm", "foot_in_the_door", "door_in_the_face", "public_commitment",
       "alliance_building", "complimenting", "shared_values", "relationship_leverage", "loyalty_appeals", "favor",
       "negotiation", "encouragement", "affirmation", "positive_emotion", "negative_emotion", "storytelling",
       "anchoring", "priming", "framing", "confirmation_bias", "reciprocity", "compensation", "supply_scarcity",
       "time_pressure", "reflective_thinking", "threats", "false_promises", "misrepresentation",
       "false_information", "rumors", "social_punishment", "creating_dependency", "exploiting_weakness",
       "discouragement"]
PERSUASION = ["none", *PAP]

TACTIC = ["direct", "long_padding", "negated_marking", "multi_field", "payload_split", "multi_turn",
          "obfuscation", "code_mixed", "euphemism", "indirect_reference", "step_by_step"]

SPAN_TYPES = ["FIELD", "OLD", "NEW", "MARK", "OBJ", "STORY"]
TAGS = ["O"] + [f"{p}-{t}" for t in SPAN_TYPES for p in ("B", "I")]

HEADS = {
    "decision": DECISION,
    "intent": INTENT,
    "fact_class": FACT_CLASS,
    "fraud_pattern": FRAUD_PATTERN,
    "reliance_party": RELIANCE_PARTY,
    "legitimate_need": LEGITIMATE_NEED,
    "purpose": PURPOSE,
    "output_distinguishable": DISTINGUISHABLE,
    "record_from_text": ["no", "yes"],
    "persuasion": PERSUASION,
}
