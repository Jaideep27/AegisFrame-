"""Sample specifications: everything about a sample except its wording, decided by code.

A spec fixes the record, the edit plan, the story, the persuasion technique, tactics, language,
length and persona. Labels follow from the spec through guideline.decide(); the teacher LLM only
writes the user's words. Twins share everything except the edit.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import doclib, edits, persuasion as P, regions as R
from .guideline import decide

# held out from training entirely: these persuasion techniques appear only in the OOD test split
OOD_TECHNIQUES = frozenset({"door_in_the_face", "reciprocity", "rumors", "creating_dependency", "anchoring"})

HARMFUL_MIX = {"alter_fact": 0.6, "fabricate_record": 0.2, "remove_marking": 0.1, "forge_auth": 0.1}


@dataclass
class Spec:
    id: str
    group: str                  # twin group: samples sharing a record and story
    role: str                   # harmful | benign_twin | grey
    family: str
    category: str
    locale: str
    region: str
    lang: str
    story: str
    technique: str
    tactics: list[str]
    length: str
    persona: str
    style: str
    deceptive_purpose: bool
    plan: dict
    context_layout: dict | None  # doclib.Doc layout as dict, or None for no image
    decision: str
    reliance_party: str
    ood: list[str] = field(default_factory=list)

    @property
    def edit(self) -> edits.EditPlan:
        p = dict(self.plan)
        p["changes"] = [edits.Change(**c) for c in p["changes"]]
        return edits.EditPlan(**p)


def _layout_dict(doc: doclib.Doc, plan: edits.EditPlan) -> dict | None:
    if not plan.has_image:
        return None
    lay = doc.layout()
    lay.fields = [(k, plan.context_overrides.get(k, v)) for k, v in lay.fields]
    if plan.context_marking:
        lay.text.append(f"{plan.context_marking}")
    d = asdict(lay)
    return d


def _labels(plan: edits.EditPlan, doc: doclib.Doc, deceptive: bool, rng: random.Random) -> tuple[str, str]:
    decision = decide(plan.intent, plan.distinguishable, plan.real_record, deceptive)
    harmful_like = plan.intent in ("alter_fact", "remove_marking", "forge_auth", "fabricate_record",
                                   "own_correction")
    reliance = rng.choice(doc.reliance) if harmful_like or deceptive else "none"
    return decision, reliance


def make_group(rng: random.Random, gid: str, family: str | None = None, locale: str | None = None,
               ood_family: bool = False, lang: str | None = None) -> list[Spec]:
    """One record, one story, and either a harmful/benign twin pair or a single grey request."""
    family = family or rng.choice(doclib.ood_families() if ood_family else doclib.train_families())
    locale = locale or rng.choice(list(R.LOCALES))
    loc = R.LOCALES[locale]
    doc = doclib.make(family, locale, rng)
    lang = lang or R.pick_lang(rng, loc)
    held_out_tech = rng.random() < 0.06
    story = P.pick_story(rng, True)
    technique = rng.choice(sorted(OOD_TECHNIQUES)) if held_out_tech else \
        P.pick_technique(rng, story, OOD_TECHNIQUES)
    tactics = P.pick_tactics(rng, lang)
    length = P.pick_length(rng, tactics)
    persona, style = rng.choice(P.PERSONA), rng.choice(P.STYLE)
    ood = (["family"] if doclib.FAMILIES[family].ood else []) + (["technique"] if held_out_tech else [])

    def spec(role: str, plan: edits.EditPlan, deceptive: bool, sid: str) -> Spec:
        decision, reliance = _labels(plan, doc, deceptive, rng)
        tac = [t for t in tactics if not (t == "negated_marking" and plan.marking_mode != "negate")]
        tac = tac or ["direct"]
        return Spec(sid, gid, role, family, doc.category, locale, loc.region, lang, story, technique, tac, length,
                    persona, style, deceptive, asdict(plan), _layout_dict(doc, plan), decision, reliance, ood)

    if rng.random() < 0.18:  # a group holds one grey request vs two twins: ~10% of samples
        plan = edits.grey(doc, rng)
        return [spec("grey", plan, rng.random() < 0.15, f"{gid}-g")]

    intent = rng.choices(list(HARMFUL_MIX), list(HARMFUL_MIX.values()))[0]
    n_fields = rng.randint(2, 4) if "multi_field" in tactics else 1
    negate = "negated_marking" in tactics
    harm = edits.plan_for(doc, rng, intent, n_fields, negate)
    # a minority of harmful stories say outright who will be deceived; the twin keeps a benign version of it
    deceptive = rng.random() < 0.12
    twin = edits.benign_twin(doc, harm, rng)
    return [spec("harmful", harm, deceptive, f"{gid}-h"), spec("benign_twin", twin, False, f"{gid}-b")]


def sample(n_groups: int, seed: int, ood_share: float = 0.1, lang: str | None = None) -> list[Spec]:
    """lang fixes the message language for every group (e.g. "en"); records stay worldwide."""
    rng = random.Random(seed)
    out: list[Spec] = []
    for i in range(n_groups):
        out += make_group(rng, f"g{seed}-{i:06d}", ood_family=rng.random() < ood_share, lang=lang)
    return out
