"""Session memory: retries after a block, and edits built up across turns.

Images are keyed by a perceptual hash, so re-encoding or re-uploading the same
record still matches. Only hashes and EditSpecs are kept, never the image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .schema import Action, EditSpec

HASH_SIZE = 16
MATCH_DISTANCE = 12  # of 256 bits


def image_key(image: bytes) -> str:
    """Average hash (aHash), hex encoded."""
    img = Image.open(io.BytesIO(image)).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.Resampling.LANCZOS)
    px = np.asarray(img, dtype=float).ravel()
    bits = "".join("1" if p > px.mean() else "0" for p in px)
    return f"{int(bits, 2):0{HASH_SIZE * HASH_SIZE // 4}x}"


@dataclass
class _Entry:
    key: str
    edit: EditSpec
    action: Action


@dataclass
class SessionMemory:
    entries: list[_Entry] = field(default_factory=list)

    @staticmethod
    def same_image(a: str, b: str) -> bool:
        return bin(int(a, 16) ^ int(b, 16)).count("1") <= MATCH_DISTANCE

    def _for(self, key: str) -> list[_Entry]:
        return [e for e in self.entries if self.same_image(e.key, key)]

    def record(self, key: str, edit: EditSpec, action: Action) -> None:
        self.entries.append(_Entry(key, edit, action))

    def turns(self, key: str) -> int:
        """How many earlier edits on this image in this session."""
        return len(self._for(key))

    def halted_before(self, key: str) -> bool:
        """Was an edit on this image already blocked or sent to review in this session?"""
        return any(e.action != Action.ALLOW for e in self._for(key))

    def cumulative(self, key: str, edit: EditSpec) -> EditSpec:
        """This edit plus every earlier allowed edit on the same original image."""
        prior = [e.edit for e in self._for(key) if e.action == Action.ALLOW]
        if not prior:
            return edit
        changes = [c for p in prior for c in p.field_changes] + list(edit.field_changes)
        visuals = [v for p in prior for v in p.visual_changes] + list(edit.visual_changes)
        return edit.model_copy(update={
            "field_changes": changes,
            "visual_changes": visuals,
            "real_person_false_act": edit.real_person_false_act or any(p.real_person_false_act for p in prior),
        })
