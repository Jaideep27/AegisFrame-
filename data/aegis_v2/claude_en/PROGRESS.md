# English build (teacher: claude-opus-5-5)

- Specs: `specs.jsonl` = the 138 English groups from `../claude_v1_mixed` (seed 5005) + `spec.sample(140, 5006, lang="en")`.
  Do NOT regenerate: the story table grew after 5005 was sampled, so re-sampling 5005 gives different specs.
- `batch_00_v1_english.json`: the 249 English messages carried over unchanged.
- New stories added for this build (persuasion.STORY + labels.PURPOSE): jailbreak_persona, prior_compliance, ai_staff,
  issuer_permission, replacement_copy, restoration, hypothetical, minimising, prank_meme, template_bait, grandma_roleplay.
- Check: `python -m reframebench.datagen_v2.handwritten data/aegis_v2/claude_en`
- Done: all batches (00, 16-22). 510 English records, 0 rejects; dataset.jsonl written.
