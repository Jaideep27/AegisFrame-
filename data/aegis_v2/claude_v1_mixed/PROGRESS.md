# Hand-written samples (teacher: claude-opus-5-5)

- Specs: `specs.jsonl` = `spec.sample(280, 5005)`, 280 groups, 509 samples. Regenerate briefs with the
  scratchpad script `mkbriefs.py` (same seed) if needed.
- Batches: `batch_NN.json`, compact format (see `reframebench/datagen_v2/handwritten.py`).
- Check: `python -m reframebench.datagen_v2.handwritten data/aegis_v2/claude` writes records.jsonl and rejects.jsonl.
- Done: all 15 batches, 280 groups, 510 records, 0 rejects. dataset.jsonl written (train 337 / val 63 / test_id 30 / test_ood 80).
- Generator bugs fixed while writing: "(amended)" fallback mutations, ID mutate crash, twins targeting the
  marking or signature, multi-year "misprint" corrections, "ss" typo no-op, flight return before departure,
  same-city route, Chinese word count, police-report loss with no number, currency prefix (ZAR→R), month names from the wrong language, forge_auth naming a field label ("Officer"), refusal regex rejecting "I won't".
- Not checked by a second annotator yet; the human audit should cover these too.
