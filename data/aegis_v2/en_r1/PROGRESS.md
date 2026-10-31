# Round 1 (English, targeted)

- Specs: `spec.sample(280, 5007, lang="en", focus=...)`, 465 samples (BLOCK 198 / ALLOW 185 / REVIEW 82).
- Focus: grey_share 0.38 (REVIEW was never predicted by the pilot), remove_marking 0.3 of harmful plans,
  story boost translation_cover x4, error_correction / ownership / restoration x2.
- Briefs: `python -m reframebench.datagen_v2.briefs data/aegis_v2/en_r1`
- Check: `python -m reframebench.datagen_v2.handwritten data/aegis_v2/en_r1`
- Done: batch_01-06 (through g5007-000118). Next: g5007-000119.
