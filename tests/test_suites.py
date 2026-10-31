from aegisframe.schema import Action
from reframebench.run import AegisSystem, KeywordBaseline, run_suite
from reframebench.suites import BENIGN_DEV, PARAPHRASE_DEV


def test_suites_are_labelled():
    assert len(PARAPHRASE_DEV) >= 15 and all(c.expected == "halt" for c in PARAPHRASE_DEV)
    assert len(BENIGN_DEV) >= 10 and all(c.expected == "allow" for c in BENIGN_DEV)


def test_run_suite_reports_rates():
    stats = run_suite(KeywordBaseline(), PARAPHRASE_DEV)
    assert stats["n"] == len(PARAPHRASE_DEV)
    assert stats["allow"] + stats["review"] + stats["block"] == 1.0


def test_aegis_halts_every_dev_paraphrase_and_blocks_no_benign_edit():
    assert run_suite(AegisSystem(), PARAPHRASE_DEV)["allow"] == 0.0
    assert run_suite(AegisSystem(), BENIGN_DEV)["block"] == 0.0
