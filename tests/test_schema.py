from aegisframe.schema import Action, EditSpec, FieldChange, Marking, Severity, strictest


def test_actions_are_ordered_by_strictness():
    assert Action.ALLOW < Action.REVIEW < Action.BLOCK
    assert strictest(Action.ALLOW, Action.BLOCK, Action.REVIEW) == Action.BLOCK
    assert strictest() == Action.ALLOW


def test_severity_is_ordered():
    assert Severity.NONE < Severity.LOW < Severity.MODERATE < Severity.HIGH < Severity.CRITICAL


def test_replace_counts_as_factual_change():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="Amount", old="500", new="5000")])
    assert spec.factual_changes == spec.field_changes


def test_visible_redaction_is_not_a_factual_change():
    spec = EditSpec(field_changes=[FieldChange(kind="redact", field="Phone", old="98765 43210")])
    assert spec.factual_changes == []


def test_unmarked_factual_change_passes_as_genuine():
    spec = EditSpec(field_changes=[FieldChange(kind="replace", field="Amount", old="500", new="5000")])
    assert spec.passes_as_genuine is True


def test_marking_over_changed_fields_stops_passing_as_genuine():
    spec = EditSpec(
        field_changes=[FieldChange(kind="replace", field="Amount", old="500", new="5000")],
        marking=Marking(kind="watermark", text="SAMPLE", covers_changed_fields=True),
    )
    assert spec.passes_as_genuine is False


def test_marking_elsewhere_still_passes_as_genuine():
    # a tiny corner label does not stop the changed amount being shown as real
    spec = EditSpec(
        field_changes=[FieldChange(kind="replace", field="Amount", old="500", new="5000")],
        marking=Marking(kind="label", text="sample", covers_changed_fields=False),
    )
    assert spec.passes_as_genuine is True


def test_edit_spec_has_no_free_text_purpose_field():
    # the separator's output schema must leave nowhere for a justification to live
    forbidden = {"purpose", "reason", "justification", "context", "intent", "notes"}
    assert forbidden.isdisjoint(EditSpec.model_fields)
    assert forbidden.isdisjoint(FieldChange.model_fields)
