import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_prefix_requests.py"
SPEC = importlib.util.spec_from_file_location("audit_prefix_requests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_common_prefix_length() -> None:
    assert MODULE.common_prefix_length([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert MODULE.common_prefix_length([]) == 0


def test_volatile_request_fields() -> None:
    findings = MODULE.find_volatile_fields(
        {
            "request_id": "abc",
            "message": "generated 2026-09-04T12:30:00",
            "nested": {"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479"},
        }
    )
    assert "$.request_id: volatile key" in findings
    assert "$.message: timestamp/UUID-like value" in findings
    assert "$.nested.uuid: volatile key" in findings


def test_canonical_hash_ignores_object_key_order() -> None:
    assert MODULE.stable_hash({"a": 1, "b": 2}) == MODULE.stable_hash({"b": 2, "a": 1})
