import hashlib
import sys

from scripts import verify_model_snapshot


def test_verifier_rejects_partial_snapshot(tmp_path, monkeypatch, capsys) -> None:
    data = b"complete"
    expected = {
        "model.bin": {
            "Path": "model.bin",
            "Type": "blob",
            "Size": len(data),
            "Sha256": hashlib.sha256(data).hexdigest(),
        }
    }
    monkeypatch.setattr(verify_model_snapshot, "_metadata", lambda *_: expected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", str(tmp_path), "--model-id", "org/model"],
    )

    (tmp_path / "model.bin").write_bytes(data[:-1])

    assert verify_model_snapshot.main() == 1
    assert "expected 8, got 7" in capsys.readouterr().err


def test_verifier_accepts_matching_snapshot(tmp_path, monkeypatch) -> None:
    data = b"complete"
    expected = {
        "model.bin": {
            "Path": "model.bin",
            "Type": "blob",
            "Size": len(data),
            "Sha256": hashlib.sha256(data).hexdigest(),
        }
    }
    monkeypatch.setattr(verify_model_snapshot, "_metadata", lambda *_: expected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", str(tmp_path), "--model-id", "org/model"],
    )
    (tmp_path / "model.bin").write_bytes(data)

    assert verify_model_snapshot.main() == 0
