import json

import pytest

from scripts.hanguard.repair_wildguard_translation import read_jsonl_records


def test_source_and_translation_with_unicode_separators_roundtrip(tmp_path):
    records = [dict(original_english="first\u2028second\u2029third\vlast",
                    translation="第一段\u2028第二段\u2029第三段\v结束"),
               dict(original_english="next", translation="下一条")]
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    assert list(read_jsonl_records(path)) == records


def test_incomplete_last_record_is_not_silently_accepted(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"translation":"完整"}\n{"translation":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl_records(path))
