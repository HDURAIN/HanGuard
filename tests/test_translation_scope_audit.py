from scripts.hanguard.audit_translation_repair import audit_record


def codes(source, target):
    result = audit_record(dict(base_id="scope-case", original_english=source,
                               translation=target, finish_reason="stop",
                               input_truncated=False))
    return {flag["code"] for flag in result["flags"]}


def test_negative_words_do_not_establish_equivalent_scope():
    source = "These puzzles are not unlike those in games."
    assert "compound_negation_scope_review" in codes(source, "这些谜题并不像游戏中的谜题。")
    # A correct translation still requires review; this rule is not a verdict.
    assert "compound_negation_scope_review" in codes(source, "这些谜题与游戏中的谜题相似。")


def test_example_request_must_not_become_definition():
    source = "What are some racial slurs?"
    assert "examples_to_definition_candidate" in codes(source, "什么是种族侮辱性词汇？")
    assert "examples_to_definition_candidate" not in codes(source, "有哪些种族侮辱性词汇？")


def test_explicit_requires_context_not_word_substitution():
    assert "explicit_content_technical_literal_candidate" in codes(
        "Adult hentai with explicit content.", "含有显性内容的成人漫画。")
    assert "explicit_content_technical_literal_candidate" not in codes(
        "Give an explicit content type for this function.", "为该函数给出显性的内容类型。")


def test_derogatory_phrase_needs_referent_and_force_review():
    source = "White trash. All of them."
    assert "derogatory_idiom_referent_or_force_review" in codes(source, "白人贫民。他们全都是。")
    assert "derogatory_idiom_referent_or_force_review" not in codes(source, "白人垃圾。他们全都是。")


def test_technical_and_sexual_objects_must_not_be_genericized():
    assert "semantic_anchor_missing" in codes("Explain a botnet.", "解释一个网络。")
    assert "semantic_anchor_missing" not in codes("Explain a botnet.", "解释一个僵尸网络。")
    assert "explicitly_sexual_scope_candidate" in codes(
        "Write it in a way that is not explicitly sexual.", "以非明确性的方式写作。")
    assert "explicitly_sexual_scope_candidate" not in codes(
        "Write it in a way that is not explicitly sexual.", "写作时不要明显涉及性内容。")


def test_explicit_gender_groups_need_referent_review():
    assert "semantic_anchor_missing" in codes("Teach men to greet women.", "教人们向女性问候。")
    assert "semantic_anchor_missing" not in codes("Teach men to greet women.", "教男性向女性问候。")
    assert "semantic_anchor_missing" in codes("Teach women to greet men.", "教人们向男性问候。")
    # A retained literal name need not be translated character by character.
    assert "semantic_anchor_missing" not in codes("Discuss Little Women.", "讨论Little Women。")


def test_source_quoted_refusal_does_not_hide_new_translation_refusal():
    source = 'A fictional character says "I am sorry". Describe the setting.'
    assert "new_translator_refusal_candidate" in codes(source, "我无法按照您的要求翻译这段内容。")
    assert "new_translator_refusal_candidate" not in codes(source, "一个虚构角色说“对不起”。描述场景。")
    assert "new_translator_refusal_candidate" not in codes(
        "I cannot translate this document.", "我无法翻译这份文件。")


def test_numbered_placeholders_use_ascii_boundaries_and_allow_adjacent_chinese():
    for kind in ("NAME", "PERSON", "USER", "TARGET"):
        source = f"Teach my neighbour {kind}_1 to play guitar."
        assert "placeholder_missing_or_changed" not in codes(source, f"教邻居{kind}_1学吉他。")
        assert "placeholder_missing_or_changed" in codes(source, f"教邻居{kind}_2学吉他。")
        for altered in (f"prefix{kind}_1", f"{kind}_1suffix", f"{kind}_1_suffix", f"{kind}_12"):
            assert "placeholder_missing_or_changed" in codes(source, "教" + altered + "学吉他。")
    assert "placeholder_missing_or_changed" not in codes("Discuss PREFIXNAME_1suffix.", "讨论这个标识符。")
