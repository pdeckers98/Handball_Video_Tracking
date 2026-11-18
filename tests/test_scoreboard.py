import numpy as np

from handball.scoreboard import (
    GLYPH_SHAPE,
    Reading,
    _nearest,
    default_templates,
    normalize_glyph,
    read_frame,
    segment,
)


def test_templates_cover_all_symbols():
    t = default_templates()
    assert set(t.score_labels) == set("0123456789")
    assert set(t.clock_labels) == set("0123456789")
    assert set(t.half_labels) == {"1ST", "2ND"}
    assert {"SEW", "PSV"} <= set(t.name_labels)
    assert t.score.shape[1:] == GLYPH_SHAPE


def test_each_template_matches_its_own_label():
    t = default_templates()
    for kind in ("score", "clock", "name", "half"):
        templates, labels = getattr(t, kind), getattr(t, f"{kind}_labels")
        for template, label in zip(templates, labels):
            assert _nearest(template, templates, labels, 0.01) == label


def test_digit_templates_are_distinct():
    t = default_templates()
    flat = t.score.reshape(len(t.score), -1)
    for i, label_i in enumerate(t.score_labels):
        for j, label_j in enumerate(t.score_labels):
            if label_i != label_j:
                # Closest pairs are 0/6/8/9 at ~0.05; real glyphs match their own at <= 0.065
                # and stay >= 0.027 closer to it than to any other digit.
                assert np.abs(flat[i] - flat[j]).mean() > 0.04


def test_segment_and_normalize():
    mask = np.zeros((30, 40), bool)
    mask[5:25, 3:10] = True  # glyph 1
    mask[5:25, 15:30] = True  # glyph 2
    mask[1:3, 35:37] = True  # speck: dropped
    glyphs = segment(mask)
    assert [g.shape[1] for g in glyphs] == [7, 15]
    norm = normalize_glyph(glyphs[0])
    assert norm.shape == GLYPH_SHAPE
    assert norm.max() == 1.0


def test_frame_without_overlay():
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    assert read_frame(frame) == Reading(present=False)
    assert Reading(present=True, home=3, away=2).score == (3, 2)
