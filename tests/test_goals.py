from pathlib import Path

from handball.goals import Sample, clock_consistency, detect_goals, stable_states
from handball.scoreboard import Reading


def _samples(scores, start_clock=0):
    """Keyframe samples every 4 s; ``None`` = overlay hidden, ``"?"`` = unreadable score."""
    out = []
    for i, score in enumerate(scores):
        t = 4.0 * i
        if score is None:
            reading = Reading(present=False)
        elif score == "?":
            reading = Reading(present=True, home_name="SEW", away_name="PSV", half="1ST")
        else:
            reading = Reading(
                present=True, home_name="SEW", away_name="PSV", home=score[0], away=score[1],
                half="1ST", clock=start_clock + 4 * i,
            )
        out.append(Sample(t=t, index=int(t * 50), reading=reading))
    return out


def _detect(scores):
    return detect_goals(Path("clip.mp4"), _samples(scores), refine=False)


def test_stable_states_ignore_single_misreads():
    states = stable_states(_samples([(0, 0), (0, 0), (8, 0), (0, 0), "?", None, (1, 0), (1, 0)]))
    assert [s.score for s in states] == [(0, 0), (1, 0)]
    assert states[0].count == 2  # the lone (0, 0) between misreads is dropped


def test_goals_in_order_with_team_and_score():
    result = _detect([(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1), (2, 1), (2, 1)])
    assert [(g.team, g.team_name, g.score) for g in result.goals] == [
        ("home", "SEW", "1-0"), ("away", "PSV", "1-1"), ("home", "SEW", "2-1"),
    ]
    assert result.goals[0].t == 8.0  # first keyframe showing the new score
    assert result.consistent()
    assert (result.home_name, result.away_name, result.half) == ("SEW", "PSV", "1ST")


def test_hidden_overlay_still_counts_goals_and_flags_gap():
    result = _detect([(3, 2), (3, 2)] + [None] * 10 + [(4, 3), (4, 3)])
    assert result.counts() == (1, 1)
    assert all("multiple_goals_one_interval" in g.flags for g in result.goals)
    assert all(any(f.startswith("overlay_unreadable") for f in g.flags) for g in result.goals)
    assert result.consistent()


def test_operator_correction_removes_goal():
    result = _detect([(0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 1), (0, 1)])
    assert [(g.team, g.score) for g in result.goals] == [("away", "0-1")]
    assert result.corrections == 1
    assert result.consistent()


def test_clip_starting_mid_game():
    result = _detect([(4, 3), (4, 3), (5, 3), (5, 3)])
    assert result.start_score == (4, 3)
    assert [g.score for g in result.goals] == ["5-3"]


def test_clock_consistency():
    samples = _samples([(0, 0)] * 5)
    assert clock_consistency(samples) == 1.0
    assert clock_consistency(_samples([None, None])) is None
