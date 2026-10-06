import io
import re

from avis_agent.style import enabled, paint

ANSI = re.compile(r"\033\[[0-9;]*m")
CARD = """──── Confirm your extension ────
Reservation    AVS-77001020
New return     Tuesday, October 6, 2026 at 6:00 PM PDT
Total          $60.08 USD"""


def plain(s: str) -> str:
    return ANSI.sub("", s)


def test_styling_never_changes_the_words():
    """kills: a styled reply that drops, reorders or alters text (a price or date lost in wrapping)."""
    long = "Agent: Sure.\n\n- One very long bullet " + "word " * 40 + "end."
    text = long + "\n— Connecting you with a representative —"
    styled = plain(paint(text, 60))
    assert styled.split() == text.split()
    assert max(len(line) for line in styled.splitlines()) <= 60


def test_card_box_is_aligned_and_keeps_every_value():
    """kills: a ragged box border, or a card value cut off by the box."""
    lines = plain(paint(CARD, 100)).splitlines()
    assert len({len(line) for line in lines}) == 1
    for row in CARD.splitlines()[1:]:
        assert any(row in line for line in lines)


def test_off_when_not_a_terminal_or_no_color(monkeypatch):
    """kills: escape codes written into piped output or logs."""
    assert not enabled(io.StringIO())

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert enabled(Tty())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not enabled(Tty())
