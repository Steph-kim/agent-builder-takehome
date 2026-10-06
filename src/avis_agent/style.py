"""Terminal styling for a real TTY: colour, wrapping and boxed cards. Presentation only.

`chat()` takes plain `read`/`write`; tests and sims pass their own, so they see the unstyled text and
nothing they grade changes. Only `main()` wraps the real terminal with these. Off when stdout isn't a
TTY, NO_COLOR is set, or TERM=dumb (https://no-color.org).
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from collections.abc import Callable

RESET, BOLD, DIM, ITALIC = "\033[0m", "\033[1m", "\033[2m", "\033[3m"
ACCENT = "\033[38;5;99m"  # violet
GREEN = "\033[38;5;35m"
AMBER = "\033[38;5;214m"
CYAN = "\033[38;5;44m"

AGENT = "Agent: "
CARD_RULE = "────"
TRANSFER = "— Connecting you with a representative —"
LABEL_W = 15  # render_card / render_receipt pad labels to this
MAX_WIDTH = 100


def enabled(stream=sys.stdout) -> bool:
    return stream.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def width() -> int:
    return min(shutil.get_terminal_size((MAX_WIDTH, 24)).columns, MAX_WIDTH)


def paint(text: str, cols: int) -> str:
    """One `write()` call's text, styled. The words are unchanged: only colour, wrapping and the card box."""
    if CARD_RULE in text.split("\n", 1)[0]:
        return _box(text)
    if text.startswith(AGENT):
        return _agent(text[len(AGENT) :], cols)
    if text.startswith("(") and text.endswith(")"):  # status, e.g. "(Checking availability…)"
        return f"{DIM}{ITALIC}{text}{RESET}"
    return text


def prompt(text: str) -> str:
    if text == "You: ":
        return f"\n{BOLD}{CYAN}You:{RESET}   "
    if text.startswith("Approve"):
        return f"{BOLD}{AMBER}{text}{RESET}"
    return f"{DIM}{text}{RESET}"  # email / hidden payment prompts


def banner(cols: int) -> str:
    title = " Avis support "
    hint = "type 'exit' or Ctrl-D to leave"
    rule = "─" * max(cols - len(title) - 2, 0)
    return f"{ACCENT}──{BOLD}{title}{RESET}{ACCENT}{rule}{RESET}\n{DIM}{hint}{RESET}\n"


Reader = Callable[[str], str]


def terminal_io(
    write: Callable[[str], None] = print, read: Reader = input
) -> tuple[Callable[[str], None], Reader, Callable[[Reader], Reader]]:
    """(write, read, wrap_secret) for the real terminal."""

    def styled_write(text: str) -> None:
        write(paint(text, width()))

    def styled_read(p: str) -> str:
        return read(prompt(p))

    def wrap_secret(read_secret: Reader) -> Reader:
        return lambda p: read_secret(prompt(p))

    return styled_write, styled_read, wrap_secret


def _agent(body: str, cols: int) -> str:
    indent = " " * len(AGENT)
    out = []
    for i, line in enumerate(body.split("\n")):
        lead = f"{BOLD}{ACCENT}Agent:{RESET} " if i == 0 else indent
        if not line.strip():
            out.append("")
            continue
        if line.strip() == TRANSFER:
            out.append(f"{lead}{BOLD}{AMBER}{TRANSFER}{RESET}")
            continue
        bullet = line.lstrip().startswith(("- ", "• ", "* "))
        hang = indent + " " * (len(line) - len(line.lstrip()) + (2 if bullet else 0))
        wrapped = textwrap.wrap(line.strip(), max(cols - len(hang), 20), break_on_hyphens=False) or [""]
        first = indent + " " * (len(line) - len(line.lstrip())) + wrapped[0]
        out.append(lead + first[len(indent) :])
        out.extend(hang + w for w in wrapped[1:])
    return "\n".join(out)


def _box(text: str) -> str:
    head, *rows = text.split("\n")
    title = head.strip("─ ").strip()
    colour = GREEN if "confirmed" in title.lower() else ACCENT
    inner = max([len(r) + 2 for r in rows] + [len(title) + 4])
    top = f"{colour}╭─ {BOLD}{title}{RESET}{colour} {'─' * (inner - len(title) - 3)}╮{RESET}"
    body = []
    for r in rows:
        label, value = r[:LABEL_W], r[LABEL_W:]
        strong = BOLD if label.strip() in ("Total", "Charged", "Confirmation") else ""
        cell = f"{DIM}{label}{RESET}{strong}{value}{RESET}"
        body.append(f"{colour}│{RESET} {cell}{' ' * (inner - len(r) - 1)}{colour}│{RESET}")
    bottom = f"{colour}╰{'─' * inner}╯{RESET}"
    return "\n".join([top, *body, bottom])
