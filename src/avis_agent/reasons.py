"""Handoff reason codes — must match the table in docs/aop-extend.md (tests/test_reasons.py checks).

*Code*-raised reasons come from a gate or tool result; the model can't skip them. *Model*-raised
reasons are the only ones `handoff_to_human` accepts from the model, so it can't invent a gate
outcome (e.g. claim `high_value` to dodge a charge) or launder one into another.
"""

from __future__ import annotations

from enum import StrEnum


class ReasonCode(StrEnum):
    # raised by code
    NOT_ACTIVE = "not_active"
    OUT_OF_MARKET = "out_of_market"
    OVERDUE_BEYOND_POLICY = "overdue_beyond_policy"
    HIGH_VALUE = "high_value"
    AVAILABILITY_UNKNOWN = "availability_unknown"
    VEHICLE_UNAVAILABLE = "vehicle_unavailable"
    NOT_AN_EXTENSION = "not_an_extension"
    VERIFICATION_FAILED = "verification_failed"
    PAYMENT_DECLINED = "payment_declined"
    WRITE_ALREADY_DONE = "write_already_done"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CONFIRMATION_MISMATCH = "confirmation_mismatch"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    # raised by the model
    CUSTOMER_REQUESTED = "customer_requested"
    SAFETY_INCIDENT = "safety_incident"
    DISPUTE = "dispute"
    UNSUPPORTED_INTENT = "unsupported_intent"
    PAYMENT_CHANGE = "payment_change"
    SECOND_RESERVATION = "second_reservation"
    LANGUAGE_UNSUPPORTED = "language_unsupported"
    KB_GAP = "kb_gap"


MODEL_RAISED: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.CUSTOMER_REQUESTED,
        ReasonCode.SAFETY_INCIDENT,
        ReasonCode.DISPUTE,
        ReasonCode.UNSUPPORTED_INTENT,
        ReasonCode.PAYMENT_CHANGE,
        ReasonCode.SECOND_RESERVATION,
        ReasonCode.LANGUAGE_UNSUPPORTED,
        ReasonCode.KB_GAP,
    }
)
CODE_RAISED: frozenset[ReasonCode] = frozenset(ReasonCode) - MODEL_RAISED

# Code-raised reasons where the bot carrying on could do harm (a charge in an unknown or wrong
# state): transfer right after the agent's reply. Every other code-raised reason is *offered* —
# code locks the action, the customer decides when to go to a person, and can keep asking
# policy questions meanwhile. Model-raised reasons transfer when the model calls the tool.
TRANSFER_NOW: frozenset[ReasonCode] = frozenset(
    {ReasonCode.OUTCOME_UNKNOWN, ReasonCode.CONFIRMATION_MISMATCH}
)
OFFERED: frozenset[ReasonCode] = CODE_RAISED - TRANSFER_NOW

# Must match the AOP table's "Customer hears" column (tests/test_reasons.py checks). Offered reasons
# are relayed by the agent as an offer; the rest are printed by the CLI at transfer.
CUSTOMER_COPY: dict[ReasonCode, str] = {
    ReasonCode.NOT_ACTIVE: (
        "This reservation isn't active, so I can't change it here. "
        "A representative can help — want me to connect you?"
    ),
    ReasonCode.OUT_OF_MARKET: (
        "Changes for this location are handled by our team. Want me to connect you with a representative?"
    ),
    ReasonCode.OVERDUE_BEYOND_POLICY: (
        "Because this rental is past its return time, a representative needs to review the extension. "
        "Want me to connect you?"
    ),
    ReasonCode.HIGH_VALUE: (
        "An extension this size needs a representative to confirm. Want me to connect you? "
        "I'll pass on the quote so you don't have to repeat anything."
    ),
    ReasonCode.AVAILABILITY_UNKNOWN: (
        "I can't confirm the car is free for those dates right now. Want me to pass this to a representative?"
    ),
    ReasonCode.VEHICLE_UNAVAILABLE: (
        "Your car isn't available for those dates. "
        "A representative can look at other options with you — want me to connect you?"
    ),
    ReasonCode.NOT_AN_EXTENSION: (
        "That would shorten the rental, which a representative handles. Want me to connect you?"
    ),
    ReasonCode.VERIFICATION_FAILED: (
        "I wasn't able to verify the reservation, so I can't look it up in this chat. "
        "I can still answer policy questions, or connect you with a representative who can verify you "
        "another way."
    ),
    ReasonCode.PAYMENT_DECLINED: (
        "The card on file was declined. A representative can help with payment — want me to connect you?"
    ),
    ReasonCode.WRITE_ALREADY_DONE: (
        "I've already made a change to this reservation in this chat, so a representative will need to "
        "handle anything further. Want me to connect you?"
    ),
    ReasonCode.OUTCOME_UNKNOWN: (
        "I couldn't confirm whether the change went through. It may have. "
        "A representative will check and confirm with you."
    ),
    ReasonCode.CONFIRMATION_MISMATCH: (
        "Your change was submitted and charged, but the details don't match what you approved. "
        "A representative will reconcile it."
    ),
    ReasonCode.INTERNAL_ERROR: (
        "Something went wrong on my side. Want me to pass this to a representative so you don't have to "
        "start over?"
    ),
    ReasonCode.SERVICE_UNAVAILABLE: (
        "Our systems aren't responding right now. Want me to pass this to a representative?"
    ),
    ReasonCode.CUSTOMER_REQUESTED: "Of course — connecting you with a representative now.",
    ReasonCode.SAFETY_INCIDENT: (
        "I'm sorry — let's get you to a person right away. If anyone is hurt or in danger, call 911 now."
    ),
    ReasonCode.DISPUTE: "I'll connect you with a representative who can look into that.",
    ReasonCode.UNSUPPORTED_INTENT: (
        "I can't make that change here yet, but a representative can. I'm connecting you now."
    ),
    ReasonCode.PAYMENT_CHANGE: (
        "I can only charge the card on file. A representative can take a new card securely."
    ),
    ReasonCode.SECOND_RESERVATION: "I'll pass the other reservation to a representative.",
    ReasonCode.LANGUAGE_UNSUPPORTED: (
        "I can only help in English right now. I can connect you with a representative."
    ),
    ReasonCode.KB_GAP: "I don't have that in our policy information. A representative can answer it.",
}
