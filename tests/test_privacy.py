import pytest

from avis_agent.privacy import REDACTED, scrub


@pytest.mark.parametrize(
    "text, kind, leaked",
    [
        ("my card is 4111 1111 1111 1111 thanks", "card_number", "4111"),
        ("card 4111-1111-1111-1111", "card_number", "1111-1111"),
        ("use 4111111111111111", "card_number", "4111111111111111"),
        ("my cvv is 847", "cvv", "847"),
        ("CVV: 8470", "cvv", "8470"),
        ("security code 123, go ahead", "cvv", "123"),
        ("billing zip 90210", "zip", "90210"),
        ("zip code is 90210-1234", "zip", "90210"),
        ("my email is sarah.johnson@example.com", "email", "sarah.johnson@example.com"),
        ("call me at 310 555 0199", "phone", "555 0199"),
        ("phone on file (310) 555-0142.", "phone", "555-0142"),
        ("text +1 310.555.0142", "phone", "555.0142"),
        ("cell 3105550142 thx", "phone", "3105550142"),
    ],
)
def test_volunteered_secrets_are_redacted(text, kind, leaked):
    result = scrub(text)
    assert kind in result.kinds
    assert leaked not in result.text and REDACTED in result.text


def test_volunteered_cvv_alongside_a_reservation_keeps_the_id():
    result = scrub("Extend AVS-29471835 to Friday, cvv 847 zip 90210")
    assert result.text == f"Extend AVS-29471835 to Friday, cvv {REDACTED} zip {REDACTED}"


@pytest.mark.parametrize(
    "text",
    [
        "Reservation AVS-29471835 please",
        "AVS-1234567812345670",  # Luhn-valid digits inside an id
        "return it 2027-06-17T14:00:00-07:00",
        "on 2027-06-17 at 14:00",
        "the total was $100.49, confirmation EXT-20270617-8841",
        "flight AA 2189 lands 14:30, gate 23",
        "I paid 459.80 for 10 days",
        "ref 0123456789 and 1234567890",  # US area codes/exchanges never start with 0 or 1
        "4111 1111 1111 1112",  # fails Luhn → not a card
        "I'll be 30 minutes late, flight UA 1234",
    ],
)
def test_ordinary_support_text_is_untouched(text):
    result = scrub(text)
    assert result.text == text and not result.redacted
