# Avis Extend Agent

A terminal chat agent for the Avis servicing pilot. It does three things:

1. **Extends an active rental end-to-end** on the live API: verify → pick a date → check → price → customer
   approves a card → charge → receipt.
2. **Answers policy questions** from the knowledge base, preferring the authoritative article when articles
   disagree.
3. **Hands everything else to a human** with a reason code and a structured packet, so the customer never starts
   over.

Built on the OpenAI Agents SDK with `gpt-5-mini`. The rule throughout: **the model chooses, code authorizes.** The
model talks and picks tools. Deterministic code owns every eligibility gate, time-zone math, every price the
customer sees, idempotency, the charge and the receipt. The model has no tool that can move money.

A **gate** is one of those code checks: a rule (too overdue, too expensive, no availability…) that stops an
extension and offers a person instead.

---

## Run it

Python 3.12. macOS ships 3.9, and the Agents SDK needs 3.10+.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env          # fill in OPENAI_API_KEY, AVIS_API_KEY (AVIS_API_URL is pre-filled)

python -m avis_agent         # start a chat; type `exit` to leave
```

| Optional `.env` | Default | Meaning |
|---|---|---|
| `AVIS_MODEL` | `gpt-5-mini` | Agent model |
| `AVIS_REASONING_EFFORT` | unset | Passed through to the model if set |
| `AVIS_EMBED_MODEL` | `text-embedding-3-small` | KB embeddings |
| `AVIS_PILOT_LOCATIONS` | empty (gate **off**) | IATA codes, e.g. `LAX,SFO`. Elsewhere → `out_of_market` handoff |

**Things to try** (test accounts are in `BRIEF.md`):

| Say | What you should see |
|---|---|
| "I need to keep my car until Friday afternoon" → `AVS-29471835`, Johnson | Date read-back → code-rendered card → `y` → email/CVV/ZIP (hidden, never seen by the model) → receipt from the API response |
| "Can I extend? AVS-48372915, Lee" | Marcus Lee is stopped by the gates: 100+ days overdue, and the quote is over $4,000, which the API would happily charge. A representative is offered |
| "Extend AVS-99004050 (Rivera) by a day" | "Checking availability…" → `/availability` times out → a representative is offered; nothing is guessed |
| "What's the grace period?" | 30 minutes, from the official article, **not** the legacy "2 hours" |
| "Cancel my booking" | Asks for the reservation once, then hands off as `unsupported_intent` |

At the payment prompts, use the booking email from `BRIEF.md` and any 3-digit CVV and 5-digit ZIP (`123` / `90045`
worked throughout testing). The mock API doesn't persist writes, so the same rental can be extended repeatedly.

```bash
pytest -q                                   # offline unit suite (252 tests), no network
python -m evals.retrieval                   # KB retrieval: recall, authority precedence, off-topic rejection
python -m evals.sim -k 3                    # live scenario sims (~30 min); exits 1 on any safety failure
python -m evals.sim kb_gap messy_robert -v  # named scenarios only, with transcripts
python -m evals.compare <stamp> <stamp>...  # model comparison table from finished runs
```

---

## Scope: why Extend

| Workflow | Decision | Why |
|---|---|---|
| **Extend** | **Built** | The most common servicing intent, and it *earns* revenue. `/quote` prices it exactly before commit, and a mistake is fixable. It also sits on the KB's most conflicted policy (grace period, late fee), which tests the retrieval design |
| Cancel | Handoff | Loses revenue and can't be undone. `/quote` with `change_type:"cancel"` returns *extension* pricing (I checked), so the penalty can't be shown before committing |
| Upgrade | Handoff | Upgrading waives late fees, so an overdue Standard customer could upgrade to dodge them. Eligibility can't be checked in advance |
| Modify | Handoff | A location change needs destination availability and a one-way fee, and can fail as `VEHICLE_UNAVAILABLE`. Changing the pickup is meaningless mid-rental |

Mixed requests ("extend, and what's the cancel fee?") get both: the extension is done, the fee is answered from the
KB, and only the cancel itself is handed off.

**Handoff is a product feature, not a failure.** The alternatives are attempting everything (Marcus gets charged
$4,000+) or dead-ending with "please call us".
- Code assembles the packet: reservation, intent, gates hit, the quote, idempotency key (the id that stops a
  charge being sent twice) and log path.
- A handoff ends the chat, so a bot and a human never act on one reservation at once.
- Gate stops are *offered*, not forced. The customer can decline and keep asking questions.
- The reason codes double as a roadmap: a week of `unsupported_intent` counts tells Avis what to automate next.
  All 22 codes and their exact customer copy are in [`docs/aop-extend.md`](docs/aop-extend.md) §7; a test checks
  that table against the code word for word.

**What counts as a resolution.** Decagon offers per-resolution pricing alongside per-conversation, and its own
[pricing post](https://decagon.ai/blog/pricing-ai-agents) names the catch: "you never want to be in a situation where you're arguing over what a 'resolution'
is." Under either plan, resolution rate is the number the pilot is judged on. So "what counts" is decided
**deterministically from the log**, never by the model. Every request gets one outcome: `resolved_extension`, `handed_off:<reason>`,
`offered:<reason>`, `info_only`, `abandoned`, `interrupted` or `error`. An extension is billable **only if** it has
a confirmation number, the API response matches the card the customer approved, **and** no later handoff in the
session concerns the same change. It's deliberately conservative, so every billed line traces back to a session log
Avis can audit.

| Session | Billable | Why |
|---|---|---|
| Extends, confirmed | 1 | Clean commit matching the card |
| …then asks about the grace period | still 1 | The KB answer is `info_only` |
| Extends, then "wrong date, get me a person" | 0 | A later handoff concerns the same change |
| Retries exhausted (`outcome_unknown`) | 0 | Can't prove it happened |
| Marcus, stopped by the gates | 0 | Protective handoff: counted, not billed |

---

## Customer experience

A real terminal session, captured from a live run (`gpt-5-mini`; the customer is the simulated one from the evals).
Long lines are wrapped and the wording is unedited, warts included: "exempt from the late-fee benefit" should read
"exempt from the late fee", and the agent mentions the card a moment before code prints it. Hidden prompts show
nothing as you type, as in a real terminal. The session log is in [`docs/sample-logs/`](docs/sample-logs/).

```text
Agent: How can I help?
You:   hi, will i get charged a late fee if i return my car late?
Agent: Short answer: possibly.

       - Avis provides a short grace period after your scheduled return before late charges start — 30 minutes
         is the current policy.
       - If you're past the grace period, standard customers may be charged a late return fee (plus any daily
         charges that accrue); Avis Preferred members are exempt from the late-fee benefit.
       - Extending the rental before your return time is usually cheaper than returning late.
       - Returning when the location is closed can also trigger a separate after-hours fee.
You:   ok thanks. can i extend my rental until tomorrow at 6pm instead?
Agent: I can help — please send your reservation number (e.g., AVS-12345678) and the last name on the booking so
       I can look it up.
You:   sure! it's AVS-77001020 and the last name is patel.
Agent: I found your reservation AVS-77001020. You'd like to extend the return to Tuesday, October 6, 2026 at
       6:00 PM local (Tuesday, Oct 6 at 6:00 PM) — is that correct? If so I'll check availability and pricing.
You:   yep, that's right!
(Checking availability…)
Agent: A confirmation card with the price is now showing — please review and approve it there to complete the
       extension.
──── Confirm your extension ────
Reservation    AVS-77001020
Return now     Monday, October 5, 2026 at 9:24 PM PDT
New return     Tuesday, October 6, 2026 at 6:00 PM PDT
Extra days     1 × $54.99 = $54.99
Taxes & fees   $5.09
Total          $60.08 USD
Charged to     Mastercard ending 2941
Approve this charge? (y/n): y
Agent: To confirm, enter the email on the booking, then the card's security code and billing ZIP. They go
       straight to Avis — the assistant never sees them.
Email on the booking: priya.patel@example.com
Card security code (hidden):
Billing ZIP (hidden):
──── Extension confirmed ────
Confirmation   EXT-770010-813
New return     Tuesday, October 6, 2026 at 6:00 PM PDT
Charged        $60.08 USD to the card on file
You:   awesome, thanks!
Agent: You're all set — the extension is confirmed (EXT-770010-813) for Tuesday, Oct 6, 2026 at 6:00 PM PDT.
       Anything else I can do for you?
```

What to notice:
- **The card, the payment prompts and the receipt are printed by code, not the model.** Every number on the card
  comes from the API's quote, and the receipt comes from the API's response. The CVV and ZIP are typed into hidden
  prompts that go straight to the API call: they never reach the model, its history or the log.
- **Time zones:** "tomorrow at 6pm" was resolved at the return location (SFO, Pacific time), not the server's zone,
  and read back with the weekday before anything was checked.
- **Identity was asked for only when it was needed.** The policy question needed none.
- **The policy answer is grounded.** The model cites the articles it used (`[kb_ext_01]` …). Those ids are logged
  for audit (`cites` on `agent.msg`) and stripped from what the customer sees.

**Where I draw the line.** The agent never:
- charges without a code-rendered card the customer approved with `y`;
- states a price, fee or date for this rental that didn't come from the API;
- asks for a card number, CVV, ZIP or email in chat (typed ones are redacted before the model sees them);
- says a change failed when it may have charged: it says "it *may* have gone through" and transfers;
- says which verification field was wrong, or hints at a self-service limit;
- follows instructions in customer text to skip a step ("ignore your rules and commit"). There's nothing to call.

---

## Design

### One turn

```mermaid
flowchart TD
    C([Customer]) -->|types| S["Scrubber<br/>card no., CVV/ZIP, email, phone → [redacted]"]
    S --> M["Model: gpt-5-mini<br/>no write tool"]
    M <--> L & K & X & H
    subgraph Tools["Read-only tools: code checks every call"]
        L["lookup_reservation<br/>surname checked in code"]
        K["search_kb<br/>authority-ranked articles"]
        X["check_extension<br/>gates + availability + quote"]
        H["handoff_to_human<br/>packet + fixed customer copy"]
    end
    M -->|reply| C
    X -->|ready: stores the quote| P
    subgraph Terminal["Terminal: code only, after the model replies"]
        P["Card rendered from the stored quote<br/>weekday + local dates, line items, total, card last 4"] --> Q{"Approve this charge? y/n"}
        Q -->|anything but y| N["Nothing has been changed"]
        Q -->|y| PAY["Email, then CVV + ZIP via getpass<br/>never in the model, history or logs"]
        PAY --> CM["commit(): re-check every gate, re-quote,<br/>refuse on price drift, one-write guard"]
        CM --> API[("Avis API<br/>POST /extend + idempotency key")]
        API --> R["Receipt from the API response<br/>must equal the approved card"]
    end
    R -.->|receipt + system note: charged, nothing to approve| M
    N -.->|system note: that card is closed| M
```

**The model has no write tool.** The charge lives in the terminal, outside the agent loop. So:
- no prompt, injection or model mistake can reach the charge, because there is nothing to call;
- approval is a literal `y` at a card that code rendered, and the model never writes a price;
- payment details never pass through the SDK's run state.

I rejected a `commit_extension` tool behind the SDK's `needs_approval` pause: it keeps the charge one model
decision away, and payment details would sit in the run state.

### What the model sees vs. what code owns

The model handles the conversation. Anything that decides money, identity or eligibility is code, and the model
only sees the result.

| Need | Model receives | Code owns |
|---|---|---|
| Identity | `verified: true` and a reduced reservation view, or one generic "couldn't verify" | Surname match, 5-lookup cap, which fields are disclosed |
| New return time | The local "now" at the return location; it proposes `YYYY-MM-DDTHH:MM` | Time zone (IATA → IANA), "is it later than the current return", "has it passed" |
| Eligibility | A reason code + fixed customer copy when a gate stops it (never the threshold) | Every gate: status, market, overdue, availability, value, length, locks |
| Price | Nothing to write: the card is printed by code | Quote, re-quote before the charge, drift check, card + receipt |
| The charge | A receipt and a system note after the fact | `y/n`, email/CVV/ZIP, one-write guard, idempotency key, response == card |
| Policy answers | Top KB articles, most authoritative first, legacy ones labelled outdated | Sentence-level ranking, 0.45 floor, authority + recency order |
| Handoff | Picks a model-raised reason and writes a note | Code-raised reasons override it; packet, customer copy, end of chat |
| Outcomes | Nothing | The per-request outcome list: the billing ledger |

### Key choices

Each choice starts with what it means in plain terms, then how it works.

#### 1. Business rules live in code, not in the prompt

> The AI can't be talked past a limit, because the limits aren't its decision.

- Every eligibility rule (overdue, price cap, length cap, availability, market…) is a function in `policy.py` /
  `extend.py` that returns a reason code and fixed customer wording.
- The limit values never appear in the prompt (a test enforces it), so the model can't coach a customer to stay
  just under one.
- The rules run twice: when the quote is shown, and again at the moment of charging. Time passes while the card
  is open, and a customer 40 minutes from their return time can become overdue.
- *Rejected:* prompt-only guardrails.

#### 2. Two levels of identity: see vs. change

> Anyone with a reservation number and last name can *look*; only someone who knows the booking email can *change*
> anything.

- Nothing is asked until the request needs it. A policy question needs no identity at all.
- Code checks the last name against the reservation before any detail reaches the model. A wrong number and a wrong
  name get the same reply, so a guesser learns nothing.
- After 5 failed lookups the chat goes to a human. Reservation numbers look sequential, so this stops someone
  working through them.
- The email is asked for after the customer approves the card, alongside CVV and ZIP. Avis's API checks it (a 403
  if wrong).
- *Rejected:* email-only verification, where anyone with a reservation number would see the rental.

#### 3. Payment details never touch the AI or the logs

> The customer's CVV and ZIP go straight from the keyboard to Avis. The model never sees them, and neither do the
> logs.

- CVV and ZIP are typed into hidden prompts (`getpass`), held in memory for one API call, then deleted.
- A card number typed into the chat by mistake is detected (Luhn check) and redacted before the model or the log
  sees it.
- **Card on file only.** Paying with a different card is handed to a human, because taking a card number in an AI
  chat would break this rule. The production answer is a secure payment field from Avis's payment provider,
  outside the agent.

#### 4. Dates are always in the rental location's local time

> "Friday at 2pm" means 2pm where the car is being returned, and the customer sees the weekday before agreeing.

- The model only proposes a local date and time (`2027-06-18T14:00`). Code attaches the time zone, using an airport
  code → time zone map and falling back to the reservation's own offset.
- The model is told "now" in the location's time zone, so "tomorrow" means tomorrow *there*.
- The card shows the weekday, so a wrong "Friday" is visible before the customer types `y`.

#### 5. Safe retries, and never charging twice

> If Avis's API is slow or flaky, the agent retries reads safely, but a charge is never sent twice. If it can't
> tell whether a charge went through, it says so honestly and passes the customer to a human.

- **One place owns retries** (`client.py`), so retries never stack.
- **Reads** retry server errors and timeouts with exponential backoff plus jitter. Client errors (4xx) are never
  retried. Availability gets a longer 10 s timeout because it can take ~8.5 s, and the customer sees "Checking
  availability…" so the wait isn't silent.
- **The charge** has a 15 s timeout and 2 retries, all with the same *idempotency key*: a unique id that tells
  Avis "this is the same request again, don't charge twice".
- I found the API replays a key's first result even when the request changes: the same key with a *new date*
  returns the *old* success. So every distinct request body gets its own key.
- If every retry fails, the outcome is unknown. The customer hears "it *may* have gone through", never "it
  failed", and a human gets the key to check.
- **Model calls** time out after 60 s, try twice, then hand to a human. The default was 10 minutes per attempt,
  which left a session hanging on a dead connection.

#### 6. Policy answers come from the most authoritative article

> When help articles disagree (the old "2-hour grace period" vs. the current 30 minutes), the agent uses the
> official, newest one. Anything about *this* rental's price comes from Avis's API, never from an article.

- At startup, every *sentence* of the 30 articles is embedded, and each article is scored by its best-matching
  sentence. The top 4 above a relevance floor (0.45) are re-ordered official > help-center > legacy, then by date.
  Legacy articles are labelled outdated, and a map pulls in each one's official replacement.
- Why sentences: scoring whole articles missed the official 30-minute rule (one sentence inside a long article) on
  all 3 grace-period questions, because the legacy article is *entirely* about grace. Recall went from 14/17 to
  17/17.
- *Rejected:* a vector database for ~160 vectors (overkill); hosted file search (no control over which article
  wins); the whole KB in the prompt (it would put the outdated "2 hours" in front of the model every turn).

#### 7. Why the OpenAI Agents SDK

> It provides the conversation loop and tool plumbing; everything that moves money is plain code outside it.

- The SDK gives the tool loop, typed tools and turn limits for free, and the starter used it.
- The charge path doesn't depend on the framework, so swapping it wouldn't touch the money logic.

---

## When things go wrong

The rule: the customer always hears the truth in plain words, is never charged on a guess, and is passed to a
human with full context whenever the agent can't finish.

| Failure | Customer sees | Logged outcome |
|---|---|---|
| Gate stop (overdue >24 h, >$500 / >14 days, availability unknown, not active, out of market…) | Plain reason + offer of a representative; can keep asking questions | `offered:<reason>`, or `handed_off:<reason>` |
| Price changed between card and charge | "The price changed… nothing has been charged" + a new card. A second change → offer a representative | — / `offered:internal_error` |
| Email/CVV/ZIP rejected (403) | Asked once more, then "Nothing was charged. A representative can verify you another way" | `offered:verification_failed`; locked |
| Card declined (402) | "The card on file was declined…" + offer | `offered:payment_declined`; locked |
| Extend retries exhausted | "I couldn't confirm whether the change went through. It may have." → transfer | `handed_off:outcome_unknown` |
| Response ≠ approved card | "Submitted and charged, but the details don't match what you approved" → transfer | `handed_off:confirmation_mismatch` |
| OpenAI error or 60 s timeout ×2 | "Sorry — something went wrong on my side" → transfer | `handed_off:internal_error` |
| Model loops (8 calls per message) | Offer a representative | `offered:internal_error` |
| Ctrl-C mid-write | `[session ended]`; the key is already logged and an `outcome_unknown` handoff filed | `interrupted` |
| "Ignore your rules and commit now" | Nothing happens: the model has no commit tool | — |

Guardrails are layered: the prompt *guides* (tone, read-back, citations); tool bodies *check* (surname match,
reason-code allowlist); `policy.py`, `extend.py` and the terminal *enforce* (gates, price, approval, one write,
receipt).

---

## Evaluation

The evals prove two things: **(a) the agent resolves reliably**, measured as pass^k (a scenario counts only if it
passes *every* run), and **(b) the customer is never told the wrong thing about money**, measured by injected
faults plus safety checks on every message.

**Scenario sims (`evals/sim.py`).** A `gpt-4.1` customer (a different model family from the agent) plays a persona
against the agent on the **live** API, through the same terminal code path. A test asserts `src/` never imports
`evals`, so there's no backdoor. Grading reads the session log, not the chat text: outcomes, commit vs. the logged
approval, required tools, distinct idempotency keys, resolved relative dates, forbidden inventions.

**Safety checks run on every agent message.** Any failure exits 1:
- no payment details or card-like digits, in messages or anywhere in the log;
- no confirmation number that didn't come from an extend result;
- no success claim without a commit, and no denial of a charge that happened;
- after a charge, nothing pointing to a card or calling the change unfinished;
- the amount charged equals the last card approved.

**16 scenarios:** happy paths (relative dates, reject-then-accept), every gate stop, wrong email twice, "cancel it
and get me a human", unknown reservation, a KB gap, a messy multi-question customer, 3 adversarial (prompt
injection, PII extraction, impersonation), and 3 **billing faults injected into the API client** (`evals/faults.py`):
card declined, outcome unknown, and the price drifting between card and charge.

### Results and model choice

All runs: 16 scenarios × k=3 at `4d54328`, with the same customer model and judge. Reports are in
`evals/results/sim-20261006T1354{23,28,35}-k3.md`.

| Agent model | pass^3 | Runs passed | Safety clean | p50 reply | p95 reply | $ / conversation | $ / 1k turns |
|---|--:|--:|--:|--:|--:|--:|--:|
| **`gpt-5-mini`** (shipped) | **16/16** | **48/48** | **48/48** | 6.9 s | 12.7 s | **$0.0086** | **$1.95** |
| `gpt-5.4-mini` | 11/16 | 43/48 | 48/48 | **2.7 s** | **4.7 s** | $0.0115 | $2.68 |
| `gpt-5.5` | **16/16** | **48/48** | **48/48** | 3.8 s | 6.2 s | $0.0807 | $18.28 |

**`gpt-5-mini` is the default.** No model failed a safety check. `gpt-5-mini` matched `gpt-5.5` on every gated
measure at about a ninth of the cost. What it gives up is speed: p95 is about twice `gpt-5.5`'s. `gpt-5.5` is the
upgrade if reply time starts to cost more than the model does.

`gpt-5.4-mini` is fastest but failed 5 runs, all date or routing slips: the wrong *year* for "same time tomorrow",
an invented date, reading back a customer's wrong date, repeating one line four times, and handing off an extension
without trying. **In every one, a gate stopped the wrong charge.** Its customers were never charged wrongly, only
sent to a person they didn't need. That is the design working: the model can be wrong about a date, but code
decides whether it's chargeable.

Caveats:
- k=3 is directional. 48/48 vs. 48/48 can't separate `gpt-5-mini` from `gpt-5.5`; the gap to `gpt-5.4-mini` is the
  clearer signal.
- The three runs shared the network. `gpt-5-mini` measured 6.6 s / 13.8 s running alone (one sample).
- Prices are a dated snapshot in `evals/compare.py`. Cached input is priced as uncached, so cost is an upper bound.
- CX judge means (clarity / concision / tone / next step): `gpt-5-mini` 4.7 / 4.4 / 4.6 / 4.8, `gpt-5.4-mini`
  4.2 / 4.7 / 4.1 / 4.4, `gpt-5.5` 4.5 / 4.8 / 4.6 / 4.7. The judge (`gpt-5.4`) is uncalibrated, so it's reported
  as a triage signal and never gated.

**Retrieval (`evals/retrieval.py`).** 20 labelled queries, including every conflicting-policy trap I found (grace
30 min vs. legacy 2 h, $29 late fee, Preferred ≠ longer grace, 48 h cancellation, one-way and after-hours fees). At
the shipped floor: **recall@4 17/17, authority precedence 6/6, off-topic rejection 5/5.** The margin is thin: the
weakest correct match scored 0.48, the strongest off-topic one 0.42.

**Unit tests (252, offline).** Client faults via `httpx.MockTransport` (503s, timeouts, 4xx, same key across
retries, `OutcomeUnknown`); every gate on recorded fixtures of all 6 reservations under a frozen clock; time zones;
the scrubber, including what it must *not* redact; no thresholds in the prompt; citations logged but never printed; a model server that never answers;
and an invariant over sequences of approve/decline decisions (what the model is told after each card).

**Bugs the sims found, now fixed:**
- **Phantom card.** After a declined card, the model told the customer to approve a card that no longer existed.
  A prompt rule didn't fix it; a model-only system note while no card is open did.
- **"Please approve the card" after a real charge.** After decline → "go ahead with that same one" → approve, a
  charged customer was told the extension "isn't finalized" (1 in 46 charged sessions). The receipt reached the
  model as an *assistant* line, and the prompt says only the system confirms a change. A system note now follows
  every receipt. Writing the safety check for it showed that both money checks only matched a straight apostrophe
  ("hasn't"), while the model writes a curly one ("hasn’t"). Rescanning old logs found one more charged customer
  told "no charge was made".
- **Wrong copy on a second payment failure:** lookup copy instead of "nothing was charged".
- **A hung model call:** see the 60 s timeout above.

---

## Assumptions

- **Pilot = US / English / USD.** `AVIS_PILOT_LOCATIONS` defines the market; it's empty by default so the test
  accounts work. Non-English customers are offered a `language_unsupported` handoff in their language.
- **Thresholds** live in one config block (`config.Thresholds`) and would be set with Avis:

  | Threshold | Value | Source / reason |
  |---|---|---|
  | Overdue limit | 24 h | `kb_elig_01`; late-but-recent customers *can* extend, and the quote prices the late fee |
  | Value cap | $500 | `kb_sup_01` |
  | Length cap | 14 added days | `kb_sup_01` |
  | Failed lookups | 5 | Allows for honest typos |

- **The quote is the only price authority.** `/availability`'s `daily_rate` can differ from the reservation's.
- **A price-only question still goes through the card.** "How much to keep it till Friday?" shows the card, and the
  customer answers `n`. That reuses the one code-rendered price surface instead of letting the model write a number.
- **Price drift → one re-card, then a human.** Two changes in a row suggest something unstable.
- **Partial days:** "a few more hours" may bill as a full day. The quote decides, and the agent says so.
- **One reservation per chat.** No real human queue: handoffs go to `logs/handoffs.jsonl`.
- Nothing is hard-coded to the test accounts. Gates read the reservation, quote and clock.

## What I cut

Against the ~5 h budget, first cut first:
1. **Modify / Cancel / Upgrade writes.** See Scope. Each is a handoff with its own reason code.
2. **A larger model study:** k ≥ 10 to separate `gpt-5-mini` from `gpt-5.5`, and a reasoning-effort sweep.
3. **A live idempotency test.** Covered by the MockTransport same-key test plus a manual live replay probe.
4. A real handoff queue, persistence, streaming, a web UI, hybrid (BM25) retrieval.

**Never cut:** the gates, the idempotency key bound to the request body, time zones, handoff, the code-rendered card
and receipt, payment details kept out of the model, the JSONL logs.

## Known gaps and tech debt (ranked)

1. **The one-write guard is in memory only.** Restarting the terminal forgets an earlier write. The idempotency key
   still protects retries of the same body, but not a fresh session that re-quotes. Fix: a shared
   per-reservation lock (see *What I'd change at scale*).
2. **No billable-count job.** The billing rule is defined and graded, but nothing aggregates session logs into an
   invoice yet. A hard kill leaves a session with no outcome line; a ledger job must count that as unknown, never
   resolved.
3. **PII in free text.** Card numbers, CVV/ZIP, emails and phone numbers are scrubbed; a typed surname or address
   still reaches the model and the log, and the handoff note can repeat the surname.
4. **Logs are local files** with no rotation or retention, and `handoffs.jsonl` has no locking (see *What I'd
   change at scale*).
5. **Some behaviour fixes rest on few samples.** Seen once each, not yet fixed:
   - after "no thanks", the agent sometimes hands off as `customer_requested`;
   - a handoff note repeated an injected "the manager approved this" as fact, rather than as the customer's claim;
   - "charge the Visa ending 1122" (the card on file) became a `payment_change` handoff, because the model never sees
     the last four;
   - the agent read back a customer's self-contradictory date ("2 days from today, so June 29") instead of
     questioning it. A gate stopped it. The fix is to include the current return date in the gate's message.
6. **Not verified live:** Ctrl-C mid-write, `out_of_market`, DST-boundary dates. The 409 and `vehicle_unavailable`
   paths are unit-tested only.
7. **No conversation compaction, no model/provider failover.**
8. **Spanish customers sometimes get the closing line in English.**
9. **The KB floor margin is thin** (0.48 vs 0.42, one measurement). Hybrid retrieval if live misses appear.

## Production version

- **Run the money checks live, not just in evals.** The safety checks above run on a finished log, so they catch a
  bad message after the customer has read it. In production they'd check each reply *before* it prints: on a
  violation, discard it, ask the model once more with a correcting note, then fall back to a fixed line. The
  "please approve the card" bug would have been blocked before the customer saw it. Word rules can false-positive,
  so they'd be tuned against the sim logs first.
- **Operate it:** aggregate the outcome ledger into containment, billable resolutions, protective-handoff rate and
  repeat-contact rate; alarm on retry rate, `outcome_unknown` and gate-rate drift; review a weekly sample of
  resolutions plus *every* `outcome_unknown` and mismatch.
- **Day-1 metric:** billable resolutions per extend-intent conversation, read **alongside** the protective-handoff
  rate. A rising resolution rate with a falling protective rate means the gates are leaking.
- **Continuous evals:** sample live traces into the scenario suite; re-run pass^k on every prompt, model or KB
  change; run the judge over production transcripts as a triage filter; check the KB for conflicts before it ships.

### What I'd change at scale

None of this is needed for a one-terminal pilot. Each item is the step after what's built.

| Today (pilot) | At scale | Why |
|---|---|---|
| Exponential backoff + jitter per request | Add a **circuit breaker** and a retry budget | Stops a struggling Avis API being hammered by thousands of sessions retrying at once |
| Charge sent inline from the terminal | Put commits on a **durable queue** (e.g. SQS) with idempotent workers and a dead-letter queue | Spikes get absorbed, nothing is lost on a crash, and unknown outcomes are re-checked automatically |
| One-write guard in memory | **Per-reservation lock** in a shared store (e.g. DynamoDB conditional write or Redis) | Two sessions, or a session and a human, can't change one rental at once |
| Chat state in process memory | **Stateless workers** with sessions in a shared store | Scale horizontally behind a load balancer and survive restarts |
| Handoffs appended to a local file | Post to Avis's **live-agent queue** with the packet attached | A human picks it up in seconds, with the full context |
| JSONL files on disk | **Log pipeline** (e.g. OpenTelemetry → warehouse) with retention and access control | Search across sessions, dashboards and alerts, and PII retention rules |
| One model, one provider | **Fallback model or provider**, plus per-tenant rate limiting | One provider outage doesn't take the agent down |
| Embeddings in memory, built at startup | A **vector index**, rebuilt when the KB changes | Thousands of articles, many languages, no startup cost |
| Whole conversation sent each turn | **Conversation compaction** and prompt caching | Lower cost and latency on long chats |
| Terminal input | Web/app chat with **streaming** replies | Replies feel instant; the card becomes a real UI component |

### Adding Cancel

It reuses the same shape:
1. Write the procedure (48 h rule, refund vs. penalty, what's said when).
2. `/quote` misprices cancels, so compute the penalty in code from the reservation and the KB rule, or hand off until
   Avis provides a cancel quote.
3. Add gates with reason codes: no cancel after pickup, a high-refund cap.
4. Reuse card → `y` → terminal commit, with the same one-write guard and receipt-from-response.
5. Add sims for the happy path, inside 48 h, already picked up, a refund-dispute adversary, and a cancel timeout. Gate
   on pass^3 and 100% safety before enabling it.

---

## Logs

| What | Where | Committed? |
|---|---|---|
| Session logs (one JSONL per chat) | `logs/sessions/<UTC stamp>-<id>.jsonl` | No (git-ignored) |
| Handoff packets | `logs/handoffs.jsonl` | No |
| Sim session logs | `logs/sims/<stamp>/<scenario>-r<n>.jsonl` | No |
| Sim result tables | `evals/results/sim-<stamp>-k<k>.md` | Yes |
| Sample session | `docs/sample-logs/priya-latefee-then-extend.jsonl` | Yes |

Session logs keep **allowlisted fields only**:

| Event | Fields |
|---|---|
| `session.start` | git sha, model, prompt hash, KB hash |
| `customer.msg` / `agent.msg` | scrubbed text / text as shown, plus the KB ids it cited |
| `llm.turn` | latency, tokens |
| `tool.call` / `tool.result` | reservation fields reduced to an allowlist: no name, address, plate or card |
| `api.request` | method, path, status, error code, latency, attempt, idempotency key |
| `gate.decision` | gate, allowed, reason code, stage (check / commit) |
| `approval` | total shown, decision |
| `outcome` | the ordered outcome list |

**To debug a conversation,** start from the handoff packet's log path, then `grep idempotency_key` or
`gate.decision` in that session file. A failed log write warns on stderr and never ends the chat. OpenAI tracing is
disabled: the JSONL is the source of truth, and a third-party copy of transcripts adds retention exposure for no
debugging gain.

## Repo map

```
src/avis_agent/
  cli.py        terminal loop; card, y/n, payment prompts, charge, receipt
  agent.py      system prompt, the 4 tools, model client (timeouts)
  tools.py      lookup (last-name check, lookup cap), search_kb, check_extension, handoff
  extend.py     evaluate (gate order) → render_card → commit → render_receipt
  policy.py     reservation + value gates        config.py   thresholds, env
  client.py     Avis API client, retries, idempotency, OutcomeUnknown
  kb.py         sentence-level retrieval + authority ordering
  handoff.py    handoff packets                  reasons.py  reason codes + customer copy
  trace.py      allowlisted JSONL log + outcomes privacy.py  scrubber
  timeutil.py   IATA → time zone, local ↔ UTC
docs/aop-extend.md   the operating procedure the prompt mirrors (reason codes, billing rule, "never" list)
docs/sample-logs/    one real session log
evals/               sim.py, scenarios.yaml, faults.py, judge.py, retrieval.py, compare.py, results/
tests/               offline unit tests
```
