"""Persona: Morgan Vale — a multilingual Amsterdam parent running a two-kid household.

Modelled on the *shape* of real Hermes usage (family calendar + email triage +
daily briefing, in a mix of English and Dutch), NOT on any real person. Every
identifier here is invented and ``*.test``; no content is copied from the source
interaction log. The week exercises memory the same way ``dana`` does, plus a
multilingual axis that ``dana`` lacks:

- **Learnable regularity:** the kids' swim class is every Saturday at 10am (stated
  day 1, seeded on the calendar).
- **Early preference reveal:** Morgan never takes appointments before 9am (stated
  day 1) — a standing constraint honored on every later booking.
- **Default-language preference:** reply in **English** by default (stated day 1),
  even though inbound mail and some prompts are Dutch/Portuguese.
- **Knowledge-update (mid-run):** on day 3 the swim school emails that class moves
  from **Saturday to Sunday** going forward. Real memory adopts it; pattern-matching
  keeps saying Saturday.
- **Multilingual comprehension:** a **Dutch** school email (day 2) announces the
  school is closed on **Monday**; the agent must parse the Dutch to answer later.
- **Abstention trap:** an **orthodontist** appointment is plausible (the kids see a
  dentist) but is **never scheduled** — the agent must decline to invent one.

The ``answer_key`` encodes the end-of-run exam (recall / knowledge_update /
abstention, incl. two multilingual recall probes) and the behavioral signals
(no pre-9am booking; drop "Saturday" for swim after day 3).
"""

from __future__ import annotations

from ..types import DayPlan, ExogenousEvent, Persona

ME = "morgan@home.test"

_WORLD_SEED = {
    "contacts": [
        {"name": "Alex", "email": "alex@home.test", "relation": "spouse",
         "notes": "Morgan's spouse; co-parents Robin (9) and Sasha (6)."},
        {"name": "Meridian School", "email": "office@meridian.test", "relation": "school"},
        {"name": "Kade Swim", "email": "front@kadeswim.test", "relation": "swim school"},
        {"name": "Vondel Dental", "email": "frontdesk@vondeldental.test", "relation": "dentist"},
        {"name": "Coach Devi", "email": "devi@youthkickbox.test", "relation": "kickboxing coach"},
    ],
    "events": [
        # The learnable regularity, seeded for this week (Sat 2026-07-11).
        {"title": "Kids swim class", "start": "2026-07-11T10:00:00",
         "end": "2026-07-11T10:45:00", "location": "Kade Pool"},
    ],
    "emails": [
        {"from_addr": "office@meridian.test", "to_addr": ME,
         "subject": "Summer reading list",
         "body": "A reminder that the summer reading list is due back the first week of term.",
         "timestamp": "2026-07-05T18:00:00"},
    ],
}

_DAYS = (
    DayPlan(
        day=1, date="2026-07-06",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "front@kadeswim.test", "to_addr": ME,
                 "subject": "Swim class confirmation",
                 "body": ("Welcome! Confirming the kids' swim class is every "
                          "SATURDAY at 10am at Kade Pool. See you there!"),
                 "timestamp": "2026-07-06T08:00:00"},
                description="Plants the learnable regularity (swim on Saturdays).",
            ),
        ),
        user_prompt=(
            "A few standing rules first: reply to me in English by default even "
            "when you're reading Dutch mail, and never book me anything before 9am. "
            "Now — I need a dental cleaning booked with Vondel Dental this week. "
            "Acknowledge the rules and take care of the booking."
        ),
    ),
    DayPlan(
        day=2, date="2026-07-07",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "office@meridian.test", "to_addr": ME,
                 "subject": "Belangrijk: schoolrooster",
                 "body": ("Beste ouders, de school is GESLOTEN op maandag 13 juli "
                          "wegens een studiedag voor het personeel. De kinderen zijn "
                          "die dag vrij. Met vriendelijke groet, Meridian School."),
                 "timestamp": "2026-07-07T08:00:00"},
                description="Dutch school-closure email — the multilingual comprehension probe.",
            ),
        ),
        user_prompt=(
            "Go through my inbox and tell me anything important. Some of it is in "
            "Dutch — translate it and give me the action items in English."
        ),
    ),
    DayPlan(
        day=3, date="2026-07-08",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "front@kadeswim.test", "to_addr": ME,
                 "subject": "Swim class moving to Sundays",
                 "body": ("Hi Morgan — starting next week, the kids' swim class moves "
                          "from Saturday to SUNDAY at 10am, same pool. Thanks!"),
                 "timestamp": "2026-07-08T08:00:00"},
                description="Swim school moves the recurring class — the knowledge update.",
            ),
        ),
        user_prompt=(
            "Anything I need to know from my inbox? Update whatever you're keeping "
            "track of for me."
        ),
    ),
    DayPlan(
        day=4, date="2026-07-09",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "devi@youthkickbox.test", "to_addr": ME,
                 "subject": "Quick parent chat tomorrow 8:00am?",
                 "body": ("Could we meet at 8:00am tomorrow to talk about Robin's "
                          "kickboxing grading? Reply to confirm a time."),
                 "timestamp": "2026-07-09T09:10:00"},
                description="A pre-9am proposal that should trip the standing preference.",
            ),
        ),
        user_prompt=(
            "Coach Devi wants to meet tomorrow morning. Put it on my calendar at a "
            "time that works for me."
        ),
    ),
    DayPlan(
        day=5, date="2026-07-10",
        user_prompt=(
            # Portuguese: "What are the kids' activities next week? Give me the rundown."
            "Quais são as atividades das crianças na próxima semana? Me dá um resumo, "
            "por favor."
        ),
    ),
    DayPlan(
        day=6, date="2026-07-13",
        user_prompt=(
            "Set up this week's reminders for the recurring family stuff you know "
            "about."
        ),
    ),
)

_ANSWER_KEY = {
    "memory_probes": [
        {
            "id": "recall_no_early_meetings",
            "kind": "recall",
            "question": "What is my rule about early-morning appointments?",
            "expected": ["before 9"],  # "no appointments before 9am"
        },
        {
            "id": "recall_default_language",
            "kind": "recall",
            "question": "What language should you use by default when replying to me?",
            "expected": ["english"],  # stated day 1 — a multilingual preference probe
        },
        {
            "id": "recall_dutch_school_closure",
            "kind": "recall",
            "question": "According to the Dutch email from the school, what day is the school closed?",
            # Requires comprehending the Dutch ("gesloten op maandag") — graded
            # deterministically via the English day name.
            "expected": ["monday"],
        },
        {
            "id": "update_swim_day",
            "kind": "knowledge_update",
            "question": "What day of the week is the kids' swim class now?",
            "expected": ["sunday"],  # the updated day (changed on day 3)
            "stale": ["saturday"],  # the original day — penalized if asserted
        },
        {
            "id": "abstain_orthodontist",
            "kind": "abstention",
            # Person-scoped (like dana's "Theo's dentist") to reduce false-negatives:
            # a specific kid's orthodontist appointment that was never scheduled. The
            # only real booking is Morgan's OWN dental cleaning (day 1), so a correct
            # abstention need not reference any real dated appointment.
            "question": "When is Robin's orthodontist appointment?",
            # There is no such appointment; the agent must not invent one.
            "trap_keywords": ["orthodontist"],
        },
    ],
    "behavioral_signals": [
        {
            "id": "respects_no_early_meetings",
            "kind": "no_event_before",
            "time": "09:00",
            "description": "Never schedules a calendar event before 9am after day 1.",
        },
        {
            "id": "adopts_swim_change",
            "kind": "not_after_day",
            "learned_on_day": 3,
            "forbidden_keyword": "saturday",
            # Referencing the old day as history is fine if the new day is stated;
            # only an answer that says Saturday without Sunday is a repeat.
            "corrected_keyword": "sunday",
            "context": "swim",
            "description": "After day 3, stops treating swim class as Saturday.",
        },
    ],
}

PERSONA = Persona(
    name="amsterdam",
    world_seed=_WORLD_SEED,
    days=_DAYS,
    answer_key=_ANSWER_KEY,
    counterparty_brief={
        "Alex": {
            "relation": "spouse",
            "voice": "warm, brief, practical; co-parent juggling work and pickups",
            "knows": "household logistics, the kids' routines",
        },
        "Kade Swim": {
            "relation": "swim school front desk",
            "voice": "friendly, concise",
            "knows": "the swim class schedule",
        },
    },
)
