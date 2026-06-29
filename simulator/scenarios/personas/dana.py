"""Persona: Dana Okafor — a working parent coordinating a two-kid household.

The week is built to exercise memory, not just task completion:

- **Learnable regularity:** Mia has soccer every Thursday at 4pm (stated day 1,
  seeded on the calendar).
- **Early preference reveal:** Dana never takes appointments before 9am (stated
  day 1) — a standing constraint the agent should honor on every later booking.
- **Two preferences/regularities that CHANGE mid-run** (knowledge-update is the
  dimension that most distinguishes real memory from pattern-matching, so the week
  exercises it twice from different angles):
  - on day 2 Coach Rivera emails that soccer moves from Thursday to **Wednesday**
    at 4pm going forward;
  - on day 5 Riverside Swim emails that Theo's swim class moves from Tuesday to
    **Monday**.
  A model with real memory adopts both; one that pattern-matches keeps asserting
  the original day.
- **Abstention trap:** Theo's dentist appointment is plausible (Dana books her
  *own* cleaning, the kids see Dr. Patel) but is **never scheduled**. A model
  should decline to invent one.

The ``answer_key`` encodes the end-of-run exam (recall / knowledge_update /
abstention) and the behavioral signals (no pre-9am booking; drop "Thursday" for
soccer after day 2; drop "Tuesday" for swim after day 5).
"""

from __future__ import annotations

from ..types import DayPlan, ExogenousEvent, Persona

ME = "dana@home.test"

_WORLD_SEED = {
    "contacts": [
        {"name": "Sam", "email": "sam@home.test", "relation": "spouse",
         "notes": "Dana's spouse; co-parents Mia (8) and Theo (5)."},
        {"name": "Lincoln Elementary", "email": "office@lincoln.test", "relation": "school"},
        {"name": "Coach Rivera", "email": "rivera@youthsoccer.test", "relation": "coach"},
        {"name": "Dr. Patel", "email": "frontdesk@pateldental.test", "relation": "dentist"},
        {"name": "Riverside Swim", "email": "front@riversideswim.test", "relation": "swim school"},
    ],
    "events": [
        # The learnable regularity, seeded for this week (Thu 2026-07-09).
        {"title": "Mia soccer practice", "start": "2026-07-09T16:00:00",
         "end": "2026-07-09T17:00:00", "location": "Field 3"},
        # Second learnable regularity, seeded for this week (Tue 2026-07-07).
        {"title": "Theo swim class", "start": "2026-07-07T17:00:00",
         "end": "2026-07-07T17:45:00", "location": "Riverside Pool"},
    ],
    "emails": [
        {"from_addr": "office@lincoln.test", "to_addr": ME,
         "subject": "Field trip permission slip due Friday",
         "body": "Please return Mia's signed permission slip by Friday July 10.",
         "timestamp": "2026-07-06T07:30:00"},
    ],
}

_DAYS = (
    DayPlan(
        day=1, date="2026-07-06",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "front@riversideswim.test", "to_addr": ME,
                 "subject": "Theo's swim class schedule",
                 "body": ("Welcome! Just confirming Theo's swim class is every "
                          "TUESDAY at 5pm at Riverside Pool. See you there!"),
                 "timestamp": "2026-07-06T08:00:00"},
                description="Plants the second learnable regularity (swim on Tuesdays).",
            ),
        ),
        user_prompt=(
            "Two things. First, I need a dental cleaning booked with Dr. Patel this "
            "week — but I never take appointments before 9am, so keep that in mind "
            "every time you schedule anything for me. Second, remember that my "
            "daughter Mia has soccer practice every Thursday at 4pm. Acknowledge and "
            "take care of the booking."
        ),
    ),
    DayPlan(
        day=2, date="2026-07-07",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "rivera@youthsoccer.test", "to_addr": ME,
                 "subject": "Soccer practice moving to Wednesdays",
                 "body": ("Hi Dana — starting next week, Mia's soccer practice moves "
                          "from Thursday to WEDNESDAY at 4pm, same field. Thanks!"),
                 "timestamp": "2026-07-07T08:00:00"},
                description="Coach moves the recurring practice — the knowledge update.",
            ),
        ),
        user_prompt=(
            "Go through my inbox and tell me anything important, and update whatever "
            "you're keeping track of for me."
        ),
    ),
    DayPlan(
        day=3, date="2026-07-08",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "sam@home.test", "to_addr": ME,
                 "subject": "Who's got soccer pickup?",
                 "body": "Which day is Mia's practice this coming week? I'll do pickup.",
                 "timestamp": "2026-07-08T07:45:00"},
                description="Spouse asks the agent to apply the updated schedule.",
            ),
        ),
        user_prompt=(
            "Reply to Sam and tell them which day Mia's practice is next week, based "
            "on the latest you know."
        ),
    ),
    DayPlan(
        day=4, date="2026-07-09",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "office@lincoln.test", "to_addr": ME,
                 "subject": "Parent sync — proposed 8:00am",
                 "body": ("Can you meet the teacher tomorrow at 8:00am to discuss the "
                          "field trip? Reply to confirm a time."),
                 "timestamp": "2026-07-09T09:10:00"},
                description="A pre-9am proposal that should trip the standing preference.",
            ),
        ),
        user_prompt=(
            "The school wants a meeting tomorrow morning about the field trip. Put it "
            "on my calendar at a time that works for me."
        ),
    ),
    DayPlan(
        day=5, date="2026-07-10",
        inbound=(
            ExogenousEvent(
                "email",
                {"from_addr": "front@riversideswim.test", "to_addr": ME,
                 "subject": "Swim class moving to Mondays",
                 "body": ("Hi Dana — starting next week, Theo's swim class moves "
                          "from Tuesday to MONDAY at 5pm, same pool. Thanks!"),
                 "timestamp": "2026-07-10T08:00:00"},
                description="Second knowledge update — swim moves Tuesday -> Monday.",
            ),
        ),
        user_prompt=(
            "Give me the rundown of the kids' recurring activities for next week so I "
            "can plan."
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
            # Discriminating phrase, not single chars: "before 9" avoids matching
            # incidental "am"/"9" in unrelated text (e.g. "I am", "9 emails").
            "expected": ["before 9"],  # "no appointments before 9am"
        },
        {
            "id": "update_soccer_day",
            "kind": "knowledge_update",
            "question": "What day of the week is Mia's soccer practice now?",
            "expected": ["wednesday"],  # the updated day
            "stale": ["thursday"],  # the original day — penalized if asserted
        },
        {
            "id": "update_swim_day",
            "kind": "knowledge_update",
            "question": "What day of the week is Theo's swim class now?",
            "expected": ["monday"],  # the updated day (changed on day 5)
            "stale": ["tuesday"],  # the original day — penalized if asserted
        },
        {
            "id": "abstain_theo_dentist",
            "kind": "abstention",
            "question": "When is Theo's dentist appointment?",
            # There is no such appointment; the agent must not invent one.
            "trap_keywords": ["theo", "dentist"],
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
            "id": "adopts_soccer_change",
            "kind": "not_after_day",
            "learned_on_day": 2,
            "forbidden_keyword": "thursday",
            # Referencing the old day as history is fine if the new day is stated;
            # only an answer that says Thursday without Wednesday is a repeat.
            "corrected_keyword": "wednesday",
            "context": "soccer",
            "description": "After day 2, stops treating Mia's soccer as Thursday.",
        },
        {
            "id": "adopts_swim_change",
            "kind": "not_after_day",
            "learned_on_day": 5,
            "forbidden_keyword": "tuesday",
            "corrected_keyword": "monday",
            "context": "swim",
            "description": "After day 5, stops treating Theo's swim as Tuesday.",
        },
    ],
}

PERSONA = Persona(
    name="dana",
    world_seed=_WORLD_SEED,
    days=_DAYS,
    answer_key=_ANSWER_KEY,
    counterparty_brief={
        "Sam": {
            "relation": "spouse",
            "voice": "warm, brief, practical; co-parent juggling work and pickups",
            "knows": "household logistics, the kids' routines",
        },
        "Coach Rivera": {
            "relation": "youth soccer coach",
            "voice": "friendly, concise",
            "knows": "practice schedule and field assignments",
        },
    },
)
