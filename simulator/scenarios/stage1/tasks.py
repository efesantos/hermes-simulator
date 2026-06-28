"""The Stage-1 task definitions.

Each :class:`Stage1Task` pairs a seeded world slice + a single-shot prompt with an
``expected_state`` graded by :mod:`simulator.grading.deterministic`. Tasks are
authored to be crisp: a correct trajectory reaches the end-state, a wrong one
does not, and grading needs no judgment. Categories group them for reporting.

Standing conventions across tasks: the user is ``dana@home.test``; calendar
times are naive ISO local; "the agent must not invent" cases assert absence.
"""

from __future__ import annotations

from ..types import Stage1Task

ME = "dana@home.test"

_CONTACTS = {
    "contacts": [
        {"name": "Sam", "email": "sam@home.test", "relation": "spouse"},
        {"name": "Coach Rivera", "email": "rivera@youthsoccer.test", "relation": "coach"},
        {"name": "Dr. Patel", "email": "frontdesk@pateldental.test", "relation": "dentist"},
        {"name": "Lincoln Elementary", "email": "office@lincoln.test", "relation": "school"},
    ]
}


STAGE1_TASKS: list[Stage1Task] = [
    Stage1Task(
        id="book_after_9am",
        category="booking",
        prompt=(
            "Book a 30-minute dental cleaning with Dr. Patel on 2026-07-15. I never "
            "take appointments before 9am, so schedule it at or after 9:00."
        ),
        world_seed=_CONTACTS,
        expected_state={
            "events_present": [
                {"day": "2026-07-15", "start_time_at_or_after": "09:00"},
            ],
            "events_absent": [
                {"day": "2026-07-15", "start_time_before": "09:00"},
            ],
        },
    ),
    Stage1Task(
        id="afternoon_focus_block",
        category="booking",
        prompt=(
            "Put a 1-hour focus block on my calendar for 2026-07-21 in the afternoon "
            "(start at or after 1:00pm)."
        ),
        world_seed={},
        expected_state={
            "events_present": [
                {"day": "2026-07-21", "start_time_at_or_after": "13:00"},
            ],
        },
    ),
    Stage1Task(
        id="avoid_double_book",
        category="conflict",
        prompt=(
            "Add a 30-minute contractor call on 2026-07-16. I have Mia's soccer "
            "pickup from 4:00 to 5:00pm that day, so do not schedule the call then."
        ),
        world_seed={
            "events": [
                {"title": "Mia soccer pickup", "start": "2026-07-16T16:00:00",
                 "end": "2026-07-16T17:00:00"},
            ],
        },
        expected_state={
            "events_present": [{"title_contains": "contractor", "day": "2026-07-16"}],
            "events_absent": [
                {"title_contains": "contractor",
                 "overlaps": {"start": "2026-07-16T16:00:00", "end": "2026-07-16T17:00:00"}},
            ],
        },
    ),
    Stage1Task(
        id="flag_conflict_to_spouse",
        category="conflict",
        prompt=(
            "I want a dentist appointment on 2026-07-16 from 4:00 to 4:30pm. If that "
            "clashes with anything already on my calendar, do NOT book it — instead "
            "email Sam (sam@home.test) to flag the clash. Mention what it clashes with."
        ),
        world_seed={
            "events": [
                {"title": "Mia soccer pickup", "start": "2026-07-16T16:00:00",
                 "end": "2026-07-16T17:00:00"},
            ],
            **_CONTACTS,
        },
        expected_state={
            "emails_present": [{"to_contains": "sam@home.test", "body_contains": "soccer"}],
            "events_absent": [
                {"title_contains": "dentist",
                 "overlaps": {"start": "2026-07-16T16:00:00", "end": "2026-07-16T17:00:00"}},
            ],
        },
    ),
    Stage1Task(
        id="inbox_confirmation_to_calendar",
        category="cross_domain",
        prompt=(
            "Check my inbox for any appointment confirmation and add the confirmed "
            "appointment to my calendar."
        ),
        world_seed={
            "emails": [
                {"from_addr": "frontdesk@pateldental.test", "to_addr": ME,
                 "subject": "Your cleaning is confirmed",
                 "body": "Confirmed: dental cleaning on 2026-07-20 at 10:00 for 30 minutes.",
                 "timestamp": "2026-07-13T09:00:00"},
            ],
        },
        expected_state={
            "events_present": [{"day": "2026-07-20", "start": "2026-07-20T10:00:00"}],
        },
    ),
    Stage1Task(
        id="lookup_contact_then_email",
        category="coordination",
        prompt=(
            "Email Coach Rivera to ask which field Saturday's soccer game is on. Look "
            "up the address from my contacts."
        ),
        world_seed=_CONTACTS,
        expected_state={
            "emails_present": [{"to_contains": "rivera@youthsoccer.test"}],
        },
    ),
    Stage1Task(
        id="reschedule_meeting",
        category="calendar",
        prompt=(
            "Move my 8:00am team sync on 2026-07-17 to 9:30am the same day."
        ),
        world_seed={
            "events": [
                {"title": "Team sync", "start": "2026-07-17T08:00:00",
                 "end": "2026-07-17T08:30:00"},
            ],
        },
        expected_state={
            "events_present": [{"title_contains": "team sync", "start": "2026-07-17T09:30:00"}],
            "events_absent": [{"title_contains": "team sync", "start_time_before": "09:00"}],
        },
    ),
    Stage1Task(
        id="forward_notice_and_ask",
        category="coordination",
        prompt=(
            "The school sent an early-dismissal notice. Email Sam (sam@home.test) the "
            "gist of it and ask if he can do pickup."
        ),
        world_seed={
            "emails": [
                {"from_addr": "office@lincoln.test", "to_addr": ME,
                 "subject": "Early dismissal Friday",
                 "body": "School dismisses at 12:30pm this Friday, 2026-07-17.",
                 "timestamp": "2026-07-13T08:00:00"},
            ],
            **_CONTACTS,
        },
        expected_state={
            "emails_present": [{"to_contains": "sam@home.test", "body_contains": "pickup"}],
        },
    ),
    Stage1Task(
        id="accept_invite_two_steps",
        category="cross_domain",
        prompt=(
            "Accept the lunch invite in my inbox: add it to my calendar AND reply to "
            "the sender to confirm."
        ),
        world_seed={
            "emails": [
                {"from_addr": "jordan@work.test", "to_addr": ME,
                 "subject": "Lunch Thursday?",
                 "body": "Lunch on 2026-07-19 from 12:00 to 13:00? Reply to confirm.",
                 "timestamp": "2026-07-13T11:00:00"},
            ],
        },
        expected_state={
            "events_present": [{"day": "2026-07-19", "start": "2026-07-19T12:00:00"}],
            "emails_present": [{"to_contains": "jordan@work.test"}],
        },
    ),
    Stage1Task(
        id="do_not_invent_appointment",
        category="abstention",
        prompt=(
            "Do I have a doctor's appointment on 2026-07-18? If I do, add a reminder "
            "for it. If I don't, do not put anything on my calendar."
        ),
        world_seed={},  # nothing on 2026-07-18
        expected_state={
            "events_absent": [{"day": "2026-07-18"}],
        },
    ),
]
