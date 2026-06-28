"""The mock world: email/calendar/contacts as MCP servers over one backing store.

The agent only ever touches this world through MCP tools (the ``*_server.py``
modules). The grader reads the same store **out-of-band** via
:class:`~simulator.world.state.WorldState` — never through an agent-reachable
tool — which is what closes the reward-hack / answer-leak hole (KTD2).
"""
