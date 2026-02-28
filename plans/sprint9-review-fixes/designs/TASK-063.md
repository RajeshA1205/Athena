# TASK-063: Fix LatentSpace Broadcast Missing Unregistered Agents

## Summary

`LatentSpace.broadcast()` (via `send()`) iterates `list(self._queues.keys())` to deliver messages to all agents. But `_queues` is a `defaultdict` — entries are created lazily on first access. Any agent that has never received a direct message has no queue entry and is silently excluded from broadcasts. Since the coordinator sends the final-decision broadcast before any agent-to-agent messaging occurs, in practice no specialist agent receives it. Fix: add an explicit `register_agent()` method to `LatentSpace` and call it from `CoordinatorAgent.initialize_communication()`.

## Current State

**File:** `communication/latent_space.py`

**Lines 206-208 — _queues is a defaultdict:**
```python
        self._queues: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=message_queue_size)
        )
```

**Lines 257-262 — send() broadcast path iterates existing keys only:**
```python
                if message.recipient == "*":
                    if self.broadcast_enabled:
                        self._broadcast_queue.append(latent_msg)
                        for agent_name in list(self._queues.keys()):
                            if agent_name != message.sender:
                                self._queues[agent_name].append(latent_msg)
```

**File:** `agents/coordinator.py`

**Lines 115-138 — initialize_communication() registers agents in the router but not in LatentSpace:**
```python
    async def initialize_communication(self, latent_space: "LatentSpace") -> None:
        """Initialize LatentMAS communication infrastructure."""
        from communication.router import MessageRouter, MessagePriority
        from communication.encoder import AgentStateEncoder
        from communication.decoder import AgentStateDecoder
        ...
        # Register all known agents
        for agent_name in self.agents:
            self.router.register_agent(agent_name, {"role": self.agents[agent_name].role})
```

No call to `latent_space.register_agent(...)`.

## Proposed Change

### 1. Add `register_agent()` to `LatentSpace` (communication/latent_space.py)

Add after the `reset_stats` method (around line 492):

```python
    def register_agent(self, agent_name: str) -> None:
        """
        Pre-register an agent so it receives broadcasts even before
        receiving a direct message.

        Creates the agent's message queue eagerly, ensuring the agent
        appears in _queues.keys() during broadcast delivery.

        Args:
            agent_name: Unique agent identifier.
        """
        if agent_name not in self._queues:
            self._queues[agent_name] = deque(maxlen=self.message_queue_size)
            self.logger.debug("Registered agent queue: %s", agent_name)
```

### 2. Call `register_agent()` in coordinator's `initialize_communication()` (agents/coordinator.py)

After the existing `self.router.register_agent(...)` loop, add:

```python
        # Pre-register agents in LatentSpace so broadcasts reach them
        for agent_name in self.agents:
            latent_space.register_agent(agent_name)
```

Store `latent_space` reference on `self` so it's accessible:

```python
        self.latent_space = latent_space
```

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `communication/latent_space.py` | After line ~492 | Add `register_agent(agent_name)` method |
| `agents/coordinator.py` | ~line 135 (inside `initialize_communication`) | Add loop calling `latent_space.register_agent(agent_name)` for each agent; store `self.latent_space = latent_space` |

## Acceptance Criteria

- After `initialize_communication()`, `latent_space._queues` contains an entry for every registered specialist agent.
- A broadcast sent before any direct message delivers to all registered agents.
- `register_agent()` is idempotent — calling it twice for the same agent does not create a second queue or lose existing messages.
- 173 tests pass, 4 skipped.

## Edge Cases & Risks

1. **Idempotency**: The `if agent_name not in self._queues` guard ensures the deque is only created once. Existing queues (created by earlier `receive()` or `send()` calls) are preserved.
2. **defaultdict still works for non-registered agents**: Any agent not registered but that receives a direct message will still get its queue auto-created by the defaultdict on the `self._queues[message.recipient].append(...)` call. `register_agent` only ensures broadcast coverage for agents that have not yet been messaged.
3. **Thread safety**: `register_agent` is a synchronous method that modifies `_queues` without holding `self._lock`. This is safe as long as it is called at startup before the async event loop processes messages. If called concurrently, add `async def register_agent` with `async with self._lock`.

## Test Notes

- Verify: after `initialize_communication()`, `len(latent_space._queues) == len(coordinator.agents)`.
- Verify: a broadcast message is present in every agent's queue after `latent_space.broadcast(msg)`.
- Run `python3 -m pytest tests/ -q` — expect 173 passed, 4 skipped.
