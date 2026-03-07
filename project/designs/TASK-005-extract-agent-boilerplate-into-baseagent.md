# TASK-005: Extract duplicated think/act patterns into BaseAgent helpers

## Problem

All 5 agent `think()` and `act()` methods contain ~95 lines of identical boilerplate: (a) memory retrieval with try/except, (b) LatentMAS `router.receive()` with try/except, (c) memory store of result/failure in `act()` with try/except, and (d) LatentMAS `router.send()` to coordinator in `act()` with try/except. This violates DRY, makes maintenance error-prone, and increases risk of inconsistency when patterns evolve.

## Files to change

| File | Lines | Change |
|------|-------|--------|
| `core/base_agent.py` | After line 286 | Add `_retrieve_memory_context()`, `_receive_latent_messages()`, `_store_to_memory()`, `_send_via_latent()` helpers |
| `agents/market_analyst.py` | 126-144 (think), 262-274 (act) | Replace inline memory/latent code with helper calls |
| `agents/risk_manager.py` | 106-126 (think), 246-270 (act) | Same |
| `agents/strategy_agent.py` | 131-150 (think), 256-281 (act) | Same |
| `agents/execution_agent.py` | 104-125 (think), 242-267 (act) | Same |
| `agents/coordinator.py` | 156-177 (think), 302-347 (act) | Same |

## Approach

1. Add to `BaseAgent`:
   ```python
   async def _retrieve_memory_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
       """Retrieve relevant memory context, returning [] on failure."""
       if self.memory is None:
           return []
       try:
           return await self.memory.retrieve(query=query, top_k=top_k)
       except Exception as e:
           self.logger.warning("Memory retrieve failed: %s", e)
           return []

   async def _receive_latent_messages(self) -> List[Any]:
       """Receive pending LatentMAS messages, returning [] on failure."""
       if not hasattr(self, 'router') or self.router is None:
           return []
       try:
           return await self.router.receive(receiver_id=self.name, decode_mode="structured")
       except Exception as e:
           self.logger.warning("LatentMAS receive failed: %s", e)
           return []

   async def _store_to_memory(self, content: Any, success: bool) -> None:
       """Store result to memory with agent metadata."""
       ...

   async def _send_via_latent(self, message: Any, receiver_id: str = "coordinator", priority=None) -> None:
       """Send message via LatentMAS router."""
       ...
   ```

2. Each agent's `think()` replaces its inline memory+latent blocks with:
   ```python
   memory_context = await self._retrieve_memory_context(query=f"...")
   latent_messages = await self._receive_latent_messages()
   ```

3. Each agent's `act()` replaces inline memory store + latent send blocks with:
   ```python
   await self._send_via_latent(result)
   await self._store_to_memory(content={...}, success=True)
   ```

4. The `router` attribute needs to be declared on `BaseAgent.__init__` (currently it's only set on subclasses). Add `self.router: Optional[Any] = None` alongside the existing `self.communication` attribute.

5. Run full test suite after each agent migration to catch regressions.

## Edge cases / risks

- `router` is not currently a `BaseAgent` attribute. Adding it requires a default `None` parameter or a post-init assignment. Choose post-init `self.router = None` to avoid changing the constructor signature and breaking existing callers.
- Coordinator stores additional metadata (`operation: "coordination_summary"`, `agents_queried` list). The `_store_to_memory` helper should accept an optional `metadata` override dict.
- Some agents build the memory query differently (e.g., risk_manager uses `portfolio.get("total_value")`). The query string must remain caller-specified -- the helper only wraps the try/except.

## Acceptance criteria

- [ ] Zero duplicated memory/latent try/except blocks across the 5 agents.
- [ ] All agents produce identical behavior (same log messages, same memory content, same LatentMAS messages).
- [ ] `pytest tests/ -q` remains green.
- [ ] Each helper has its own unit test.
