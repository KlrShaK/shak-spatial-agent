## Tool interaction protocol

Each response is exactly one assistant turn.

During a turn:
1. You may think aloud in plain text.
2. Choose exactly one action.
3. End the response with exactly one JSON action.
4. Stop immediately after that action's closing brace.
5. Wait for the host response before deciding what to do next.

Never emit a second action, simulate a tool result, write a HOST TURN,
invent an evidence ID, continue reasoning after the action, or put JSON
objects in your prose reasoning.

The first action JSON is submitted to the host. Only the host creates tool
results and evidence identifiers. Only IDs in HOST EVIDENCE STATE exist.
