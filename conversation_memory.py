"""
Conversation Memory — Persistent JSON-backed conversation history.
Survives sessions: each session is tagged, and the last N turns are
loaded into context at startup.
"""

import json
import os
from datetime import datetime


class ConversationMemory:
    """
    Persists conversation turns to disk so context carries across restarts.

    Storage format (checkpoints/conversation_history.json):
    {
      "sessions": [
        {
          "id": "20240315_143022",
          "turns": [
            {"session": "...", "timestamp": "...", "question": "...", "answer": "..."},
            ...
          ]
        },
        ...
      ]
    }
    """

    MAX_SESSIONS = 10    # keep last N sessions on disk
    RUNTIME_LIMIT = 40  # max turns held in RAM during a session

    def __init__(self, path: str = "checkpoints/conversation_history.json", load_turns: int = 10):
        self.path = path
        self.load_turns = load_turns
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._history: list[dict] = []   # all turns in RAM (current + loaded)
        self._session_turns: list[dict] = []  # only this session's turns
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self):
        """Load the last `load_turns` turns from disk for context."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_turns: list[dict] = []
            for session in data.get("sessions", []):
                all_turns.extend(session.get("turns", []))
            self._history = all_turns[-self.load_turns:]
        except Exception:
            self._history = []

    def _save(self):
        """Append current session turns to the JSON file."""
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)

        # Load existing data
        data: dict = {"sessions": []}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        sessions: list[dict] = data.get("sessions", [])

        # Find or create the current session entry
        current = next((s for s in sessions if s["id"] == self.session_id), None)
        if current is None:
            current = {"id": self.session_id, "turns": []}
            sessions.append(current)

        current["turns"] = self._session_turns

        # Trim old sessions
        data["sessions"] = sessions[-self.MAX_SESSIONS:]

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Public API ─────────────────────────────────────────────────────

    def add_turn(self, question: str, answer: str):
        turn = {
            "session": self.session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
        }
        self._history.append(turn)
        self._session_turns.append(turn)

        # Trim RAM copy
        if len(self._history) > self.RUNTIME_LIMIT:
            self._history = self._history[-self.RUNTIME_LIMIT:]

        self._save()

    def get_context(self, n_turns: int = 5) -> list[dict]:
        """Return the last n_turns for use as model context."""
        return self._history[-n_turns:] if self._history else []

    def clear_session(self):
        """Forget this session's turns (keeps prior sessions on disk)."""
        self._session_turns.clear()
        self._history = [t for t in self._history if t.get("session") != self.session_id]
        self._save()

    def clear_all(self):
        """Delete all history, everywhere."""
        self._history.clear()
        self._session_turns.clear()
        if os.path.exists(self.path):
            os.remove(self.path)

    def format_recent(self, n_turns: int = 3) -> str:
        """Human-readable recent history for display."""
        recent = self.get_context(n_turns)
        if not recent:
            return "(no history yet)"
        lines = []
        for t in recent:
            ts = t.get("timestamp", "")[:16]
            lines.append(f"  [{ts}] You: {t['question']}")
            preview = t["answer"][:120].rstrip()
            if len(t["answer"]) > 120:
                preview += "..."
            lines.append(f"           AI : {preview}")
        return "\n".join(lines)

    @property
    def total_turns(self) -> int:
        return len(self._history)
