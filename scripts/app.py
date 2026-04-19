"""
app.py — Gradio Chat UI for Bible RAG (Claude backend)
=======================================================
Left sidebar (collapsible): chat history sessions.
Center: chat with live tool-call logs streamed before the final answer.
Chat history is auto-saved to logs/chat_history.json.

Run:
    python scripts/app.py
Stop:
    Ctrl+C in terminal (history is auto-saved after every response
    and on shutdown).
"""

import os
import sys
import json
import signal
import atexit
import threading
import time
from pathlib import Path
from datetime import datetime

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from claude_bible_rag import bible_query
from fhl_tools import ALL_TOOLS

HISTORY_FILE = Path(__file__).resolve().parent.parent / "logs" / "chat_history.json"

CSS = """
#chatbot { height: 75vh !important; }
#chatbot .message { font-size: 13px !important; }
#chatbot .message p, #chatbot .message li { font-size: 13px !important; }
#chatbot .message code { font-size: 12px !important; }
#chatbot .message pre { font-size: 12px !important; }
"""


# ─── Persistent chat session manager ────────────────────────────────────────

class ChatSessionManager:
    def __init__(self):
        self.sessions: list[dict] = []
        self.current_idx: int = -1
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                self.sessions = data.get("sessions", [])
                self.current_idx = data.get("current_idx", -1)
                if self.current_idx >= len(self.sessions):
                    self.current_idx = len(self.sessions) - 1
                print(f"[OK] Loaded {len(self.sessions)} chat sessions from {HISTORY_FILE}")
            except (json.JSONDecodeError, KeyError):
                self.sessions = []
                self.current_idx = -1
        if not self.sessions:
            self.new_session()

    def save(self):
        with self._lock:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {"sessions": self.sessions, "current_idx": self.current_idx}
            HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def new_session(self) -> int:
        session = {
            "title": "New chat",
            "messages": [],
            "api_history": [],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self.sessions.append(session)
        self.current_idx = len(self.sessions) - 1
        self.save()
        return self.current_idx

    def get_current(self) -> dict | None:
        if 0 <= self.current_idx < len(self.sessions):
            return self.sessions[self.current_idx]
        return None

    def get_current_messages(self) -> list:
        s = self.get_current()
        return s["messages"] if s else []

    def get_current_api_history(self) -> list:
        s = self.get_current()
        return s.get("api_history", []) if s else []

    def append_turn(self, user_text: str, assistant_display: str, assistant_clean: str):
        s = self.get_current()
        if not s:
            return
        s["messages"].append({"role": "user", "content": user_text})
        s["messages"].append({"role": "assistant", "content": assistant_display})
        s["api_history"].append({"role": "user", "content": user_text})
        s["api_history"].append({"role": "assistant", "content": assistant_clean})
        if s["title"] == "New chat":
            s["title"] = user_text[:30]
        self.save()

    def update_last_assistant(self, display_content: str):
        s = self.get_current()
        if s and s["messages"] and s["messages"][-1]["role"] == "assistant":
            s["messages"][-1]["content"] = display_content

    def get_sidebar_labels(self) -> list[str]:
        labels = []
        for i, s in enumerate(self.sessions):
            prefix = ">> " if i == self.current_idx else "   "
            labels.append(f"{prefix}{s['created']}  {s['title']}")
        return labels

    def switch_to(self, idx: int):
        if 0 <= idx < len(self.sessions):
            self.current_idx = idx
            self.save()

    def delete_current(self):
        if 0 <= self.current_idx < len(self.sessions):
            self.sessions.pop(self.current_idx)
            if self.sessions:
                self.current_idx = min(self.current_idx, len(self.sessions) - 1)
            else:
                self.current_idx = -1
            self.save()


manager = ChatSessionManager()


def _on_exit():
    manager.save()
    print(f"\n[OK] Chat history saved to {HISTORY_FILE}")

atexit.register(_on_exit)
signal.signal(signal.SIGINT, lambda *_: (manager.save(), sys.exit(0)))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_log_block(log_lines: list[str]) -> str:
    if not log_lines:
        return ""
    escaped = "\n".join(log_lines)
    tool_count = len([l for l in log_lines if "🔧" in l])
    return f"**🔧 Tool calls** ({tool_count} calls)\n```\n{escaped}\n```"


def _sidebar_choices():
    labels = manager.get_sidebar_labels()
    idx = manager.current_idx
    return gr.update(choices=labels, value=labels[idx] if 0 <= idx < len(labels) else None)


# ─── Gradio callbacks ───────────────────────────────────────────────────────

def respond(message: dict | str):
    text = message.get("text", "") if isinstance(message, dict) else message
    if not text.strip():
        yield manager.get_current_messages(), _sidebar_choices(), gr.update()
        return

    log_lines: list[str] = []
    log_lock = threading.Lock()
    result_holder: list[str | None] = [None]
    error_holder: list[str | None] = [None]
    done_event = threading.Event()

    def on_log(line: str):
        with log_lock:
            log_lines.append(line)

    def run_query():
        try:
            api_history = manager.get_current_api_history()
            answer = bible_query(
                user_question=text,
                tools=ALL_TOOLS,
                history=api_history if api_history else None,
                verbose=False,
                log_callback=on_log,
            )
            result_holder[0] = answer
        except Exception as e:
            error_holder[0] = str(e)
        finally:
            done_event.set()

    display_msgs = manager.get_current_messages().copy()
    display_msgs.append({"role": "user", "content": text})
    display_msgs.append({"role": "assistant", "content": "⏳ Calling Claude..."})
    yield display_msgs, _sidebar_choices(), None

    worker = threading.Thread(target=run_query, daemon=True)
    worker.start()

    last_log_count = 0
    while not done_event.is_set():
        time.sleep(0.3)
        with log_lock:
            current_count = len(log_lines)
            if current_count > last_log_count:
                last_log_count = current_count
                live_log = _format_log_block(list(log_lines))
                display_msgs[-1] = {"role": "assistant", "content": live_log}
                yield display_msgs, _sidebar_choices(), gr.update()

    worker.join()

    if error_holder[0]:
        assistant_display = f"❌ Error: {error_holder[0]}"
        assistant_clean = assistant_display
    else:
        answer = result_holder[0] or ""
        tool_log = _format_log_block(log_lines)
        if tool_log:
            assistant_display = tool_log + "\n\n---\n\n" + answer
        else:
            assistant_display = answer
        assistant_clean = answer

    manager.append_turn(text, assistant_display, assistant_clean)

    yield manager.get_current_messages(), _sidebar_choices(), gr.update()


def new_chat():
    manager.new_session()
    return manager.get_current_messages(), _sidebar_choices()


def switch_chat(selection: str | None):
    if selection is None:
        return manager.get_current_messages(), _sidebar_choices()
    labels = manager.get_sidebar_labels()
    try:
        idx = labels.index(selection)
    except ValueError:
        return manager.get_current_messages(), _sidebar_choices()
    manager.switch_to(idx)
    return manager.get_current_messages(), _sidebar_choices()


def delete_chat():
    manager.delete_current()
    if not manager.sessions:
        manager.new_session()
    return manager.get_current_messages(), _sidebar_choices()


# ─── UI layout ───────────────────────────────────────────────────────────────

with gr.Blocks(title="信望愛 AI 聖經助手") as demo:
    gr.Markdown("## 📖 信望愛 AI 聖經助手 — Claude × FHL Bible Tools")

    with gr.Sidebar(label="Chat History", open=True):
        new_chat_btn = gr.Button("+ New Chat", variant="primary", size="sm")
        history_list = gr.Radio(
            choices=manager.get_sidebar_labels(),
            value=manager.get_sidebar_labels()[manager.current_idx] if manager.sessions else None,
            label="",
            show_label=False,
        )
        delete_btn = gr.Button("🗑️ Delete Chat", variant="secondary", size="sm")

    chatbot = gr.Chatbot(
        value=manager.get_current_messages(),
        label="",
        elem_id="chatbot",
    )
    msg_input = gr.MultimodalTextbox(
        placeholder="問一個聖經問題... (e.g. 約翰福音3:16是什麼意思？)",
        show_label=False,
        sources=[],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input],
        outputs=[chatbot, history_list, msg_input],
    )

    new_chat_btn.click(fn=new_chat, outputs=[chatbot, history_list])
    delete_btn.click(fn=delete_chat, outputs=[chatbot, history_list])
    history_list.change(fn=switch_chat, inputs=[history_list], outputs=[chatbot, history_list])


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS, theme=gr.themes.Soft())
