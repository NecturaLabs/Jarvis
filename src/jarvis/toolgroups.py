"""Tool groups: every operation belongs to one group, which says what its tools are for.

The system prompt's toolbox lists the groups, so the model knows what it can do and plans from
that. With `tools = "routed"` the model is first offered only the core tools; `use_tools` adds a
group's tools for the rest of the request (`jarvis.router`). Membership is by operation name, one
list in one place: a test fails for an operation that no group names, so a new tool cannot be
left out. A group's summary says what its tools do, for the model to read; it is never matched
against what the owner says.
"""

from dataclasses import dataclass

USE_TOOLS = "use_tools"
CORE_NAME = "core"


@dataclass(frozen=True, slots=True)
class ToolGroup:
    """A named set of operations with a one-line summary of what they do."""

    name: str
    summary: str
    tools: tuple[str, ...]


# What every request may need, so it is always offered: the ways to answer and to go on (answer
# with no tool, stay silent, hand a spoken request to the open chat, keep the chat's todo list, ask
# the owner to choose, think harder, move to coding mode, load a group). `switch_model` must
# be the first and only call of its reply, so a detour through `use_tools` would only cost a call.
CORE = ToolGroup(
    CORE_NAME,
    "How you answer and go on: answer without a tool, stay quiet, hand off to the chat, keep "
    "todos, ask the owner, think harder, switch modes, load more tools.",
    (
        "no_action",
        "stay_quiet",
        "chat_hand_off",
        "todo_write",
        "ask_owner",
        "think_harder",
        "switch_model",
        USE_TOOLS,
    ),
)

GROUPS = (
    ToolGroup(
        "music",
        "Spotify: play music, pause, resume, skip, seek, volume, shuffle, repeat, the queue, "
        "likes, the library and what is playing.",
        (
            "spotify_play",
            "spotify_pause",
            "spotify_resume",
            "spotify_seek",
            "spotify_next",
            "spotify_previous",
            "spotify_set_volume",
            "spotify_change_volume",
            "spotify_shuffle",
            "spotify_repeat",
            "spotify_queue",
            "spotify_like",
            "spotify_library",
            "spotify_now_playing",
        ),
    ),
    ToolGroup(
        "sound",
        "The PC's sound: its volume, play and pause in other media players, hearing what the PC "
        "plays, and singing a song.",
        ("pc_volume", "pc_media", "pc_audio_listen", "sing"),
    ),
    ToolGroup(
        "screen",
        "Windows and the screen: list, focus, move and close windows, take screenshots, "
        "and watch for screen conditions, window freezes or process trouble.",
        (
            "pc_windows",
            "pc_window",
            "pc_window_close",
            "pc_screenshot",
            "pc_watch",
            "pc_watches",
            "pc_watch_stop",
        ),
    ),
    ToolGroup(
        "apps",
        "Programs: open one where the owner says, list or find running ones and stop one.",
        ("open_app", "find_processes", "stop_process"),
    ),
    ToolGroup(
        "system",
        "The PC itself: its load, temperatures, memory, disks and specs, and a shell command "
        "where the owner allows it.",
        ("pc_stats", "pc_shell"),
    ),
    ToolGroup(
        "files",
        "The owner's files: list a folder, read a file, write or change a file.",
        ("list_dir", "read_file", "write_file"),
    ),
    ToolGroup(
        "everyday",
        "The time or date anywhere, the weather, and exact arithmetic.",
        ("time_now", "weather", "calculate"),
    ),
    ToolGroup(
        "web",
        "The web: search it and Reddit, read pages and threads, find and show pictures.",
        ("web_search", "web_read", "reddit_search", "reddit_read", "image_search", "show_image"),
    ),
    ToolGroup(
        "browser",
        "A headless browser for pages that need scripts or clicking: open a page, click or type "
        "in it, show a screenshot of it.",
        ("browser_open", "browser_act", "browser_screenshot"),
    ),
    ToolGroup(
        "schedule",
        "Reminders and requests for later: set one, list them, cancel one.",
        ("schedule_add", "schedule_list", "schedule_cancel"),
    ),
    ToolGroup(
        "guides",
        "Written instructions for a kind of task: Jarvis's own skills and Omarchy's guides.",
        ("read_skill", "os_skill"),
    ),
    ToolGroup(
        "jarvis",
        "Jarvis itself: its version, model, voice, settings, limits and config files.",
        ("about_jarvis", "read_jarvis_file"),
    ),
)

_GROUP_OF = {name: group.name for group in (CORE, *GROUPS) for name in group.tools}


def group_name(op_name: str) -> str | None:
    """Return the name of the group an operation belongs to, or None when no group names it."""
    return _GROUP_OF.get(op_name)


def declared(name: str) -> ToolGroup | None:
    """Return the declared group of that name, the core one included."""
    return next((g for g in (CORE, *GROUPS) if g.name == name), None)
