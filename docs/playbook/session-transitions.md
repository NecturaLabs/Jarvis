# Session transitions

What Jarvis does when the desktop session changes under it, what was seen on the maintainer's PC, and
how to check each one. The code paths have fake-environment tests (`tests/test_voice.py`,
`tests/test_control.py`, `tests/test_audio.py`, `tests/test_model.py`, `tests/test_transitions.py`);
`tests/test_transitions_local.py` checks the real pieces those fakes stand in for
(`uv run pytest -m local tests/test_transitions_local.py`, with the owner: one test opens the microphone for
two seconds and one pauses the running Jarvis and resumes it). The [test matrix](#test-matrix) at the end
says which command and transition is checked how, and what is not.

Read the outcome of each step in the voice log: `jarvis logs`, or
`journalctl --user -u jarvis.service -u jarvis-strata.service -f`.

## Desktop session start and stop

Both units are `PartOf=` and `WantedBy=graphical-session.target`: they start when you log in and stop
when the session ends. Stopping Strata frees the model's RAM and VRAM.

- Seen on 2026-11-21: the units stopped at 15:38 when the PC was shut down; it booted at 15:40, and the
  units started at 15:42, once the session started after login.

## Lock screen

The Omarchy shell draws the lock screen itself (`WlSessionLock`) and does not tell logind, so
`loginctl show-session` keeps `LockedHint=no` while locked. Jarvis is a user service and nothing in it
stops at a lock: by default it keeps listening, and anyone at the locked PC can talk to it.

With `audio.listen_when_locked = false`, Jarvis asks the shell (`omarchy-shell lock isLocked`, a few
milliseconds) before it takes a request: after the wake word, once a follow-up is heard, for a request
spoken over a reply, and for a PC-audio request. While locked it takes none; the wake word still stops its
speech. The log says `the wake word was ignored: the screen is locked`. An answer that cannot be read
counts as locked.

- Check: `uv run pytest -m local tests/test_transitions_local.py -k unlocked` at an unlocked screen
  (asks the shell; locks nothing). Seen passing on 2026-11-22.
- With the owner: set the key to `false`, `jarvis restart`, lock, say the wake word, and look for the log
  line; unlock and it answers again. Not run yet.

## PipeWire restart, microphone unplugged

`pw-record` ends when PipeWire restarts. Not verified on this PC: whether it ends or carries on silent
when its microphone is unplugged (WirePlumber may move it to another source). Jarvis handles both.

- Seen on 2026-11-21 at 15:13 and 15:15: PipeWire restarted, `pw-record` ended, and `jarvis run` exited
  with `AudioError: the microphone stream ended`; systemd started it again 5 s later.

Now the loop opens the microphone again: a capture that ended, or that sent nothing for 5 s
(`MIC_STALL_S`), is restarted after 1 s, then twice as long after each try that brings no frame, up to 30 s.
The log says `the microphone stopped (...); opening it again`. Meanwhile `jarvis status` says
`unavailable`; a pause still wins and shows `paused`. A request the lost microphone cuts short ends without
an apology. The stall time counts only while the loop waits for a frame, not while it answers.

- With the owner: `systemctl --user restart pipewire` while Jarvis runs, then say the wake word. Unplug
  the microphone, wait 10 s, plug it back, say the wake word. Not run yet.
- A microphone set by serial (`audio.source`) gets a new serial when it is plugged back in, so the
  reopened capture falls back to PipeWire's default. Set it by node name to keep it.

## Suspend and resume

Linux's monotonic clock stands still while the PC sleeps and its boot clock does not. When the gap
between them grows by 2 s or more between two frames, the loop takes it as a suspend: it drops the
frames from before it, resets the wake word, and asks the model server for its state: a model whose load
failed is loaded again (not while paused); an unloaded one stays unloaded, since `jarvis unload` before a
game or the idle unload put it there. The log says `back from sleep after N s: listening afresh`, then
`back from sleep: the model is <state>`. A stall check cannot fire during the sleep itself, since it
times with the monotonic clock. A spoken yes awaited when the PC sleeps counts as a no, so a yes said after waking
never answers a question asked before it; the confirm key's wait runs on the boot clock, so a sleep uses it up.

- Not seen yet: no kept boot of this PC has suspended (`journalctl -k | grep 'PM: suspend'`).
- With the owner: suspend, wake, say the wake word; look for the log line and check
  `jarvis status --json` shows the model `loaded`.

## Model drive missing

When the model's drive is mounted `nofail`, the desktop starts without it. `jarvis-strata.service` has
`ConditionPathExists=` on the model file, so systemd skips Strata. Nothing is left to wait for.

Jarvis then answers a request at once with "My model's file is missing, so I can't answer. Is its drive
mounted?" instead of waiting 30 s for the server. `jarvis status` shows the model `missing`, and
`jarvis doctor` fails the model line with "mount its drive, or set --model-file". Once the drive is
back, `jarvis start` starts Strata.

- Check: `uv run pytest -m local tests/test_transitions_local.py -k strata` (one status GET). Seen
  passing on 2026-11-22, with the drive mounted.
- Not run: booting without the drive.

## Test matrix

Each command in `jarvis --help` crossed with each transition. A cell says how far it is checked:

- **F**: a fake-environment test (runs in CI, no hardware).
- **L**: a `local` test in `tests/test_transitions_local.py` that checks the real piece a fake stands in for; it may not exercise the transition itself. A note says when it has not been run.
- **O**: the real transition was seen on this PC. **O\*** means the transition was seen, but with the code
  before the current handling, so the handling itself is not yet seen.
- **U**: not tested at all; the lists below say what is missing.
- **-**: the command does not read or change what the transition changes, so there is nothing to test.
  `setup`, `spotify-login`, `logs`, `wake`, `consent` and `pc-audio` are `-` in every column: they
  write the config or show a log.

"Strata down" is the model server stopped, crashed or restarting with its model file present.

| Command | Mic lost | Suspend | Lock | Drive missing | Session start, stop | Strata down |
|---|---|---|---|---|---|---|
| `run` | F, L, O\* | F, U | F, L | F, L, U | F, L, O | F |
| `ask` | - | - | - | F | - | F |
| `chat` | - | - | - | F | - | F |
| `status` | F | F | - | F | F, L | F |
| `pause` | F | F | - | F | F, L | F |
| `resume` | F | - | - | F | F, L | F |
| `hush` | F | - | - | - | F, L | - |
| `confirm` | F | F | U | - | F | - |
| `unload` | - | F | - | F | F | F |
| `start` | - | - | - | F, U | F | - |
| `stop` | F | - | - | - | F | - |
| `restart` | - | - | - | F | F | F |
| `doctor` | F | - | F | F | F | F |

46 of the 78 cells apply. 45 of them have a fake test.
The one with none is `confirm` across a lock. Every `U` beside an `F` is a part of that cell the fakes do not reach; the lists below name it.

Tests are in `tests/test_voice.py` (`voice`), `test_control.py` (`control`), `test_cli.py` (`cli`),
`test_audio.py` (`audio`), `test_model.py` (`model`), `test_agent.py` (`agent`), `test_services.py`
(`services`) and `test_transitions.py` (`transitions`); the local ones are
all in `test_transitions_local.py`.

### Microphone lost

- `run`, F: voice `test_a_microphone_that_ends_on_its_own_is_reopened_and_requests_are_heard_again`,
  `test_a_microphone_lost_mid_request_ends_it_without_an_apology`,
  `test_a_microphone_ended_by_stopping_jarvis_is_not_reopened`,
  `test_a_reopen_refused_by_a_stopped_microphone_ends_the_loop`; control
  `test_a_capture_that_ended_on_its_own_is_started_again`,
  `test_each_failed_reopen_waits_longer_up_to_a_cap_and_a_frame_starts_over`,
  `test_a_capture_tool_that_cannot_start_is_tried_again_later`; audio
  `test_a_microphone_that_sends_nothing_counts_as_ended_after_its_stall_time`; cli
  `test_the_run_reopens_a_lost_microphone_and_notices_one_gone_silent`.
- `run`, L: `test_the_microphone_sends_frames_without_stalling` (a healthy microphone, with the owner;
  not run yet). It does not unplug anything.
- `run`, O\*: 2026-11-21 at 15:13 and 15:15, PipeWire restarted and `jarvis run` exited.
- `status`, F: control `test_the_lost_microphone_state_is_the_voice_loops`, `test_the_public_state_is_one_of_the_bars_states`.
- `pause`, F: control `test_a_pause_while_waiting_for_the_microphone_frees_the_model_at_once`,
  `test_a_pause_during_a_reopen_wait_leaves_the_microphone_closed_until_resume`,
  `test_a_pause_before_a_reopen_leaves_the_microphone_closed_until_resume`.
- `resume`, F: control `test_a_pause_and_resume_during_a_reopen_wait_leave_one_capture`; transitions
  `test_every_command_still_answers_while_the_microphone_is_gone`.
- `hush`, F: transitions `test_every_command_still_answers_while_the_microphone_is_gone`,
  `test_a_hush_while_the_microphone_is_gone_reaches_the_loop`.
- `confirm`, F: transitions `test_every_command_still_answers_while_the_microphone_is_gone`; voice
  `test_a_microphone_lost_while_waiting_for_a_spoken_yes_is_a_no_and_is_reopened`,
  `test_a_microphone_lost_while_waiting_for_the_confirm_key_is_a_no_and_the_next_wait_works`. A microphone
  lost while Jarvis waits for a spoken yes or the confirm key counts as a no, the wait is disarmed, and the
  next question is asked afresh once the microphone is open again.
- `stop`, F: control `test_stop_ends_a_reopen_wait_without_starting_a_capture`.
- `doctor`, F: cli `test_doctor_checks_that_the_configured_source_is_a_microphone_node`,
  `test_doctor_says_when_pipewire_cannot_be_read_rather_than_blaming_the_source`.

### Suspend and resume

- `run`, F: voice `test_after_a_suspend_the_loop_listens_afresh_and_says_so`,
  `test_no_suspend_is_seen_when_the_clocks_agree`; cli `test_after_a_suspend_the_run_loads_a_failed_model_again`;
  transitions `test_the_time_asleep_is_how_far_the_boot_clock_has_run_ahead_of_the_monotonic_clock`,
  `test_awake_the_time_asleep_does_not_move_between_two_reads`.
- `run`, F: voice `test_a_suspend_while_a_request_is_in_flight_is_noticed_once_it_is_answered` (the request
  is answered, then the loop listens afresh before the next one),
  `test_a_microphone_lost_right_after_a_suspend_is_reopened_and_listening_is_afresh` (a lost microphone
  right after a suspend, with and without a frame between them).
- `run`, U: that the stall check stays quiet through a real sleep (it times with the monotonic clock, which
  stands still while asleep), and the follow-up window across a suspend (read from the code, not tried: the
  window counts audio frames and does not ask about a sleep, so speech heard in what is left of it after
  waking would be taken without the wake word).
- `status`, F: control `test_the_public_state_is_one_of_the_bars_states` (a model whose load failed is `unavailable`).
- `pause`, F: control `test_after_a_suspend_a_paused_jarvis_leaves_the_model_alone`.
- `confirm`, F: voice `test_a_yes_spoken_after_a_suspend_is_not_a_yes_to_the_question_asked_before_it`,
  `test_a_confirm_key_wait_open_across_a_suspend_has_run_out_when_the_pc_wakes`. A spoken yes waiting when
  the PC sleeps counts as a no (the first test failed before the fix: a yes after waking approved the old
  question); the confirm key's wait runs on the boot clock, so it has run out when the PC wakes.
- `unload`, F: control `test_after_a_suspend_the_model_is_loaded_again_only_if_its_load_failed` (an unloaded
  model stays unloaded); transitions `test_after_a_suspend_a_model_still_loading_or_unreachable_is_left_alone`.

### Lock and unlock

- `run`, F: voice `test_a_wake_word_is_ignored_while_the_screen_is_locked`, `test_a_wake_word_is_answered_while_unlocked`,
  `test_a_pc_request_is_ignored_while_the_screen_is_locked`, `test_no_follow_up_is_taken_once_the_screen_locked`,
  `test_a_request_spoken_over_jarvis_at_the_lock_screen_is_ignored`,
  `test_a_wake_word_ignored_at_the_lock_screen_is_taken_after_the_unlock` (a lock and then an unlock in one
  run); cli
  `test_with_listening_at_the_lock_screen_off_the_run_asks_the_shell`,
  `test_the_lock_screen_is_not_asked_about_while_listening_there_is_on`; audio
  `test_the_lock_screen_is_read_from_the_omarchy_shell`, `test_a_lock_state_that_cannot_be_read_counts_as_locked`,
  `test_a_lock_query_that_cannot_run_counts_as_locked`.
- `run`, L: `test_the_omarchy_shell_says_the_screen_is_unlocked` (passed 2026-11-22; the locked answer was not
  read, since that needs the owner at a locked screen).
- `doctor`, F: cli `test_doctor_checks_the_lock_screen_can_be_read_when_jarvis_must_not_listen_there`,
  `test_doctor_leaves_the_lock_screen_alone_while_jarvis_listens_there`.
- `confirm`, U: the control commands are not held back by the lock (nothing in `control.py` asks about it), so a pending question would still be answerable with `jarvis confirm` (read from the code, not tried), and the spoken yes to it is not checked
  against the lock either. Whether it should be held back is undecided; nothing tests either way.
- `status`, `pause`, `resume`, `hush`, `unload`: not held back by the lock either; key bindings are not expected to fire at the lock screen (not checked). Not a test gap.

### Model drive missing

- `run`, F: model `test_a_server_down_with_its_model_file_missing_fails_at_once_and_says_why`; agent
  `test_a_missing_model_file_is_named_in_the_apology`; transitions
  `test_a_spoken_request_with_the_model_drive_missing_is_answered_at_once_with_why` (the real client and agent,
  and the apology reaches the speaker), `test_a_streamed_chat_with_the_model_drive_missing_fails_at_once`,
  `test_a_drive_that_goes_away_after_the_start_is_noticed_at_the_next_request`,
  `test_the_model_client_knows_the_configured_model_file`,
  `test_a_config_without_a_model_file_never_reports_the_drive_missing`,
  `test_a_server_that_is_not_there_is_not_loaded_or_unloaded_and_says_so`,
  `test_the_voice_service_wants_the_model_server_but_does_not_need_it`,
  `test_after_a_suspend_a_missing_model_drive_is_logged_and_nothing_is_loaded`; services
  `test_the_strata_unit_for_a_normal_config` (the `ConditionPathExists=` line).
- `run`, L: `test_strata_answers_and_its_model_file_is_there` (the drive mounted),
  `test_the_installed_units_follow_the_session_and_wait_for_the_model_file` (written, not run).
- `run`, F: voice `test_a_spoken_request_with_the_model_drive_missing_is_apologised_for_at_once_by_voice` (the
  real agent behind the voice loop, with a fake speaker).
- `run`, U: the apology through the real audio output, and booting without the drive.
- `ask`, `chat`, F: the same reply as `run`, from agent `test_a_missing_model_file_is_named_in_the_apology`.
- `status`, F: control `test_a_model_server_down_for_a_missing_model_file_says_so`,
  `test_status_without_jarvis_names_a_missing_model_file`.
- `pause`, `resume`, F: transitions `test_pause_with_the_model_drive_missing_still_lets_go_of_the_microphone`,
  `test_resume_with_the_model_drive_missing_listens_again_and_asks_for_the_model`.
- `unload`, F: transitions `test_unload_with_the_model_drive_missing_says_it_could_not`,
  `test_unload_frees_the_model_when_the_server_answers`.
- `start`, F: services (as for `run`) and cli `test_start_clears_a_failed_state_before_starting_both_units`.
  U: that `systemctl start` skips Strata and still starts Jarvis, which only a boot without the drive shows.
- `restart`, F: cli `test_restart_starts_a_stopped_strata` (a skipped Strata is a stopped one).
- `doctor`, F: cli `test_doctor_still_fails_a_missing_file_and_points_at_the_models_doc`.

### Desktop session starts and stops

- `run`, F: services `test_both_units_start_and_stop_with_the_desktop_session`,
  `test_the_voice_service_starts_after_the_audio_session`, `test_both_units_start_only_once_their_program_has_started`;
  cli `test_a_stop_signal_only_raises_the_flag_and_the_stopper_ends_everything`,
  `test_a_loop_failure_after_a_stop_ends_the_run_cleanly`; control `test_a_stale_socket_left_by_a_crash_is_replaced`,
  `test_a_second_jarvis_does_not_take_over_the_socket`; transitions
  `test_a_run_ended_with_the_session_leaves_no_control_socket_for_the_next_one`,
  `test_a_run_that_fails_leaves_no_control_socket_either`,
  `test_a_run_that_cannot_serve_the_control_socket_still_listens`.
- `run`, L: `test_the_installed_units_follow_the_session_and_wait_for_the_model_file` (written, not run).
- `run`, O: 2026-11-21, the units stopped at 15:38 at shutdown and started at 15:42 after login.
- `status`, F: control `test_status_json_when_jarvis_is_not_running`, `test_status_without_jarvis_or_model_server_fails`,
  `test_status_json_exits_zero_even_with_nothing_reachable`. L: `test_status_names_a_state_the_bar_knows` (written, not run).
- `pause`, `resume`, `hush`, `confirm`, F: control `test_control_commands_without_a_running_jarvis_fail`. L:
  `test_pause_and_resume_reach_the_running_jarvis` (written, not run), `test_hush_reaches_the_running_jarvis` (written, not run).
- `unload`, F: transitions `test_unload_with_the_model_drive_missing_says_it_could_not` (Strata not running
  answers the same).
- `start`, F: cli `test_start_clears_a_failed_state_before_starting_both_units`, `test_a_failed_reset_does_not_stop_start`.
- `stop`, F: transitions `test_stop_stops_the_voice_service_and_the_model_server_and_says_where_they_are`,
  `test_a_stop_that_systemd_refuses_fails`.
- `restart`, F: cli `test_restart_restarts_strata_when_its_settings_changed_after_it_started`,
  `test_restart_leaves_a_running_strata_alone_when_nothing_changed`, `test_restart_starts_a_stopped_strata`,
  `test_restart_clears_a_failed_state_before_starting_anything`.
- `doctor`, F: cli `test_doctor_points_an_unreachable_strata_at_jarvis_start`.

### Strata down, its model file present

- `run`, F: model `test_a_dropped_connection_is_retried_once_when_the_server_is_back`,
  `test_a_server_down_with_its_model_file_present_is_waited_for`, `test_warming_waits_for_the_server_and_retries_a_busy_answer`,
  `test_warming_gives_up_on_a_server_that_never_comes_up`; agent `test_model_failure_becomes_spoken_apology`;
  transitions `test_the_model_drive_coming_back_changes_the_answer_from_missing_to_waiting_to_a_reply` (the
  drive back, then the server).
- `ask`, `chat`, F: agent `test_model_failure_becomes_spoken_apology` (the reply is "My model isn't answering
  right now" instead of the missing-drive one).
- `status`, F: control `test_an_unreachable_model_server_shows_unavailable`.
- `pause`, `resume`, `unload`, F: the transitions tests listed under the missing drive, which fake a server
  that does not answer and so do not depend on why.
- `restart`, F: cli `test_restart_starts_a_stopped_strata`. `doctor`, F: cli
  `test_doctor_points_an_unreachable_strata_at_jarvis_start`.
- Not seen on this PC: Strata crashing or restarting under a running Jarvis.

