# ADR: Keep reminders and scheduled requests in Jarvis's own jobs file, said by the voice service

**Status:** accepted
**Date:** 2026-11-23

## Context and problem statement

The owner asked for reminders and scheduled things ("cron jobs"): a reminder said at a time, once or
again and again, and a request carried out later ("pause the music at midnight"). Three questions had to be
settled:

- where the jobs live and what runs them when they fall due;
- how a time is written: a cron line, or typed fields;
- what a scheduled request is when it runs: whose words they are, and what may confirm its actions.

Jarvis replies by voice only, so only the voice service (`jarvis run`) can say a job. The chat window and
`jarvis ask` must still be able to set one.

## Considered options

Where the jobs live:

- A systemd user timer per job (`systemd-run --user --on-calendar`), each starting `jarvis ask`.
- One private jobs file in the state folder, read by the voice service between requests.

How a time is written:

- A cron line or systemd calendar expression the model writes.
- Typed fields: minutes from now, or a time of day with a day or date, and a repeat from a fixed set.

What a scheduled request is when it runs:

- The owner's own request, run through the agent with the full policy.
- A new kind of text that needs a yes for any action.
- No scheduled requests, only reminders.

## Decision outcome

Accepted on 2026-11-23: one jobs file, typed fields, and a scheduled request runs as the owner's own.

- **The jobs file.** `~/.local/state/jarvis/schedule.json` (mode 0600), changed under a lock file beside
  it, so the voice service, the chat window and `jarvis ask` see one list. A timer per job would make Jarvis
  write systemd units for words the model wrote, run a second Jarvis process that cannot speak through the
  voice service, and leave units behind when Jarvis is removed. The voice service looks at the file at most
  once a second and only when it changed, so it costs nothing while idle.
- **Typed fields.** `schedule_add` takes `in_seconds` (up to an hour, a one-off job only; added
  2026-11-25 for timers under a minute), `in_minutes` (up to a week), or `at` (HH:MM) with `day` (today,
  tomorrow or a weekday), and `repeat` (`none`, `daily`, `weekdays`, `weekends`, `weekly`, or `every`
  with `every_minutes`, 5 to 1440). The schema bounds every value; a cron line would need its own parser to
  be bounded, and a small local model writes typed fields more reliably. At most 50 jobs, 200 characters
  each.
- **A scheduled request is the owner's own.** When it falls due, its words go to the agent as the owner's
  request, with every rule in place: the policy, the cap on actions and the audit. A question it asks is
  asked aloud, and nobody answering is a no; a `strong` action still waits for the confirm key. A new kind
  of text that needs a yes for every action would make most scheduled requests fail when nobody is at the
  PC, which is when they are wanted. Setting one already goes through the policy: when outside text (a web
  page, a file, PC audio) is in the conversation, `schedule_add` needs a yes, and the question reads out the
  whole request, so a page cannot schedule one the owner did not hear.
- **A job keeps where its request came from** (amended 2026-11-25, #43). The
  yes when a job is added is not always asked (`boundaries.bypass_consent`), and
  a yes then is not a yes to every action the job takes later. So each job records whether its request was
  heard in PC audio (`foreign`) and the origins of the outside text that request had seen (`outside`: web,
  file, memory and so on). When it falls due, a request from PC audio runs as PC audio, and one added beside
  outside text runs with that text counted as in the conversation: its side effects need a yes, and after
  file text so does sending words off the PC, as they would have when it was added. A job the owner alone
  asked for still runs as the owner's own, unasked. A jobs file from before keeps its jobs as the owner's.
  Program names and window titles remembered from a process listing count too. A job's words are the
  model's, so an address in them is never one the owner typed, and the words of a job added beside outside
  text reach a later request as data when it is listed or cancelled. The audit records each job's
  provenance when it falls due.
- **Late jobs.** A job whose time passed while the voice service was not running, the PC slept, Jarvis was
  paused, or the owner was dictating or in a meeting, is late. A one-off reminder is still said up to six
  hours late, and says how late it is. A scheduled request or a repeating reminder more than ten minutes
  late is not done but audited as missed, and a repeating job moves on to its next time: a request run hours
  late ("pause the music at midnight", at breakfast) does the wrong thing.

### Consequences

- Nothing is said while the voice service is stopped. Setting a job from the chat window or `jarvis ask`
  says so when the service is not running.
- Nothing is said while the owner dictates or a meeting runs; due jobs wait for it to end, and the late
  rules apply to the wait.
- The words of each job are kept in the jobs file until it is said or cancelled, and in the audit log with
  the `schedule_add` call that set it ([privacy](../privacy.md)).
- A reminder is said even at the lock screen and over other audio, like an alarm. A scheduled request then
  asks its questions aloud like any other.
