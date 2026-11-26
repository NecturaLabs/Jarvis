# ADR: A workspace folder Jarvis can always write in

**Status:** proposed
**Date:** 2026-11-26

## Context and problem statement

Jarvis writes files only inside `files.write_roots` ([ADR 0002](0002-typed-operation-policy.md)), asks
before each write, refuses hidden names, and a file written with a bare name went under home, where it
was refused. The owner asked for one folder for everything Jarvis makes, under Documents and
configurable, so Jarvis is never blocked by access permissions there.

## Decision outcome

1. **A configured folder.** `files.workspace`, default `~/Documents/Jarvis`. It is made mode 0700 when
   Jarvis starts, at `jarvis setup` or on first use. It is added to both the read and the write roots,
   whatever those say.
2. **Only if it is really the owner's.** It must be a real folder (not a link at its own name) owned by
   this user that no group or other user can write to, under folders that others cannot write to and that
   no other user (but root) owns (unless sticky, like `/tmp`), so that no one else can move it away and put
   another in its place. A group-writable `~` or `~/Documents` is refused even under a private group of
   only the owner (umask 002), since Jarvis cannot tell it from a shared one; the doctor's message names
   the folder and the `chmod go-w` that fixes it. Otherwise Jarvis does not use it, says
   why in the log and `jarvis doctor` fails the check. A folder the owner made is never chmodded to fit.
   The folder is judged again at every write into it, and once more on the descriptor the write goes
   through (same device and inode as judged, still owned by the owner and not writable by others): one made
   writable by others, swapped for a link or another folder, or moved since start is refused, and a missing
   one is made again, mode 0700. Config validation refuses a
   workspace that is home or holds it, a hidden folder under home or inside one (`~/.config`,
   `~/.local`), one that holds or lies inside Jarvis's state or config folders (or `/proc`, `/sys`,
   `/dev`, `/run`), or that uses `..`.
3. **Where things go.** A write with a bare name or a relative path lands in the workspace; `~/...` and
   absolute paths are unchanged. Reads of a relative path look in the workspace first, then under home as
   before. The chat's `/export` saves there and `pc_shell` starts there. The prompt names it, so the model
   saves its work there unless told otherwise.
4. **No question for what costs nothing.** In the workspace, a call that makes a new file, or writes a file
   Jarvis made there (remembered in the state folder by device, inode, change and modification time and
   size, with one name on disk), is `safe` for that call: the owner loses nothing. The operation says so
   (`Operation.unasked`), the policy applies it only when the owner has not set `write_file`'s class and the
   operation's floor allows it, and it records the reason in the audit's `tool_call`. The verdict is bound
   to the write: a "new file" write makes the file exclusively whatever its mode, so a file that appeared
   since is never touched, and a "made by Jarvis" write refuses if that file, as opened (for an append, on
   the open descriptor), is no longer the one judged. A filesystem may keep times coarsely, so an in-place
   rewrite of the same size within one tick of Jarvis's own write keeps the change time, the modification
   time and the size alike, and passes. That is a file Jarvis made, in Jarvis's own folder, rewritten by the
   owner (or another process of theirs) at the same instant; nothing outside the folder is reached, so the
   ledger stays safe on such a filesystem. Outside
   text in the conversation, or a request heard in PC audio, raises the call back to `confirm` as it does
   for any side effect. Replacing or adding to a file the owner put there, or one swapped since Jarvis made
   it, asks as usual. The bound path, new-folder and old-file checks of the write do not change.
5. **Hidden names, not secrets, and never unasked.** Inside the workspace a name that starts with a dot is
   allowed, but it always asks: `.envrc`, `.mise.toml`, `.vscode` and the like act on their own. Secret
   names and paths, `files.deny` and Jarvis's own folders still apply, and `.git`, `.hg` and `.svn` are
   refused, since a hook in them runs programs.
6. **The shell starts there.** `pc_shell` starts in the workspace (home if it is missing or is a link when
   the command is prepared). The folder is bound into the call's arguments, so the question ("in
   ~/Documents/Jarvis") and the run name the same folder; if it is gone by the run, nothing runs.

### Consequences

- **Good:** Jarvis always has a place to put notes, drafts and code, without a question or a refused path.
- **Bad:** one more config key and a small state file; a model fed outside text still cannot write
  unasked, so a page cannot use the workspace to plant files quietly.
- **Open:** a file the owner edits with an editor that renames a new file over it stops being Jarvis's
  and asks next time; that is the safe side.

## Related

- `docs/adr/0002-typed-operation-policy.md`, `docs/configuration.md` (`[files]`)
