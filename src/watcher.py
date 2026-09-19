"""Filesystem watching with per-path debounce.

inotify was verified to work on this vault before this phase was committed to:
it sits on a fuseblk (NTFS-3G) mount, where delivery is not guaranteed. A probe
watching from inside a container through a :ro bind mount received CREATE,
MODIFY, CLOSE_WRITE and DELETE for writes made from the host and from the
obsidian container's own separate mount of the same directory.

Directory arrivals are handled here rather than left to watchdog. It adds a
watch for a new subdirectory, and replays what is already inside it, only for
one it saw *created* - inotify_c.py gates both on `is_create`, with a standing
TODO for the rest. A directory moved in from elsewhere arrives as IN_MOVED_TO
and gets neither, so nothing under it is ever seen again. That is not
hypothetical: AI/Prompts/Server was moved into the vault whole on 2026-09-17
and stayed absent from index.md and from search, with no watch on it at all,
until the next restart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import documents, vault
from .config import settings

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Runs on watchdog's thread. Does nothing but hand paths to the loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[Path],
        dirs: asyncio.Queue[Path],
    ) -> None:
        self._loop = loop
        self._queue = queue
        self._dirs = dirs

    def submit(self, raw_path: str | bytes) -> None:
        path = Path(raw_path.decode() if isinstance(raw_path, bytes) else raw_path)
        if path.suffix.lower() != ".md" and not documents.is_document(path):
            return
        try:
            relative = path.resolve().relative_to(vault.ROOT)
        except ValueError:
            return
        if vault.is_protected(relative):
            return
        # Every other note goes through. The watcher used to drop anything
        # is_search_excluded() named - but index.md now has a say too, and it
        # wants a different answer: Reports/PC/ is not searched yet is
        # navigated. Deciding here would have to satisfy both, so the consumers
        # decide instead.
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    def _submit_dir(self, raw_path: str | bytes) -> None:
        """Hand a directory to the loop thread; see VaultWatcher._adopt.

        Deliberately not done here: rglob and Observer.schedule both block, and
        this runs on watchdog's dispatch thread, where scheduling a watch means
        taking a lock that thread may already hold."""
        path = Path(raw_path.decode() if isinstance(raw_path, bytes) else raw_path)
        try:
            relative = path.resolve().relative_to(vault.ROOT)
        except ValueError:
            return
        if vault.is_protected(relative):
            return
        self._loop.call_soon_threadsafe(self._dirs.put_nowait, path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            # Not only mkdir. A directory renamed in from outside the vault
            # raises IN_MOVED_TO with no IN_MOVED_FROM to pair it with, and
            # watchdog reports an unpaired IN_MOVED_TO as a *creation* - so
            # this is the branch the real failure came down, and on_moved
            # never fired at all. The raw mask is still IN_MOVED_TO, which is
            # not what inotify_c.py gates its own _add_watch on, so there may
            # be no watch here however this event is labelled.
            self._submit_dir(event.src_path)
        else:
            self.submit(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.submit(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.submit(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            # A rename within the vault, where both ends are watched. The
            # source's notes are left to the periodic reconcile: they are gone,
            # so there is nothing to walk to learn what their paths were.
            self._submit_dir(event.dest_path)
        else:
            self.submit(event.src_path)  # source: drop its rows
            self.submit(event.dest_path)  # destination: index it


class VaultWatcher:
    def __init__(self, on_change: Callable[[Path], Awaitable[None]]) -> None:
        self._on_change = on_change
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._dirs: asyncio.Queue[Path] = asyncio.Queue()
        self._observer: Observer | None = None
        self._handler: _Handler | None = None
        self._task: asyncio.Task | None = None
        self._dir_task: asyncio.Task | None = None
        self._adopted: set[Path] = set()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._observer = Observer()
        self._handler = _Handler(loop, self._queue, self._dirs)
        self._observer.schedule(self._handler, str(vault.ROOT), recursive=True)
        self._observer.start()
        self._task = asyncio.create_task(self._drain(), name="vault-watch")
        self._dir_task = asyncio.create_task(self._drain_dirs(), name="vault-watch-dirs")
        log.info("watching %s (debounce %.1fs)", vault.ROOT, settings.watch_debounce_seconds)

    async def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
        for task in (self._task, self._dir_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _adopt(self, directory: Path) -> list[Path]:
        """Watch a directory that has just appeared, and list the notes in it.

        The watch is added unconditionally rather than only when we can tell
        watchdog skipped it, because there is no way to ask it. A directory
        renamed in gets no watch from inotify_c.py, which is the bug; one made
        with mkdir does, and scheduling a second is not free but is harmless -
        the duplicate events coalesce in the debounce, and everything
        downstream recomputes from the file rather than accumulating. Paying
        that on the rare mkdir beats guessing wrong on the rename.

        Runs in a worker thread: rglob crosses the mount, and
        Observer.schedule takes watchdog's own lock, which must not be taken
        from its dispatch thread.
        """
        if not directory.is_dir():
            self._adopted.discard(directory)
            return []  # moved on again, or removed, before we got to it

        if self._observer is not None and directory not in self._adopted:
            try:
                self._observer.schedule(self._handler, str(directory), recursive=True)
                self._adopted.add(directory)
                log.info("watching %s", vault.relpath(directory))
            except OSError:
                # The notes below are still indexed by the walk; only later
                # edits to them would be missed, and reconcile catches those.
                log.exception("cannot watch %s", directory)

        # A Files/ folder arrives as a directory event when the first document
        # is filed into a note's folder, and its contents have to be adopted the
        # same way a folder of notes is - otherwise the first upload into a new
        # folder is indexed only by the next full rebuild.
        return sorted(
            path
            for path in directory.rglob("*")
            if path.suffix.lower() == ".md" or documents.is_document(path)
        )

    async def _drain_dirs(self) -> None:
        """Feed the notes under an arrived directory back through the queue."""
        while True:
            directory = await self._dirs.get()
            try:
                notes = await asyncio.to_thread(self._adopt, directory)
            except Exception:
                log.exception("cannot adopt %s", directory)
                continue
            if notes and self._handler is not None:
                log.info("%s: %d note(s) to index", vault.relpath(directory), len(notes))
                for note in notes:
                    self._handler.submit(note)

    async def _drain(self) -> None:
        """Coalesce bursts. Obsidian and Samba both emit several write events
        for one logical save; re-embedding on each would hammer Ollama."""
        pending: dict[Path, float] = {}
        debounce = settings.watch_debounce_seconds

        while True:
            timeout = debounce if pending else None
            try:
                path = await asyncio.wait_for(self._queue.get(), timeout)
                pending[path] = time.monotonic()
            except asyncio.TimeoutError:
                pass

            now = time.monotonic()
            settled = [p for p, seen in pending.items() if now - seen >= debounce]
            for path in settled:
                del pending[path]
                try:
                    await self._on_change(path)
                except Exception:
                    log.exception("reindex failed for %s", path)
