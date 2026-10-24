"""The Studio UI's small command and observation boundary.

Only this adapter knows about the task and rendering workers.  Presentation
widgets consume their signals and immutable frame packets; they never touch
the live simulation or Agent.
"""
from .runtime import AgentWorker, RenderMailbox, RenderWorker


class StudioGateway:
    def __init__(self, config):
        self.mailbox = RenderMailbox()
        self.renderer = RenderWorker(self.mailbox)
        self.worker = AgentWorker(config, self.mailbox)

    def start(self):
        self.worker.start()
        self.renderer.start()

    def submit(self, command, mode):
        self.worker.submit(command, mode)

    def stop(self):
        self.worker.stop()

    def reset(self):
        self.worker.reset()

    def camera(self, state):
        self.mailbox.set_camera(state)

    def size(self, width, height):
        self.mailbox.set_size((width, height))

    def frame(self, last_revision):
        return self.mailbox.take(last_revision)

    def close(self):
        self.worker.quit_event.set()
        self.worker.stop()
        self.renderer.quit_event.set()

    def is_running(self):
        return self.worker.isRunning() or self.renderer.isRunning()
