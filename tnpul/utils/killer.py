import signal


class GracefulKiller():
    kill_now = False
    running = True

    def __init__(self, disable=False):
        if not disable:
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
            signal.signal(signal.SIGALRM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        self.running = False
