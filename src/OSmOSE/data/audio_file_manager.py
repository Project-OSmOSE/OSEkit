import soundfile as sf


class AudioFileManager:
    def __init__(self):
        self.opened_file = None
        self.calls = 0
        self.opens = 0

    def close(self):
        if self.opened_file is None:
            return
        self.opened_file.close()
        self.opened_file = None

    def open(self, path):
        self.opened_file = sf.SoundFile(path, "r")

    def switch(self, path):
        self.calls += 1
        if self.opened_file is None:
            self.open(path)
        if self.opened_file.name == str(path):
            return
        self.close()
        self.open(path)
        self.opens += 1

    def read(self, path, start: int, stop: int):
        self.switch(path)
        self.opened_file.seek(start)
        return self.opened_file.read(stop-start)

    def info(self, path):
        self.switch(path)
        return self.opened_file.samplerate, self.opened_file.frames, self.opened_file.channels
