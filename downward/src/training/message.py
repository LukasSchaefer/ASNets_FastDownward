class Message(object):
    def __init__(self):
        self._messages = []

    def add(self, msg):
        self._messages.append(msg)

    def last(self):
        return self._messages[-1]

    def all(self):
        return self._messages

    def get(self, idx):
        return self._messages[idx]

    def size(self):
        return len(self._messages)