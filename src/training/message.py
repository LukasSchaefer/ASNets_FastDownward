class Message(object):
    """
    Message objects provide a way for different components to communicate. They
    provide the possibility for objects to add messages and multiple other
    objects can read their content. If desired for each reader their reading
    progress can be stored.
    Message objects to not provide a feature to store who has added which
    message. If multiple objects shall have a discussion, you might want to
    submit messages as tuple (sender, message).
    """
    def __init__(self):
        # Stores which are the last messages an 'something' has read. None
        # is the general reader if no reader is given.
        self._read_by_reader = {}
        self._messages = []

    def size(self):
        """
        Number of messages managed
        :return:
        """
        return len(self._messages)

    def add(self, msg):
        """
        Add new message to the list
        :param msg: new message
        :return:
        """
        self._messages.append(msg)

    def last(self, update=True, reader=None):
        """
        Return the newest message
        :param update: if True, updates the reading progress of the reader
        :param reader: identifier for the reader or None
        :return:
        """
        if self.size() == 0:
            return None
        if update:
            self._read_by_reader[reader] = self.size() - 1
        return self._messages[-1]

    def all(self, update=True, reader=None):
        """
        Return all messages
        :param update: if True, updates the reading progress of the reader
        :param reader: identifier for the reader or None
        :return:
        """
        if update:
            self._read_by_reader[reader] = self.size() - 1
        return self._messages

    def new(self, update=True, reader=None):
        """
        Return unread messages
        :param update: if True, updates the reading progress of the reader
        :param reader: identifier for the reader or None
        :return:
        """
        old_bookmark = -1 if not reader in self._read_by_reader else self._read_by_reader[reader]
        if update:
            self._read_by_reader[reader] = self.size() - 1
        return self._messages[old_bookmark + 1:]

    def get(self, idx, till=None, update=True, reader=None):
        """
        Return some message(s). If a single message is demanded, then this
        single message is returned. If a list of messages is looked up, then
        a list is returned.
        :param idx: index of the first message to receive
        :param till: optional. index of the last message to receive
        :param update: if True, updates the reading progress of the reader
        :param reader: identifier for the reader or None
        :return:
        """
        if till is None:
            if update:
                self._read_by_reader[reader] = max(self._read_by_reader[reader], idx)
            return self._messages[idx]
        else:
            if update:
                self._read_by_reader[reader] = max(self._read_by_reader[reader], till)
            return self._messages[idx: till + 1]
