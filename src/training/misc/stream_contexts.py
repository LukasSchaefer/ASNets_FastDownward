from .misc import MatchRegex

from .. import parser
from .. import parser_tools as parset
from .. parser_tools import ArgumentException, main_register

import os
import gzip
import sys

class StreamDefinition(object):
    arguments = parset.ClassArguments("StreamDefinition", None,
        ("file", True, None, str, "File to write to. If None, this field will"
                                "be automatically filled on next."),
        ("directory", True, None, str,
        "Works only with file=None. Sets the file path given on next to this"
        "directory plus the name of the file given in the file path"),
        ("mode", True, 0, int, "Write Mode: 0 = write with overwriting previous, "
                               "1 = write without overwrite a previous file, "
                               "2 = raise error if it should overwrite a previous file "
                               "3 = append to previous (or create if no previous)"),
        ("valid", True, None, main_register.get_register(MatchRegex), "true when this stream apples"),
        ("suffix", True, ".data", str, "If not using 'file', then this suffix is"
                                  "appended to all automatically generated target"
                                  "file names."),
        ("id", True, None, str, "ID")
        )
    def __init__(self, open, close, file=None, directory=None, mode=1,
                 valid=None, suffix=".data", convert=None, write=None, id=None):
        """

        :param file: file for opening
        :param mode: mode for opening
        :param open: callable (file, mode) opening the stream
        :param valid: callable(file) telling if the stream shall be used while
                      working on the given file (e.g. to disable output for
                      some files)
        :param convert: callable(*args, **kwargs) which converts the input given
                        to write to the format needed the file descriptor
        :param close: callable(file handle) to close the file descriptor
        :param id:
        """
        self.file = file
        self.directory = directory
        self.mode = "a" if mode == 3 else "w"
        self.allow_reuse = mode == 1
        self.allow_overwrite = mode == 0 or mode == 3
        self.error_overwrite = mode == 2
        self.valid = (lambda x: True) if valid is None else valid
        self.suffix = suffix
        self.open = open
        self.close = close
        self.converter = StreamDefinition.default_convert if convert is None else convert
        self._write = StreamDefinition.default_write if write is None else write
        self.id = id

        if self.file is not None and self.directory is not None:
            raise ValueError("Either a directory in which the data can"
                             "be stored shall be given OR a file in which all"
                             "data shall be stored, but NOT BOTH.")

        self.fd = None  # File handle
        self.currently_valid = False
        self.new = True


    @staticmethod
    def default_write(fd, *args, **kwargs):
        fd.write(*args, **kwargs)

    @staticmethod
    def default_convert(*args, **kwargs):
        return args, kwargs

    def may_reuse(self, path):
        """

        :param path: path to the file for which the writing is done
        :return:
        """
        path = self.get_next_path(path)
        return os.path.exists(path) and self.allow_reuse

    def check_overwrite_condition(self, path):
        if not os.path.exists(path):
            return True
        if self.error_overwrite:
            raise ValueError("The mode does not allow to overwrite a file: "
                             + str(path))
        return self.allow_overwrite

    def get_next_path(self, arg):
        path = None
        if self.file is not None:
            return self.file
        elif self.directory is not None:
            path = os.path.join(self.directory, os.path.basename(arg))
        else:
            path = arg
        return os.path.splitext(path)[0] + self.suffix

    def next(self, arg):
        self.currently_valid = self.valid(arg)

        # First time, if one file for all
        if self.file is not None:
            if self.new:
                if not self.check_overwrite_condition(arg):
                    raise ValueError("File to write to exists already and"
                                     "overwriting is not allowed.")
                self.fd = self.open(self.file, self.mode)
                self.new = False
        else:
            if self.currently_valid:
                arg = self.get_next_path(arg)
                self.currently_valid = (
                    self.currently_valid and self.check_overwrite_condition(arg))
                if self.currently_valid:
                    self.fd = self.open(arg, self.mode)

    def exit(self):
        if self.file is None:
            self.finalize()
        self.currently_valid = False

    def write(self, *args, **kwargs):
        if self.currently_valid:
            a, k = self.converter(*args, **kwargs)
            self._write(self.fd, *a, **k)

    def finalize(self):
        if self.fd is not None:
            self.close(self.fd)
            self.fd = None
        self.currently_valid = False

    @staticmethod
    def parse(tree, item_cache):
        obj = parser.try_lookup_obj(tree, item_cache, StreamDefinition, None)
        if obj is not None:
            return obj
        else:
            raise ArgumentException("The definition of the base Stream can "
                                    "only be used for look up of any previously"
                                    " defined condition via 'Bridge(id=ID)'")


main_register.append_register(StreamDefinition, "stream")


class OpenStreamDefinition(StreamDefinition):
    arguments = parset.ClassArguments("OpenStreamDefinition", StreamDefinition.arguments)

    def __init__(self, file, directory, mode, valid, suffix, id):
        StreamDefinition.__init__(self,
                                  open=open, close=lambda x: x.close(),
                                  file=file, directory=directory,
                                  mode=mode, valid=valid, suffix=suffix, id=id)

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  OpenStreamDefinition)


main_register.append_register(OpenStreamDefinition, "open")


class GzipStreamDefinition(StreamDefinition):
    arguments = parset.ClassArguments("GzipStreamDefinition", StreamDefinition.arguments)

    def __init__(self, file, directory, mode, valid, suffix, id):
        conv = None
        if sys.version_info[0] > 2:
            conv = GzipStreamDefinition.default_py3_convert
        StreamDefinition.__init__(self,
                                  open=gzip.open, close=lambda x: x.close(),
                                  file=file, directory=directory,
                                  mode=mode, convert=conv,
                                  valid=valid, suffix=suffix, id=id)
        self.mode += "b"

    @staticmethod
    def default_py3_convert(*args, **kwargs):
        a = list(args)
        for i in range(len(a)):
            a[i] = str(a[i]).encode()
        return a, kwargs

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  GzipStreamDefinition)


main_register.append_register(GzipStreamDefinition, "gzip")


# Dummy Context manager
class DummyContext(object):
    arguments = parset.ClassArguments("DummyContext", None)

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, *args, **kwargs):
        pass

    def next(self, *args):
        pass

    def finalize(self):
        pass


class StreamContext(object):
    arguments = parset.ClassArguments("StreamContext", None,
        ("streams", True, None, main_register.get_register(StreamDefinition),
         "(List of ) stream object(s) to feed the write data to."),
        ("id", True, None, str))

    def __init__(self, streams=None, id=None):
        self._streams = [] if streams is None else streams
        if not isinstance(self._streams, list):
            self._streams = [self._streams]
        self._id = id

    def may_reuse(self, arg):
        for stream in self._streams:
            if not stream.may_reuse(arg):
                return False
        return True

    def next(self, *args):
        for stream in self._streams:
            stream.next(*args)
        return self

    def __enter__(self):
        return self

    def write(self, *args, **kwargs):
        for stream in self._streams:
            stream.write(*args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.finalize()
        else:
            for stream in self._streams:
                stream.exit()

    def finalize(self):
        for stream in self._streams:
            stream.finalize()

    @staticmethod
    def parse(tree, item_cache):
        return parser.try_whole_obj_parse_process(tree, item_cache,
                                                  StreamContext)


main_register.append_register(StreamContext, "streams")
