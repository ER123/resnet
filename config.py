#coding:utf-8
# This is a variable scope aware configuation object for TensorFlow

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].items():
            root[k] = v
        self.stack = [ root ]

    def iteritems(self):
        return self.to_dict().items()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top fo the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if not name in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)
