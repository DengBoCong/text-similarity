import collections


class Foo(object):
    def __init__(self):
        super(Foo, self).__init__()
        self.is_whole = 1
    def te(self):
        pass

f = Foo()
print(hasattr(f, "twe"))