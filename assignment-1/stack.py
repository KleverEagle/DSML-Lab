class Stack:
    def __init__(self, *args):
        self._list = []
        for i in args:
            self.add(i)

    def pop(self):
        if self._list:
            return self._list.pop()

    def top(self):
        if self._list:
            return self._list[-1]

    def add(self, value):
        self._list.append(value)

    def is_empty(self):
        return not self._list

    def print_stack(self):
        print(self._list)

stack = Stack(1, 2, 3)
print(stack.top())
print(stack.pop())
stack.add(5)
print(stack.top())

stack.print_stack()
