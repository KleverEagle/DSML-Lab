class Queue:
    def __init__(self, *args):
        self._list = []
        for i in args:
            self.enqueue(i)

    def enqueue(self, value):
        self._list.append(value)

    def dequeue(self):
        if self._list:
            return self._list.pop(0)

    def get_front(self):
        if self._list:
            return self._list[0]

    def is_empty(self):
        return not self._list

    def print_queue(self):
        print(self._list)


q = Queue(1, 2, 3, 4)
print(q.dequeue())
q.enqueue(5)
print(q.get_front())
q.print_queue()
