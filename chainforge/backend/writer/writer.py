from io import StringIO

class Block:
  def __init__(self, writer):
    self.writer = writer

  def __enter__(self):
    self.writer('{')
    self.writer.mv_right()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.writer.mv_left()
    self.writer('}')

  def __call__(self, line):
    self.writer(line)


class Writer:
  def __init__(self, indent_factor=2):
    self.factor = indent_factor
    self.curr_indent = 0
    self.stream = StringIO()

  def __enter__(self):
    self.text = []

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def __call__(self, line):
    ws = (' ' * self.factor) * self.curr_indent
    self.stream.write(f'{ws}{line}\n')

  def mv_left(self):
    self.curr_indent -= 1

  def mv_right(self):
    self.curr_indent += 1

  def block(self, block_name=None):
    if block_name:
      self.__call__(block_name)
    return Block(writer=self)

  def new_line(self):
    self.__call__('')

  def insert_pragma_unroll(self):
    self.__call__(f'#pragma unroll')

  def get_src(self):
    return self.stream.getvalue()
