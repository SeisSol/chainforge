class ShrMemObject:
  def __init__(self, name, size=None, mults_per_block=None):
    self.name = name
    self._size_per_mult = size
    self._mults_per_block = mults_per_block

  def set_size_per_mult(self, size):
    self._size_per_mult = size

  def set_mults_per_block(self, num_mults):
    self._mults_per_block = num_mults

  def get_size_per_mult(self):
    return self._size_per_mult

  def get_mults_per_block(self):
    return self._mults_per_block

  def get_total_size(self):
    return self._size_per_mult * self._mults_per_block

  def get_total_size_as_str(self):
    total_size = self.get_total_size() if self._size_per_mult else 'not yet defined'
    return total_size

  def __str__(self) -> str:
    total_size = self.get_total_size() if self._size_per_mult else 'not yet defined'
    return f'name {self.name}: total size = {total_size}'


class RegMemObject:
  def __init__(self, name, size=None):
    self.name = name
    self.size = size

  def __str__(self):
    return f'name: {self.name}; size = {self.size}'
