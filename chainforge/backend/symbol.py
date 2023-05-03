import enum
from typing import Union


class SymbolType(enum.Enum):
  Batch = 1
  Global = 2
  SharedMem = 3
  Register = 4


class DataView:
  def __init__(self, rows: int, columns: int, lead_dim: int, is_transposed: bool):
    self.rows = rows
    self.columns = columns
    self.lead_dim = lead_dim
    self.is_transposed: bool = is_transposed

  def __str__(self):
    return f'rows: {self.rows}, cols: {self.columns}, lid: {self.lead_dim}, trans: {self.is_transposed}'


class Symbol:
  def __init__(self,
               name: str,
               stype: SymbolType,
               obj):
    self.name = name
    self.stype = stype
    self.obj = obj
    self.data_view: Union[DataView, None] = None
    self._users = []

  def add_user(self, user):
    self._users.append(user)

  def get_user_list(self):
    # set by instructions
    return self._users

  def get_fist_user(self):
    return self._users[0]

  def pop_user(self, user_index):
    self._users.pop(user_index)

  def remove_user(self, user):
    self._users.remove(user)

  def __str__(self):
    return f'name: {self.name}, type: {self.stype}'
