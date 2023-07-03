import enum
from typing import Union, List
from copy import deepcopy


class SymbolType(enum.Enum):
  Batch = 1
  Global = 2
  SharedMem = 3
  Register = 4


class DataView:
  def __init__(self, rows: int, columns: int, is_transposed: bool, bbox: List[int] = None):
    self._rows = rows
    self._columns = columns
    self.is_transposed = is_transposed
    if not bbox:
      bbox = [0, 0, rows, columns]
    self._bbox = bbox
    self._lead_dim = self.get_lead_dim()
    self._offset = self.get_offset()

  def get_bbox(self):
    return deepcopy(self._bbox)

  def reset_bbox(self, bbox):
    assert bbox[2] - bbox[0] <= self._rows
    assert bbox[3] - bbox[1] <= self._columns
    self._bbox = bbox
    self._offset = self.get_offset()

  def get_offset(self):
    return self._bbox[0] + self._bbox[1] * self._lead_dim

  def get_volume(self):
    return self._rows * self._columns

  def get_lead_dim(self):
    return self._rows

  def get_dim_size(self, index):
    assert index >= 0 and index < 2
    return self._bbox[2 + index] - self._bbox[index]

  def get_address(self, row_idx, column_idx):
    addr = f'{row_idx} + {column_idx} * {self._lead_dim}'
    if self._offset:
      addr = f'{self._offset} + {addr}'
    return addr

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

  def __str__(self):
    return f'name: {self.name}, type: {self.stype}'
