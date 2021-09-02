from typing import List
from .exceptions import InternalError
from .symbol import Symbol


class InverseSymbolTable:
  def __init__(self):
    self._symbols = {}

  def pop(self, obj):
    if obj in self._symbols:
      self._symbols.pop(obj)

  def items(self):
    return self._symbols.items()

  def values(self):
    return self._symbols.values()

  def keys(self):
    return self._symbols.keys()

  def __setitem__(self, obj, symbol: Symbol):
    self._symbols[obj] = symbol

  def __getitem__(self, obj):
    return self._symbols[obj]

  def __contains__(self, item):
    return item in self._symbols


class Scopes:
  GLOBAL_SCOPE = 0

  def __init__(self):
    self._global_scope = InverseSymbolTable()
    self._inv_tables: List[InverseSymbolTable] = [self._global_scope]

  def add_to_global(self, symbol: Symbol):
    try:
      self._does_name_exist([self._inv_tables[Scopes.GLOBAL_SCOPE]], symbol.name)
      self._inv_tables[Scopes.GLOBAL_SCOPE][symbol.obj] = symbol
    except InternalError:
      # NOTE: probably the same batch was used in different gemm descriptions
      # In other words, the same matrix is need in different places.
      # It is ok to not react on the exceotion
      pass

  def add_symbol(self, symbol: Symbol):
    self._does_name_exist(self._inv_tables, symbol.name)
    self._inv_tables[-1][symbol.obj] = symbol

  def delete_symbol(self, obj):
    self._inv_tables[-1].pop(obj)

  def delete_from_global(self, obj):
    self._inv_tables[Scopes.GLOBAL_SCOPE].pop(obj)

  def get_symbol(self, obj):
    for table in reversed(self._inv_tables):
      if obj in table:
        return table[obj]

  def get_global_scope(self):
    return self._inv_tables[Scopes.GLOBAL_SCOPE]

  def add_scope(self):
    self._inv_tables.append(InverseSymbolTable())

  def remove_scope(self):
    if len(self._inv_tables) > 1:
      self._inv_tables.pop()
    else:
      raise InternalError("attempt to delete global scope")

  def get_num_scopes(self):
    return len(self._inv_tables)

  def print_scope(self, level=-1):
    if level > len(self._inv_tables):
      raise InternalError(f'level {level} exceeds num scopes equal to {len(self._inv_tables)}')

    for symbol in self._inv_tables[level].values():
      print(symbol)

  def print_scopes(self):
    for level, _ in enumerate(self._inv_tables):
      print('*' * 80)
      self.print_scope(level)

  def __contains__(self, obj):
    result = False
    for scope in reversed(self._inv_tables):
      if obj in scope.keys():
        result = True
        break

    return result

  def __str__(self):
    data = []
    for counter, table in enumerate(self._inv_tables):
      table_name = f'scope level {counter - 1}'
      if counter == 0:
        table_name = 'global scope'

      data.append(table_name)
      for key, value in table.items():
        data.append(f'  name: {value}; obj-id: {hex(id(key))}')

    return '\n'.join(data)

  def _does_name_exist(self, scope_list: List[InverseSymbolTable], name: str):
    for scope_level, scope in enumerate(scope_list):
      for item in scope.values():
        if item.name == name:
          raise InternalError(f'name has already been occupied: {item.name} - {name}')
