import enum
from typing import Dict, Union
from chainforge.common import DenseMatrix, Addressing


class ObjType(enum.Enum):
  MATRIX = 0
  SCALAR = 1


class Attribute:
  def __init__(self, descr):
    self.descr = descr
    self.obj_type = None
    if isinstance(descr, DenseMatrix):
      self.obj_type = ObjType.MATRIX
    elif isinstance(descr, float):
      self.obj_type = ObjType.SCALAR
    else:
      raise ValueError(f'supports only scalars and matrices, given: {type(descr)}')

  def __str__(self):
    if self.obj_type == ObjType.SCALAR:
      return str(self.descr)
    else:
      mat = self.descr
      text = f'r = {mat.get_actual_num_rows()}, c = {mat.get_actual_num_cols()}, '
      text += f'ld = {mat.num_rows}, sd = {mat.num_cols}, '
      text += f'addr = {Addressing.addr2str(mat.addressing)}'
    return text


class Scope:
  def __init__(self):
    self.vars: Dict[str, Union[Attribute]] = {}

  def __str__(self):
    res = []
    for item in self.vars:
      res.append(f'{item}: {self.vars[item]}')

    return '\n'.join(res)


class SymbolTable:
  def __init__(self):
    self.scopes = [Scope()]

  def add(self, name, descr):
    top = self.scopes[-1]
    top.vars[name] = Attribute(descr)

  def find(self, name):
    for scope in reversed(self.scopes):
      if name in scope.vars:
        return scope.vars[name]
    return None

  def remove(self, name, level=-1):
    if name in self.scopes[level].vars:
      self.scopes[level].vars.pop(name)

  def force_remove(self, name):
    for level, scope in enumerate(self.scopes):
      self.remove(name, level)

  def add_scope(self):
    self.scopes.append(Scope())

  def pop_scope(self):
    self.scopes.pop()

  def get_matrices(self, from_level=-1):
    matrices = []
    for name, attr in self.scopes[from_level].vars.items():
      if attr.obj_type == ObjType.MATRIX:
        matrices.append((name, attr))
    return matrices

  def get_all_matrices(self):
    matrices = []
    for level in range(len(self.scopes)):
      matrices.extend(self.get_matrices(level))
    return matrices

  def print(self):
    level = 0
    for scope in self.scopes:
      print(f'level: {level}')
      print(scope)
