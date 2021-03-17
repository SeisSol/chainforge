from abc import ABC, abstractmethod


class Node(ABC):
  def __init__(self, name):
    self.name = name

  def __str__(self):
    return f'{self.name}'

  def get_label(self):
    return f'{self.name}'


class DeadNode(Node):
  def __init__(self):
    super().__init__(name='dead')


class VarNode(Node):
  def __init__(self, name):
    super().__init__(name)


class ScalarNode(VarNode):
  _counter = 0
  def __init__(self, name):
    super().__init__(name=name)

  @classmethod
  def gen_next_name(cls):
    ScalarNode._counter += 1
    return f'c{ScalarNode._counter}'


class MatrixNode(VarNode):
  def __init__(self, name, is_trans):
    super().__init__(name=name)
    self.is_trans = is_trans

  def _get_suffix(self):
    return '^T' if self.is_trans else ''

  def __str__(self):
    return f'{self.name}{self._get_suffix()}'

  def get_label(self):
    return self.__str__()


class BinarryOps(Node):
  def __init__(self, op_name, left, right):
    super().__init__(name=op_name)
    self.left = left
    self.right = right
    self._op = op_name

  def __str__(self):
    return f'{self.left} {self._op} {self.right}'


class AddNode(BinarryOps):
  def __init__(self, left, right):
    super().__init__('+', left, right)


class MultNode(BinarryOps):
  def __init__(self, left, right):
    super().__init__('*', left, right)


class GemmNode(BinarryOps):
  _counter = 0

  def __init__(self, left, right, res=None, alpha=1.0, beta=0.0):
    super().__init__('gemm', left, right)
    self.res = res
    self.alpha = alpha
    self.beta = beta

  def __str__(self):
    return f'gemm({self.alpha}, {self.left}, {self.right}, {self.beta}, {self.res})'

  @classmethod
  def make_tmp_name(cls):
    GemmNode._counter += 1
    return f't{GemmNode._counter}'


class ScaleMatrixNode(BinarryOps):
  def __init__(self, left, right):
    if not isinstance(left, ScalarNode):
      raise ValueError('left operand of ScaleNode must be scalar')

    super().__init__('scale', left, right)

  def __str__(self):
    return f'scale({self.left}, {self.right})'


class AssignNode(BinarryOps):
  def __init__(self, name, left, right):
    super().__init__(op_name='=', left=left, right=right)
    self.name = name

  def __str__(self):
    return f'{self.left} = {self.right}'


class CompoundNode(Node):
  def __init__(self, name):
    super().__init__(name)
    self.children = []

  @abstractmethod
  def add_node(self, node):
    pass


class GemmListNode(CompoundNode):
  def __init__(self):
    super().__init__(name=f'gemm_list')
    self.appendix = []

  def add_node(self, node):
    if isinstance(node, MatrixNode):
      self.appendix.append(node)
    elif isinstance(node.right, MatrixNode):
      self.appendix.append(node)
    elif isinstance(node.right, GemmNode):
      self.children.append(node)
    else:
      raise ValueError(f'Expected right child to be either GemmNode ro MatrixNode, '
                       f'given {type(node)}')


class StatementsNode(CompoundNode):
  def __init__(self):
    super().__init__(name=f'statements')
    self._children_names = []

  def add_node(self, node):
    if not isinstance(node, AssignNode):
      raise ValueError(f'expected AssignNode, given {type(node)}')

    if node.name in self._children_names:
      raise ValueError(f'name of assignment expr `{node.name}` has already been used')
    else:
      self._children_names.append(node.name)

    self.children.append(node)
