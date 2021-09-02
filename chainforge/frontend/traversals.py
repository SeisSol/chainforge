import os
from graphviz import Digraph
from chainforge.common import generate_tmp_matrix
from chainforge.common import GemmDescr
from .nodes import VarNode, ScalarNode, MatrixNode, DeadNode
from .nodes import BinarryOps, AssignNode, AddNode, MultNode
from .nodes import StatementsNode, GemmListNode
from .nodes import GemmNode, ScaleMatrixNode


class ConstantPropagation:
  def __init__(self, symbol_table):
    self.symbol_table = symbol_table

  def traverse(self, node):
    if isinstance(node, BinarryOps):
      left_float, node.left = self.traverse(node.left)
      right_float, node.right = self.traverse(node.right)
      if isinstance(node, MultNode):
        return left_float * right_float, node
      elif isinstance(node, AddNode):
        self._check_add_node(node, left_float, right_float)
        left = self._gen_scale_matrix_node(left_float, node.left)
        if isinstance(left.right, MultNode):
          node.left = left

        right = self._gen_scale_matrix_node(right_float, node.right)
        if isinstance(right.right, MultNode):
          node.right = right

        return 1.0, node
      elif isinstance(node, AssignNode) and isinstance(node.right, MultNode):
        node.right = self._gen_scale_matrix_node(right_float, node.right)
      return 1.0, node

    if isinstance(node, ScalarNode):
      value = self.symbol_table.find(node.name).descr
      self.symbol_table.remove(node.name)
      return value, DeadNode()

    if isinstance(node, MatrixNode):
      return 1.0, node

  def _gen_scale_matrix_node(self, value, node):
    scalar_name = ScalarNode.gen_next_name()
    self.symbol_table.add(scalar_name, value)
    return ScaleMatrixNode(left=ScalarNode(scalar_name),
                           right=node)

  def _check_add_node(self, node, left_float, right_float):
    if isinstance(node.left, DeadNode):
      raise ValueError(f'found a dead node in assignment: {left_float}')
    if isinstance(node.right, DeadNode):
      raise ValueError(f'found a dead node in `add` with value: {right_float}')


class DeadNodesElimination:
  def traverse(self, node):
    if isinstance(node, ScaleMatrixNode):
      node.right = self.traverse(node.right)
      return node

    if isinstance(node, BinarryOps):
      node.left = self.traverse(node.left)
      node.right = self.traverse(node.right)
      if isinstance(node, MultNode):
        if not (isinstance(node.left, DeadNode) or isinstance(node.right, DeadNode)):
          return node
        elif isinstance(node.left, DeadNode):
          return node.right
        elif isinstance(node.right, DeadNode):
          return node.left
      else:
        return node
    else:
      return node

    return node


class PrimaryGemmFolder:
  def __init__(self, symbol_table):
    self._symbol_table = symbol_table

  def _retrieve_name(self, node):
    if isinstance(node, VarNode):
      return node.name
    elif isinstance(node, GemmNode):
      return node.res.name
    else:
      raise ValueError('expected VarNode and GemmNode')

  def _make_gemm_tmp_res(self, gemm):
    # generate a tmp node i.e. matrix product
    left_attr = self._symbol_table.find(self._retrieve_name(gemm.left))
    right_attr = self._symbol_table.find(self._retrieve_name(gemm.right))
    descr = generate_tmp_matrix(op1=left_attr.descr,
                                op2=right_attr.descr)
    name = GemmNode.make_tmp_name()
    descr.alias = name
    self._symbol_table.add(name=name, descr=descr)
    return MatrixNode(name=name, is_trans=False)

  def traverse(self, node):
    if isinstance(node, BinarryOps):
      node.left = self.traverse(node.left)
      node.right = self.traverse(node.right)

      is_matrix_t1 = isinstance(node.left, (MatrixNode, GemmNode))
      is_matrix_t2 = isinstance(node.right, (MatrixNode, GemmNode))
      if is_matrix_t1 and is_matrix_t2 and isinstance(node, MultNode):
        node = GemmNode(left=node.left,
                        right=node.right,
                        alpha=1.0,
                        beta=0.0)
        node.res = self._make_gemm_tmp_res(node)
      return node
    elif isinstance(node, VarNode):
      return node
    else:
      return node


class GemmListFolder:
  def __init__(self):
    self._children = []

  def traversal(self, node):
    if isinstance(node, AssignNode):
      self.traversal(node.right)

      gemm_list = GemmListNode()
      for child in self._children:
        gemm_list.add_node(child)
      node.right = gemm_list
      return node

    if isinstance(node, AddNode):
      self.traversal(node.left)
      self.traversal(node.right)
      return node

    if isinstance(node, (ScaleMatrixNode, MatrixNode)):
      self._children.append(node)
      return node

    return node


class AstToList:
  def __init__(self, symbol_table):
    self._symbol_table = symbol_table
    self._gemm_list = []
    self._curr_root = None
    self._initial_alpha = 0.0

  def convert(self, ast):
    self._curr_root = ast
    self._check_root(self._curr_root)
    self._initial_alpha = self._compute_initial_alpha(self._curr_root)
    self._to_list(ast)
    return self._gemm_list

  def _check_root(self, root):
    if not isinstance(root, AssignNode):
      raise ValueError(f'expected AssignNode as a root, given {type(root)}')

    if not isinstance(root.left, MatrixNode):
      raise ValueError(f'expected MatrixNode as the left child of the root')

    if not isinstance(root.right, GemmListNode):
      raise ValueError(f'expected GemmListNode as the right child of the root')

  def _compute_initial_alpha(self, root):
    lhs = root.left
    gemm_list = root.right

    value = 0.0
    for item in gemm_list.appendix:
      if isinstance(item, MatrixNode):
        name = item.name
        value += 1.0
      elif isinstance(item, ScaleMatrixNode):
        name = item.right.name
        attr = self._symbol_table.find(item.left.name)
        value += attr.descr
      else:
        raise ValueError(f'expected either MatrixNode or ScaleMatrixNode, given {type(item)}')

      if name != lhs.name:
        raise ValueError(f'cannot add {name} to {lhs.name} using only gemms')
    return value

  def _make_gemm_descr(self, gemm):
    if not isinstance(gemm, GemmNode):
      raise ValueError(f'expected GemmNode, given {type(gemm)}')
    mat_a = self._symbol_table.find(gemm.left.name).descr
    mat_b = self._symbol_table.find(gemm.right.name).descr
    mat_c = self._symbol_table.find(gemm.res.name).descr
    return GemmDescr(trans_a=gemm.left.is_trans,
                     trans_b=gemm.right.is_trans,
                     a=mat_a,
                     b=mat_b,
                     c=mat_c,
                     alpha=gemm.alpha,
                     beta=gemm.beta)

  def _to_list(self, node):
    if isinstance(node, AssignNode):
      self._to_list(node.right)
      return node

    if isinstance(node, GemmListNode):
      for index, child in enumerate(node.children):
        gemm = child.right
        gemm.res = self._curr_root.left
        # adjust alpha and beta for the top gemm of each child
        gemm.alpha = self._symbol_table.find(child.left.name).descr
        gemm.beta = self._initial_alpha if index == 0 else 1.0
        self._to_list(gemm)

      return node

    if isinstance(node, BinarryOps):
      node.left = self._to_list(node.left)
      node.right = self._to_list(node.right)
      if isinstance(node, GemmNode):
        self._gemm_list.append(self._make_gemm_descr(node))
        # return matrix node as a results
        return node.res
      else:
        return node

    return node


class Printer():
  def __init__(self):
    self.dot = Digraph(comment='ast')
    self._counter = 0

  def _new_node_name(self, node):
    self._counter += 1
    name = f'node{self._counter}'
    self.dot.node(name, node.get_label())
    return name

  def _traverse(self, node):
    if isinstance(node, StatementsNode):
      self_name = self._new_node_name(node)
      for child in node.children:
        child_name = self._traverse(child)
        self.dot.edge(self_name, child_name)
      return node

    elif isinstance(node, GemmListNode):
      self_name = self._new_node_name(node)
      for child in node.children:
        child_name = self._traverse(child)
        self.dot.edge(self_name, child_name)
      for item in node.appendix:
        item_name = self._traverse(item)
        self.dot.edge(self_name, item_name)
      return self_name

    elif isinstance(node, BinarryOps):
      name1 = self._traverse(node.left)
      name2 = self._traverse(node.right)
      self_name = self._new_node_name(node)
      self.dot.edge(self_name, name1)
      self.dot.edge(self_name, name2)
      return self_name

    elif isinstance(node, (VarNode, DeadNode)):
      return self._new_node_name(node)

  def print(self, ast, filename, view=True):
    self._traverse(ast)
    cwd = os.getcwd()
    graphs_dir = os.path.join(cwd, '.graphs')
    if not os.path.exists(graphs_dir):
      os.makedirs(graphs_dir)
    self.dot.render(f'{graphs_dir}/{filename}.gv', view=view)
