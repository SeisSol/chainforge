from .nodes import StatementsNode
from .traversals import ConstantPropagation, DeadNodesElimination
from .traversals import GemmListFolder, PrimaryGemmFolder
from .traversals import AstToList, Printer


class PostProcessor:
  def __init__(self, statements, symbol_table):
    if not isinstance(statements, StatementsNode):
      raise ValueError(f'expected StatementsNode, given {type(statements)}')

    self.stmts = statements
    self.symbol_table = symbol_table

  def process(self, visualize=False):
    gemm_dicts = {}
    for child in self.stmts.children:
      _, ast = ConstantPropagation(self.symbol_table).traverse(child)
      ast = DeadNodesElimination().traverse(ast)
      ast = PrimaryGemmFolder(self.symbol_table).traverse(ast)
      ast = GemmListFolder().traversal(ast)
      if visualize:
        Printer().print(ast=ast, filename=child.name, view=False)

      gemm_list = AstToList(self.symbol_table).convert(ast)
      gemm_dicts[child.name] = gemm_list

    return gemm_dicts
