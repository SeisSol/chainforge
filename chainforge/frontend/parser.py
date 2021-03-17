from lark import Lark
from lark import Transformer
from .nodes import ScalarNode, MatrixNode, DeadNode
from .nodes import AssignNode, AddNode, MultNode
from .nodes import StatementsNode
from .symbol_table import SymbolTable, ObjType
from .aux import VarFactory


class CustomTransformer(Transformer):
  def __init__(self):
    super().__init__()
    self.symbol_table = SymbolTable()
    self.root = StatementsNode()

  def assign(self, items):
    node = AssignNode(name=items[2].value,
                      left=self.create_var_node(items[0]),
                      right=items[1])
    if isinstance(node.left, ScalarNode):
      raise ValueError('lhs of an expression must be a matrix, given scalar')
    return node

  def add(self, items):
    return AddNode(left=items[0], right=items[1])

  def single_term(self, term):
    (term,) = term
    return term

  def multiply(self, items):
    return MultNode(left=items[0], right=items[1])

  def parentheses(self, items):
    (items, ) = items
    return items

  def single_factor(self, factor):
    (factor,) = factor
    return factor

  def id(self, var_name):
    (var_name,) = var_name
    return self.create_var_node(var_name)

  def id_trans(self, var_name):
    (var_name,) = var_name
    return self.create_var_node(var_name, True)

  def def_immediate_scalar(self, immediate_scalar):
    (immediate_scalar, ) = immediate_scalar
    node = ScalarNode(ScalarNode.gen_next_name())
    self.symbol_table.add(name=node.name,
                          descr=float(immediate_scalar))
    return node

  def create_var_node(self, name, is_trans=False):
    var = self.symbol_table.find(name)
    if not var:
      raise ValueError(f'symbol {name} was not defined')
    return ScalarNode(name) if var.obj_type == ObjType.SCALAR else MatrixNode(name, is_trans)


  def def_create_scalar(self, token):
    (token,) = token
    self.symbol_table.add(name=token.children[0].value,
                          descr=float(token.children[1].value))

    return DeadNode()

  def def_create_matrix(self, token):
    (token, ) = token

    # Note: matrix has been received as a list of tuples (see below)
    matrix_dict = {}
    parsed_desc = token.children[1]
    for item in parsed_desc.children:
      key, value = item
      matrix_dict[key] = value
    matrix_dict['name'] = token.children[0].value

    # populate symbol table with a matrix
    name, descr = VarFactory.produce_matrix(matrix_dict)
    self.symbol_table.add(name, descr)

    return DeadNode()

  def def_num_rows(self, token):
    (token, ) = token
    return 'num_rows', int(token.value)

  def def_num_cols(self, token):
    (token, ) = token
    return 'num_cols', int(token.value)

  def def_addressing(self, mode):
    (mode, ) = mode
    allowed = ['none', 'strided', 'pointer_based']
    if not mode in allowed:
      allowed_str = ', '.join(allowed)
      raise ValueError(f'allowed addr. modes are {allowed_str}. Given: {mode}')
    return 'addressing', mode.value

  def def_int_list(self, tokens):
    (tokens, ) = tokens
    numbers = [int(token) for token in tokens.children]
    return 'bbox', numbers

  def fold(self, assign_node):
    (assign_node, ) = assign_node
    self.root.add_node(assign_node)
    return self.root


class Parser:
  def __init__(self):
    self.calc_parser = Lark(r"""
      prog : construct*
           
      construct : definitions
                | assign                               -> fold
      
      ?definitions : matrix_definition                 -> def_create_matrix
                   | scalar_definition                 -> def_create_scalar
      
      matrix_definition : STRING "=" matrix_desc ";"  
      scalar_definition : STRING "=" SIGNED_FLOAT ";"
      
      matrix_desc : "{" pair ("," pair)~3 "}"
      pair : "rows" ":" INT                           -> def_num_rows
           | "cols" ":" INT                           -> def_num_cols
           | "addr" ":" STRING                        -> def_addressing
           | "bbox" ":" list                          -> def_int_list
                
      list : "[" INT ("," INT)~3 "]"
           
      assign : STRING "=" expr kernel_name ";"         -> assign
      
      ?kernel_name: "->" STRING
      
      expr : term "+" expr                             -> add
           | term                                      -> single_term
      
      term : factor "*" term                           -> multiply
           | factor                                    -> single_factor
      
      factor : "(" expr ")"                            -> parentheses
             | STRING                                  -> id
             | STRING"^""T"                            -> id_trans
             | SIGNED_FLOAT                            -> def_immediate_scalar
      
      STRING : /[a-zA-Z0-9_.-]{2,}/
      COMMENT : "#" /[^\n]/*
      
      %import common.WORD
      %import common.INT
      %import common.SIGNED_FLOAT
      %import common.WS
      
      %ignore WS
      %ignore COMMENT
      """, start='prog', parser='lalr')

  def parse(self, translation_unit):
    tree = self.calc_parser.parse(translation_unit)
    transformer = CustomTransformer()
    transformer.transform(tree)
    return transformer.root, transformer.symbol_table
