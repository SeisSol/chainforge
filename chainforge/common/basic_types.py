import enum


class DataFlowDirection(enum.Enum):
  SOURCE = 0
  SINK = 1


class Addressing(enum.Enum):
  NONE = 0
  STRIDED = 1
  PTR_BASED = 2

  @classmethod
  def addr2ptr_type(cls, addr_type):
    map = {Addressing.NONE: '*',
           Addressing.STRIDED: '*',
           Addressing.PTR_BASED: '**'}
    return map[addr_type]

  @classmethod
  def str2addr(cls, string):
    map = {'none': Addressing.NONE,
           'strided': Addressing.STRIDED,
           'pointer_based': Addressing.PTR_BASED}
    if string not in map:
      raise ValueError(f'arg must be either none, strided or pointer_based, given: {string}')
    return map[string]

  @classmethod
  def addr2str(cls, addr):
    map = {Addressing.NONE: 'none',
           Addressing.STRIDED: 'strided',
           Addressing.PTR_BASED: 'pointer_based'}
    return map[addr]


class FloatingPointType(enum.Enum):
  FLOAT = 0
  DOUBLE = 1

  @classmethod
  def as_str(cls, fp):
    map = {FloatingPointType.FLOAT: 'float',
           FloatingPointType.DOUBLE: 'double'}
    return map[fp]

  @classmethod
  def str2enum(cls, as_str: str):
    map = {'float': FloatingPointType.FLOAT,
           'double': FloatingPointType.DOUBLE}
    return map[as_str]


class GeneralLexicon:
  NUM_ELEMENTS = 'numElements'
  EXTRA_OFFSET = '_extraOffset'
  STREAM_PTR_STR = 'streamPtr'
  ALPHA_SYMBOL_NAME = 'alpha'
  BETA_SYMBOL_NAME = 'beta'
