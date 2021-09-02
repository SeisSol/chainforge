from chainforge.backend.exceptions import GenerationError
from chainforge.common import Addressing, DataFlowDirection
from typing import Union, List


class Matrix():
  def __init__(self,
               num_rows: int,
               num_cols: int,
               addressing: Addressing,
               bbox: Union[List[int], None]=None,
               alias: Union[str, None]=None,
               is_tmp: bool = False):

    self.name = None
    self.alias = alias
    self.num_rows = num_rows
    self.num_cols = num_cols
    self.is_tmp = is_tmp
    self.direction: Union[DataFlowDirection, None] = None

    if bbox is not None:
      self.bbox = bbox

      # check whether bbox was given correctly
      coords = [coord for coord in self.bbox]
      if (self.num_rows < self.get_actual_num_rows()) or (self.num_cols < self.get_actual_num_cols()):
        raise GenerationError('Matrix size {}x{} is '
                              'smaller than bbox {}'.format(self.num_rows,
                                                            self.num_cols,
                                                            coords))

      if (self.num_rows < self.bbox[2]) or (self.num_cols < self.bbox[3]):
        raise GenerationError('Bbox {} is '
                              'outside of Matrix {}x{}'.format(coords,
                                                               self.num_rows,
                                                               self.num_cols))

    else:
      self.bbox = (0, 0, num_rows - 1, num_cols - 1)

    if isinstance(addressing, Addressing):
      self.addressing = addressing
      self.ptr_type = Addressing.addr2ptr_type(self.addressing)
    else:
      raise ValueError(f'Invalid matrix addressing type, given: {type(addressing)}')

  def set_data_flow_direction(self, direction: DataFlowDirection):
    self.direction = direction

  def get_actual_num_rows(self):
    return self.bbox[2] - self.bbox[0]

  def get_actual_num_cols(self):
    return self.bbox[3] - self.bbox[1]

  def get_actual_volume(self):
    return self.get_actual_num_rows() * self.get_actual_num_cols()

  def get_real_volume(self):
    return self.num_rows * self.num_cols

  def get_offset_to_first_element(self):
    return self.num_rows * self.bbox[1] + self.bbox[0]

  def _set_name(self, name):
    self.name = name

  def is_similar(self, other):
    is_similar = self.num_rows == other.num_rows and self.num_cols == other.num_cols
    is_similar &= self.addressing == other.addressing
    for item1, item2 in zip(self.bbox, other.bbox):
      is_similar &= item1 == item2
    return is_similar

  def is_same(self, other):
    return self.is_similar(other) and self.alias == other.alias and self.is_tmp == other.is_tmp

  def __str__(self):
    return self.name

  def gen_descr(self):
    string = f'{self.name} = {{'
    string += f'rows: {self.num_rows}, '
    string += f'cols: {self.num_cols}, '
    string += f'addr: {Addressing.addr2str(self.addressing)}, '
    string += f'bbox: {self.bbox}'
    string += f'}};'
    return string


class DenseMatrix(Matrix):
  def __init__(self,
               num_rows,
               num_cols,
               addressing,
               bbox=None,
               alias=None,
               is_tmp=False):
    super(DenseMatrix, self).__init__(num_rows,
                                      num_cols,
                                      addressing,
                                      bbox,
                                      alias,
                                      is_tmp)
