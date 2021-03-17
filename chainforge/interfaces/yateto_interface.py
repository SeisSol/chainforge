from chainforge.common import DenseMatrix


class YatetoInterface:
  def __init__(self):
    pass


  @classmethod
  def deduce_bbox(cls, yateto_ranges, mem_layout):
    """Converts yateto memory layout (bounding boxes) and ranges to GemmForge bounding boxes i.e.,
       a box is a list of rows and columns indices where the actual data is located within
       a memory patch and should be computed

    Args:
      yateto_ranges (set[loopRanges]): a range of rows and columns to operate on
      mem_layout (BoundingBox): memory layout given as yateto bounding box

    Returns:
      (list): bounding box in GemmForge format
    """
    first, last = yateto_ranges

    return [first.start - mem_layout[0].start,
            last.start - mem_layout[1].start,
            first.stop - mem_layout[0].start,
            last.stop - mem_layout[1].start]

  @classmethod
  def gen_dense_matrix(cls,
                       yateto_ranges,
                       yateto_memory_layout_bbox,
                       addressing,
                       name,
                       is_tmp):

    chainforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                     mem_layout=yateto_memory_layout_bbox)

    return DenseMatrix(num_rows=yateto_memory_layout_bbox[0].size(),
                       num_cols=yateto_memory_layout_bbox[1].size(),
                       addressing=addressing,
                       bbox=chainforge_bbox,
                       alias=name,
                       is_tmp=is_tmp)
