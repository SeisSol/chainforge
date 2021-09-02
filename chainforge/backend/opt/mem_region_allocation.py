from .abstract import AbstractOptStage, Context
from .coloring import Vertex
from .coloring import GraphColoring
from collections import OrderedDict
from chainforge.backend.symbol import Symbol
from copy import copy
from typing import Dict, Set, Union, List, Tuple


class Region:
  def __init__(self):
    self._items: List[Symbol] = []
    self._counter: int = 0

  def add_item(self, item: Symbol) -> None:
    self._items.append(item)

  def __iter__(self):
    return self._items.__iter__()

  def __contains__(self, item):
    return item in self._items

  def print(self) -> None:
    for item in self._items:
      print(item.name)


class MemoryRegionAllocation(AbstractOptStage):
  def __init__(self, context: Context, live_map):
    super(MemoryRegionAllocation, self).__init__(context)
    
    self._live_map: Dict[int, Set[Symbol]] = live_map
    self._vertex_counter: int = 0
    self._adj_list: List[Vertex] = []
    self._objects2vertices_map: Union[Dict[Symbol, Vertex], None] = None
    self._regions: Union[List[Region], None] = None

  def apply(self) -> None:
    num_regions = MemoryRegionAllocation.compute_num_regions(self._live_map)
    variable_set = self._get_variable_set()
    self._adj_list, self._objects2vertices_map = self._generate_vertices(variable_set)
    self._assign_neighbours()

    self._regions: List[Region] = [Region() for i in range(num_regions)]
    gc = GraphColoring(graph=copy(self._adj_list), user_objects=self._regions)
    coloring_map: Dict[Vertex, object] = gc.apply()

    vertices2obkects = {vertex: name for name, vertex in self._objects2vertices_map.items()}
    for vertex in self._adj_list:
      mem_region = coloring_map[vertex]
      mem_region.add_item(vertices2obkects[vertex])

  def get_regions(self) -> List[Region]:
    return self._regions

  def _get_variable_set(self) -> Dict[Symbol, None]:
    ordered_variable_set = OrderedDict()
    for live_vars in self._live_map.values():
      for var in live_vars:
        ordered_variable_set[var] = None
    return ordered_variable_set

  def _generate_vertices(self, variable_set: Dict[Symbol, None]) -> Tuple[List[Vertex],
                                                                          Dict[Symbol, Vertex]]:
    objects2vertices_map: Dict[Symbol, Vertex] = {}
    vertices: List[Vertex] = []
    for variable in variable_set.keys():
      vertex = Vertex(self._gen_new_vertex_id())
      vertices.append(vertex)
      objects2vertices_map[variable] = vertex
    return vertices, objects2vertices_map

  def _assign_neighbours(self) -> None:
    for live_vars in self._live_map.values():
      if live_vars:
        for var1 in live_vars:
          for var2 in live_vars:
            vertex1 = self._objects2vertices_map[var1]
            vertex2 = self._objects2vertices_map[var2]
            vertex1.add_neighbor(vertex2)

  def _gen_new_vertex_id(self) -> int:
    id = self._vertex_counter
    self._vertex_counter += 1
    return id

  @classmethod
  def compute_num_regions(self, live_map: Dict[int, Set[Symbol]]) -> int:
    num_regions = 0
    for prog_point in live_map.values():
      num_regions = max(num_regions, len(prog_point))
    return num_regions
