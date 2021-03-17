from .vertex import Vertex
from copy import copy
from typing import List, Dict, Set, Union


class VertexStack:
  def __init__(self):
    self._vertices: List[Vertex] = []

  def add_edges(self, vertex: Vertex):
    self._vertices.append(copy(vertex))

  def pop_edges(self) -> Vertex:
    return self._vertices.pop()

  def print_stack(self) -> None:
    for edge in self._vertices:
      print(edge)

  def empty(self) -> bool:
    return len(self._vertices) == 0


class GraphColoring:
  def __init__(self, graph: List[Vertex], user_objects: List[object]):
    self._graph: List[Vertex] = graph
    self._max_num_colors: int = len(user_objects)
    self._colors: List[int] = [i for i in range(self._max_num_colors)]
    self._allowed_color_set: Set[int] = set(self._colors)
    self._color2object_map: Dict[int, object] = {color: user_object
                                                 for color, user_object
                                                 in zip(self._colors, user_objects)}
    self._stack: VertexStack = VertexStack()
    self._vertex2color_map: Dict[Vertex, Union[int, None]] = {v: None for v in self._graph}

  def apply(self) -> Dict[Vertex, object]:
    self._graph = sorted(self._graph, key=lambda x: x.get_num_neighbours(), reverse=True)

    # run the first part of the algorithm
    while self._coarse_graph():
      pass

    # it is the bottom case i.e., a graph consists of only nodes without edges
    for vertex in self._graph:
      free_color = self._colors[0]
      self._vertex2color_map[vertex] = free_color

    # run the second part of the algorithm
    while not self._stack.empty():
      self._restore_graph_and_color()

    # map Dict[vertex, color] to Dict[vertex, object] as it was required by the user
    vertex2object = {}
    for vertex, color in self._vertex2color_map.items():
      vertex2object[vertex] = self._color2object_map[color]
    return vertex2object

  def print_graph(self) -> None:
    print('~'*80)
    for vertex in self._graph:
      print(vertex)

  def _coarse_graph(self) -> bool:
    for index, vertex in enumerate(self._graph):
      if not vertex.get_neighbors() == set():
        if self._max_num_colors > vertex.get_num_neighbours():
          candidate = self._graph.pop(index)
          self._stack.add_edges(candidate)
          self._remove_edges(candidate)
          return True
    return False

  def _remove_edges(self, vertex) -> None:
    for neighbour in vertex.get_neighbors():
      for item in self._graph:
        if neighbour == item:
          item.remove_neighbour(vertex)

  def _restore_graph_and_color(self) -> None:
    vertex = self._stack.pop_edges()
    self._assign_color(vertex)
    self._add_edges_to_graph(vertex)

  def _assign_color(self, vertex) -> None:
    occupied_colors = set()
    for neighbour in vertex.get_neighbors():
      assigned_color = self._vertex2color_map[neighbour]
      occupied_colors.add(assigned_color)
    free_colors = self._allowed_color_set - occupied_colors
    self._vertex2color_map[vertex] = free_colors.pop()

  def _add_edges_to_graph(self, vertex) -> None:
    for neighbour in vertex.get_neighbors():
      for item in self._graph:
        if item == neighbour:
          item.add_neighbor(vertex)
