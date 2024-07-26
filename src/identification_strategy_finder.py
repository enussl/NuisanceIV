class AdjustmentSetFinder:
    def __init__(self, dag: nx.DiGraph, X: Set[str], Y: str, hidden_nodes: Set[str] = set(), type_effect: str = "direct", one_suffices: bool = True):
        self.dag = dag
        self.X = X
        self.Y = Y
        self.type_effect = type_effect
        self.hidden_nodes = hidden_nodes
        self.one_suffices = one_suffices
        self.nodes_on_causal_path = set()
        self.descendant_Y = set()
        self.modified_graph = None
        self.original_graph = None
        self.subsets = []
        self.subset_combinations = []

    def get_all_causal_paths(self, source: str, destination: str) -> List[List[str]]:
        """
        Get all causal paths between the provided source and destination.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param source: Source node identifier.
        :param destination: Destination node identifier.
        :return: A list of lists specifying the node identifiers on the path between source and destination (inclusive of
                 the source and destination node identifiers). An empty list is returned if no paths are available.
        """
        assert nx.is_directed_acyclic_graph(self.dag), 'This method only works for DAGs but the current graph is not a DAG.'
        assert source in self.dag.nodes and destination in self.dag.nodes, 'Source or destination node not in graph.'

        if source == destination:
            return []

        return list(nx.all_simple_paths(self.dag, source, destination))
    
    def remove_causal_paths(self):
        """
        Removes nodes on all causal paths from X to Y in the DAG.
        """
        self.modified_graph = self.dag.copy()
        causal_nodes = set()
        
        for node_x in self.X:
            causal_paths_from_x_to_Y = self.get_all_causal_paths(node_x, self.Y)
            for path in causal_paths_from_x_to_Y:
                causal_nodes.update(path)

        for node_x in self.X:
            causal_paths_from_x_to_Y = self.get_all_causal_paths(node_x, self.Y)
            for path in causal_paths_from_x_to_Y:
                self.nodes_on_causal_path.update(path)
                has_incoming_edge = False

                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i+1]

                    if has_incoming_edge:
                        break

                    if current_node != node_x:  
                        incoming_edges = list(self.dag.in_edges(current_node, data=True))
                        for edge in incoming_edges:
                            if edge[0] not in causal_nodes:
                                has_incoming_edge = True
                                break

                    if not has_incoming_edge:
                        if self.modified_graph.has_edge(current_node, next_node):
                            self.modified_graph.remove_edge(current_node, next_node)

        return self.modified_graph
            
    def generate_subsets(self, type_set: str, removed_nodes: Set[str]) -> List[Set[str]]:
        """
        Generates all possible subsets of the remaining nodes in the modified graph.

        Args:
        - type_set: Either "instrument" or "conditional".
        - removed_nodes: Set of nodes that have been removed from the graph.

        Returns:
        - List of all possible subsets of nodes in the graph with size >= |X| if type_set is "instrument",
          otherwise, it returns all possible subsets of nodes in the graph.
        """
        
        nodes = list(self.modified_graph.nodes())
        for cn in self.nodes_on_causal_path.union(removed_nodes).union(self.hidden_nodes):
            nodes.remove(cn)

        if type_set == "instrument":
            subsets = [set(comb) for comb in self.powerset(nodes) if len(comb) >= len(self.X)]
            subsets.append(self.X)
        elif type_set == "conditional":
            subsets = [set(comb) for comb in self.powerset(nodes)]
        else:
            raise ValueError("Invalid type_set. Must be either 'instrument' or 'conditional'.")
        return subsets

    def generate_all_subset_combinations(self):
        """
        Generates all possible subsets of the remaining subsets of the modified graph for each subset.
        """
        self.subsets = self.generate_subsets("instrument", set())
        self.subset_combinations = []

        for subset in self.subsets:
            subgraph = self.modified_graph.copy()
            subgraph.remove_nodes_from(subset)

            removed_nodes = subset
            remaining_subsets = self.generate_subsets("conditional", removed_nodes)
            self.subset_combinations.append(remaining_subsets)
            
        return self.subset_combinations

    def is_d_separator(self, x, y, z, graph: Optional[nx.DiGraph] = None) -> bool:
        """
        Returns whether node sets `x` and `y` are d-separated by `z` in the specified graph.

        Parameters:
        - x : node or set of nodes
            First node or set of nodes.
        - y : node or set of nodes
            Second node or set of nodes.
        - z : node or set of nodes
            Potential separator (set of conditioning nodes). Can be empty set.
        - graph: Directed graph object. If None, the method uses self.modified_graph.

        Returns:
        - bool: True if `x` is d-separated from `y` given `z`.
        """
        graph = graph or self.modified_graph 

        try:
            x = set(x) if isinstance(x, (set, list, tuple)) else {x}
            y = set(y) if isinstance(y, (set, list, tuple)) else {y}
            z = set(z) if isinstance(z, (set, list, tuple)) else {z}

            y = y - x
            z = z - (x | y)

            if not x or not y:
                return True 

            set_v = x | y | z
            if set_v - graph.nodes:
                raise nx.NodeNotFound(f"The node(s) {set_v - graph.nodes} are not found in G")
        except TypeError:
            raise nx.NodeNotFound("One of x, y, or z is not a node or a set of nodes in G")

        if not nx.is_directed_acyclic_graph(graph):
            raise nx.NetworkXError("graph should be directed acyclic")

        forward_deque = deque([])
        forward_visited = set()
        backward_deque = deque(x)
        backward_visited = set()

        ancestors_or_z = set().union(*[nx.ancestors(graph, node) for node in x]) | z | x

        while forward_deque or backward_deque:
            if backward_deque:
                node = backward_deque.popleft()
                backward_visited.add(node)
                if node in y:
                    return False
                if node in z:
                    continue

                backward_deque.extend(graph.pred[node].keys() - backward_visited)
                forward_deque.extend(graph.succ[node].keys() - forward_visited)

            if forward_deque:
                node = forward_deque.popleft()
                forward_visited.add(node)
                if node in y:
                    return False

                if node in ancestors_or_z:
                    backward_deque.extend(graph.pred[node].keys() - backward_visited)
                if node not in z:
                    forward_deque.extend(graph.succ[node].keys() - forward_visited)

        return True
    
    def find_adjustment_sets_conditional(self) -> List[Dict[str, Set[str]]]:
        """
        Tests whether each subset is d-separated from Y given the combination in the modified graph,
        and whether each subset is d-separated from X given the combination in the original graph.

        Returns:
        - List containing sets d-separated from Y in the modified graph and not d-separated from X in the original graph,
          along with their corresponding conditional sets.
        """
        d_separated_sets = []

        for i, subset in enumerate(self.subsets):
            for combination in self.subset_combinations[i]:
                if subset == self.X:
                    is_dsep_from_Y = self.is_d_separator(subset, self.Y, combination, self.modified_graph)
                    if is_dsep_from_Y:
                        d_separated_sets.append({"Instrument Set": subset, "Conditional Set": combination})
                        if self.one_suffices:
                            return d_separated_sets
                else:
                    is_dsep_from_Y = self.is_d_separator(subset, self.Y, combination, self.modified_graph)
                    is_dsep_from_X = self.is_d_separator(subset, self.X, combination, self.dag)

                    if is_dsep_from_Y and not is_dsep_from_X:
                        d_separated_sets.append({"Instrument Set": subset, "Conditional Set": combination})
                        if self.one_suffices:
                            return d_separated_sets
        return d_separated_sets
    
    def find_adjustment_sets_nuisance(self) -> List[Dict[str, Union[Set[str], Dict[str, Set[str]]]]]:
        """
        Finds adjustment sets for nuisance IV.
        
        Returns:
        - List containing adjustment sets for each subset.
        """
        
        all_nodes_except_XY_hidden = set(self.dag.nodes()) - self.X - {self.Y} - self.hidden_nodes
        subsets = list(self.powerset(all_nodes_except_XY_hidden))
        adjustment_sets_result = []
        
        if self.type_effect == "direct":
            only_direct_effects = True
            nodes_on_causal_path = set()
            
            for node_x in self.X: 
                causal_paths_from_x_to_Y = list(nx.all_simple_paths(self.dag, node_x, self.Y))
                for path in causal_paths_from_x_to_Y:
                    nodes_on_causal_path.update(path)
                
            for node in nodes_on_causal_path:
                if node not in self.X and node not in {self.Y}:
                    only_direct_effects = False
                    break
            if not only_direct_effects:
                return []

        if self.type_effect == "total" or only_direct_effects:
            for subset in subsets:
                subset_set = set(subset)
                X = self.X | subset_set

                adjustment_set_finder = AdjustmentSetFinder(self.dag, X, self.Y, self.hidden_nodes, self.type_effect, self.one_suffices)
                adjustment_set_finder.remove_causal_paths()
                adjustment_set_finder.generate_all_subset_combinations()
                adjustment_sets = adjustment_set_finder.find_adjustment_sets_conditional()

                if adjustment_sets:
                    for adj_set in adjustment_sets:
                        if not subset_set.intersection(adj_set["Instrument Set"]):
                            adjustment_set_result = {
                                "Nuisance Set": subset,
                                "Instrument Set": adj_set["Instrument Set"],
                                "Conditional Set": adj_set["Conditional Set"]
                            }
                            if self.one_suffices:
                                return [adjustment_set_result]
                            else:
                                adjustment_sets_result.append(adjustment_set_result)
        return adjustment_sets_result

    def reset(self):
        """
        Reset the internal state of the AdjustmentSetFinder.
        """
        self.nodes_on_causal_path = set()
        self.descendant_Y = set()
        self.modified_graph = None
        self.original_graph = None
        self.subsets = []
        self.subset_combinations = []

    @staticmethod
    def powerset(iterable):
        """
        Generates all possible subsets of the given iterable.

        Args:
        - iterable: Iterable object.

        Returns:
        - Generator yielding all possible subsets.
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))