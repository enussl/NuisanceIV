from .identification_strategy_finder import AdjustmentSetFinder
from .simulate_and_estimate import civ, simulate_linear_SCM

class BackdoorAdjustmentSetFinder:
    def __init__(self, graph: nx.DiGraph, X: Set[str], Y: str, hidden_nodes: Set[str] = set()):
        """
        Initializes the BackdoorAdjustmentSetFinder class.

        Args:
        - graph: Directed graph (DiGraph) representing the causal structure.
        - X: Set of treatment variables.
        - Y: Outcome variable.
        - hidden_nodes: Set of hidden nodes in the graph.
        """
        self.graph = graph
        self.X = X
        self.Y = Y
        self.hidden_nodes = hidden_nodes
        self.type_effect = 'total'
        self.longest_paths = self.find_longest_paths_directed_towards_Y()
        self.backdoor_nodes = self.find_backdoor_nodes()
        self.adjustment_sets = []

    def dfs_paths(self, graph, start, end, path=None):
        """
        Finds all paths from start node to end node using Depth-First Search (DFS).

        Args:
        - graph: Directed graph (DiGraph).
        - start: Starting node for the paths.
        - end: Ending node for the paths.
        - path: Current path being traversed.

        Returns:
        - List of all paths from start to end.
        """
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in graph.predecessors(start):
            if node not in path:
                new_paths = self.dfs_paths(graph, node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def find_causal_paths(self):
        """
        Finds all causal paths from treatment variables X to the outcome variable Y.

        Returns:
        - List of causal paths.
        """
        causal_paths = []
        if self.type_effect == 'direct':
            for x in self.X:
                if self.graph.has_edge(x, self.Y):
                    causal_paths.append([x, self.Y])
        elif self.type_effect == 'total':
            for x in self.X:
                paths = self.dfs_paths(self.graph, self.Y, x)
                for path in paths:
                    causal_paths.append(path[::-1])
        return causal_paths

    def is_subpath(self, smaller_path, larger_path):
        """
        Checks if a smaller path is a subpath of a larger path.

        Args:
        - smaller_path: Path to check if it is a subpath.
        - larger_path: Path to check against.

        Returns:
        - True if smaller_path is a subpath of larger_path, False otherwise.
        """
        len_smaller = len(smaller_path)
        len_larger = len(larger_path)
        if len_smaller >= len_larger:
            return False
        for i in range(len_larger - len_smaller + 1):
            if larger_path[i:i+len_smaller] == smaller_path:
                return True
        return False

    def path_contains_edge(self, path, edge):
        """
        Checks if a path contains a specific edge.

        Args:
        - path: Path to check.
        - edge: Edge to check for.

        Returns:
        - True if path contains edge, False otherwise.
        """
        for i in range(len(path) - 1):
            if (path[i], path[i+1]) == edge:
                return True
        return False

    def find_longest_paths_directed_towards_Y(self):
        """
        Finds the longest paths directed towards the outcome variable Y.

        Returns:
        - List of longest paths directed towards Y.
        """
        all_paths = []
        for node in self.graph.nodes():
            if node != self.Y and self.graph.nodes[node].get('observed', True):
                new_paths = self.dfs_paths(self.graph, self.Y, node)
                all_paths.extend(new_paths)

        filtered_paths = []
        all_paths.sort(key=len, reverse=True)
        for path in all_paths:
            is_nested = False
            for filtered_path in filtered_paths:
                if self.is_subpath(path, filtered_path):
                    is_nested = True
                    break
            if not is_nested:
                filtered_paths.append(path)
        
        causal_paths = self.find_causal_paths()
        
        final_paths = []
        for path in filtered_paths:
            include_path = True
            for causal_path in causal_paths:
                for i in range(len(causal_path) - 1):
                    if self.path_contains_edge(path[::-1], (causal_path[i], causal_path[i+1])):  
                        include_path = False
                        break
                if not include_path:
                    break
            if include_path:
                final_paths.append(path)

        return final_paths
    
    def find_backdoor_nodes(self):
        """
        Finds backdoor nodes based on the longest paths directed towards Y.

        Returns:
        - Set of backdoor nodes.
        """
        backdoor_nodes = set()
        for path in self.longest_paths:
            for node in path[1:-1]:
                if node not in self.hidden_nodes:
                    for predecessor in self.graph.predecessors(node):
                        if predecessor in self.hidden_nodes:
                            if self.graph.has_edge(predecessor, self.Y):
                                backdoor_nodes.add(node)
                                break
        return backdoor_nodes

    def powerset(self, iterable):
        """
        Generates all possible subsets of the given iterable, excluding the empty set.

        Args:
        - iterable: Iterable object.

        Returns:
        - Generator yielding all possible subsets excluding the empty set.
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    
    def remove_first_edges(self, backdoor_node: str) -> None:
        """
        Removes the first edges on all causal paths from the backdoor node to Y.

        Args:
        - backdoor_node: The backdoor node whose causal paths' first edges are to be removed.
        """
        paths = self.dfs_paths(self.graph, self.Y, backdoor_node)
        for path in paths:
            self.graph.remove_edge(path[1], path[0])
    
    def find_adjustment_sets_for_backdoor_nodes(self) -> Dict[str, Union[Set[str], List[Dict[str, Union[Set[str], Dict[str, Set[str]]]]], nx.DiGraph]]:
        """
        Finds adjustment sets for each backdoor node.

        Returns:
        - Dictionary containing backdoor sets, their corresponding adjustment sets, and the adjusted graph and the identification
          strategy on the adjusted graph.
        """
        valid_backdoor_nodes = set()
        
        for subset in list(self.powerset(self.backdoor_nodes)):
            adjustment_finder = AdjustmentSetFinder(self.graph, set(subset), self.Y, self.hidden_nodes, self.type_effect, one_suffices=True)
            adjustment_sets = adjustment_finder.find_adjustment_sets_nuisance()
            if adjustment_sets:
                valid_backdoor_nodes.update(subset)
                self.adjustment_sets.append({"Backdoor Set": set(subset),
                                             "Nuisance Set": adjustment_sets[0]["Nuisance Set"],
                                             "Instrument Set": adjustment_sets[0]["Instrument Set"],
                                             "Conditional Set": adjustment_sets[0]["Conditional Set"]})
        
        largest_adjustment_set = max(self.adjustment_sets, key=lambda x: len(x["Backdoor Set"]))
        
        for backdoor_node in largest_adjustment_set["Backdoor Set"]:
            self.remove_first_edges(backdoor_node)
            
        final_adjustment_finder = AdjustmentSetFinder(self.graph, self.X, self.Y, self.hidden_nodes, self.type_effect, one_suffices=True)
        final_adjustment_sets = final_adjustment_finder.find_adjustment_sets_nuisance()
            
        return self.adjustment_sets, self.graph, final_adjustment_sets
    
class EstimateIterativeNIV:
    def __init__(self, data: pd.DataFrame, X: Set[str], Y: str, gba_strategy: Dict[str, Set[str]], final_strategy: Dict[str, Set[str]], W: np.ndarray):
        """
        Initializes the EstimateIterativeNIV class.

        Args:
        - data: DataFrame containing the data.
        - X: Set of treatment variables.
        - Y: Outcome variable.
        - gba_strategy: Dictionary containing the generalized backdoor adjustment strategy.
        - final_strategy: Dictionary containing the final strategy.
        - W: Weight matrix for the calculation of the sandwich covariance.
        """
        self.data = data
        self.X = X
        self.Y = Y
        self.gba_strategy = gba_strategy
        self.final_strategy = final_strategy
        self.W = W

    def estimate_nuisance(self, strategy_info: Dict[str, Set[str]]) -> np.ndarray:
        """
        Estimates the nuisance parameters using the given strategy information.

        Args:
        - strategy_info: Dictionary containing the strategy information.

        Returns:
        - Estimated nuisance parameters as a numpy array.
        """
        X = np.array(self.data[list(strategy_info["Backdoor Set"])])
        Y = np.array(self.data[self.Y])
        N = np.array(self.data[list(strategy_info["Nuisance Set"])]) if strategy_info["Nuisance Set"] else None
        B = np.array(self.data[list(strategy_info["Conditional Set"])]) if strategy_info["Conditional Set"] else None
        I = np.array(self.data[list(strategy_info["Instrument Set"])]) if strategy_info["Instrument Set"] else None
        
        eta = civ(X = X, Y = Y, N = N, B = B, I = I)
        return eta
    
    def calculate_Y_projected(self, Y: np.ndarray, eta: np.ndarray, backdoor_vars: List[str]) -> np.ndarray:
        """
        Calculates the projected outcome variable Y using the estimated nuisance parameters.

        Args:
        - Y: Original outcome variable as a numpy array.
        - eta: Estimated nuisance parameters as a numpy array.
        - backdoor_vars: List of backdoor variables.

        Returns:
        - Projected outcome variable Y_proj as a numpy array.
        """
        X = np.array(self.data[backdoor_vars])
        Y_proj = Y.copy()
        for i in range(X.shape[1]):
            Y_proj -= eta[i] * X[:, i]
        return Y_proj

    @staticmethod
    def calculate_sandwich_covariance(X: np.ndarray, residuals: np.ndarray, I: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Calculates the sandwich covariance matrix.

        Args:
        - X: Treatment variables as a numpy array.
        - residuals: Residuals from the model as a numpy array.
        - I: Instrument variables as a numpy array.
        - W: Weight matrix for the calculation.

        Returns:
        - Sandwich covariance matrix as a numpy array.
        """
        n, k = X.shape
        
        Q_XI = np.dot(X.T, I) / n
        Omega = np.zeros((I.shape[1], I.shape[1]))
        for i in range(n):
            Ii = I[i, :].reshape(-1, 1)
            Omega += np.dot(Ii, Ii.T) * (residuals[i] ** 2)
        Omega /= n
        
        Q_XI_W_Q_XI_T_inv = np.linalg.inv(np.dot(np.dot(Q_XI, W), Q_XI.T))
        filling = np.dot(np.dot(Q_XI, W), np.dot(Omega, np.dot(W, Q_XI.T)))
        
        sandwich_cov = np.dot(Q_XI_W_Q_XI_T_inv, np.dot(filling, Q_XI_W_Q_XI_T_inv)) / n
        return sandwich_cov

    def estimate_results(self, G: nx.DiGraph, n_samples: int, type_effect: str, hidden_nodes: Set[str], one_suffices: bool) -> Tuple[Dict[str, Dict[str, Union[np.ndarray, str]]], Dict[str, Dict[str, float]], Dict[str, Dict[str, Union[float, str]]]]:
        """
        Estimates the causal effect iteratively.

        Args:
        - G: Directed graph representing the causal structure.
        - n_samples: Number of samples for simulation.
        - type_effect: Type of effect to estimate ('direct' or 'total').
        - hidden_nodes: Set of hidden nodes in the graph.
        - one_suffices: Whether one valid backdoor set suffices.

        Returns:
        - Tuple containing the results dictionary, coefficients dictionary, and best results dictionary.
        """
        results_dict = {}
        data, coefficients = simulate_linear_SCM(G, n_samples)
        
        adjustment_set_finder = AdjustmentSetFinder(G, self.X, self.Y, hidden_nodes, type_effect, one_suffices)
        adjustment_sets = adjustment_set_finder.find_adjustment_sets_nuisance()

        for i, item in enumerate(adjustment_sets):
            I = list(item['Instrument Set']) if item['Instrument Set'] else None
            B = list(item['Conditional Set']) if item['Conditional Set'] else None
            if item['Nuisance Set']:
                N = list(item['Nuisance Set'])
            else:
                N = None
        
            N_data = np.array(data.loc[:, N]) if N is not None else None
            B_data = np.array(data.loc[:, B]) if B is not None else None
            I_data = np.array(data.loc[:, I]) if I is not None else None
            
            X_data = np.array(data.loc[:, list(self.X)])
            Y_data = np.array(data.loc[:, self.Y])
            
            result = civ(Y=Y_data, X=X_data, N=N_data, B=B_data, I=I_data)
            
            result_key = f"N:{N}, B:{B}, I:{I}"
            results_dict[result_key] = {
                'coefficients': result,
                'adjustment_set': result_key
            }
            
            if N_data is not None:
                XN_data = np.hstack((X_data, N_data))
            else:
                XN_data = X_data
            
            residuals = Y_data.copy()
            for j in range(XN_data.shape[1]):
                residuals -= result[j] * XN_data[:, j]
                
            I_dim = I_data.shape[1]
            
            sandwich_cov = self.calculate_sandwich_covariance(XN_data, residuals, I_data, self.W)
            standard_errors = np.sqrt(np.diag(sandwich_cov))
            
            results_dict[result_key]['standard_errors'] = standard_errors

        if type_effect == "total":
            total_causal_effect = 0
            for node_x in self.X: 
                causal_paths_from_x_to_Y = list(nx.all_simple_paths(G, node_x, self.Y))
                for path in causal_paths_from_x_to_Y:
                    path_coefficients = 1 

                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        edge_coefficient = coefficients[edge[1]][edge[0]] 
                        path_coefficients *= edge_coefficient    
                    total_causal_effect += path_coefficients
                
        best_results = {}
        for i in range(len(list(self.X))):
            min_se = float('inf')
            best_key = None
            for key, value in results_dict.items():
                if value['standard_errors'][i] < min_se:
                    min_se = value['standard_errors'][i]
                    best_key = key
            if best_key:
                best_results[f'Best result for X_{i+1}'] = {
                    'coefficient': results_dict[best_key]['coefficients'][i],
                    'standard_error': results_dict[best_key]['standard_errors'][i],
                    'adjustment_set': results_dict[best_key]['adjustment_set']
                }
            
        for key, value in results_dict.items():
            print(f"Adjustment Set: {key}")
            if type_effect == "direct":
                print(f"Population Coefficients: {coefficients[self.Y][list(self.X)[0]]}")
                
            elif type_effect == "total":
                print(f"Population Coefficients: {total_causal_effect}")
                
            else:
                raise ValueError("Invalid type_effect: must either be direct or total")

            if 'N:None' not in key:
                n_number = key.split(":")[1].split(",")[0].strip("[]")
                if n_number != 'None':
                    if n_number.isdigit():
                        n_column = int(n_number)
                    else:
                        n_column = n_number.strip("'") 
                        
                    if n_column in coefficients[self.Y]:
                        print(f"Nuisance Population Coefficients: {coefficients[self.Y][n_column]}")
                    else:
                        print(f"Nuisance Population Coefficients: Column '{n_column}' has no direct effect on Y")
                    
            for i, coef in enumerate(value['coefficients'].flatten()):
                print(f"Coefficient {i+1}: {coef}")
                print(f"Standard Error {i+1}: {value['standard_errors'][i]}")
            print()
            
        if not results_dict:
            print("The effect is not identifiable, I am sorry!")

        return results_dict, coefficients, best_results