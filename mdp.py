import argparse
class Node:
    def __init__(self, name, data, discount_factor=0.9, is_minimize=False, tolerance=0.001, iter=100, verbose=True):
        self.name = name
        self.reward = data.get("reward", 0)
        self.edges = data.get("edges", [])
        self.probabilities = data.get("probability")
        self.edge_probabilities = {}
        self.leftover_probability_value = 0
        self.X = 0
        # value and policy iteration
        self.iter = iter
        self.tolerance = tolerance
        self.values = {}  
        self.policy = {} 
        self.discount_factor = discount_factor
        self.is_minimize = is_minimize
        self.is_verbose = verbose
    

    def get_edges(self):
        return self.edges
    
    def get_probability(self):
        return self.probabilities

    def get_reward(self):
        return self.reward
    
    def number_of_valued_probability_elements(self):
        return len(self.probabilities) if self.probabilities else 0

    def valued_probability_elements(self):
        return self.probabilities if self.probabilities else []
    
    def check_nodes_have_attributes(self):
        # Check if the node has any of the required attributes
        has_required_attributes = bool(self.probabilities or self.reward is not None or self.edges)
        # print(f"  Has required attributes: {has_required_attributes}")
        return has_required_attributes
     
    def compute_probability(self):
        # print(f"Debug - Computing probability for Node: {self.name}")
        # Check if probabilities are defined and there are edges
        
        if self.probabilities:
            for edge, prob in zip(self.edges, self.probabilities):
                print(f"  Edge '{edge}': Probability {prob}")
        self.edge_probabilities = {}
        if self.probabilities is not None and len(self.edges) >= 1:
            # print(f"Debug - Node {self.name} has probabilities and edges")
            if len(self.edges) == len(self.probabilities):
                # For Chance Nodes: Equal number of edges and probabilities
                for edge, prob in zip(self.edges, self.probabilities):
                    self.edge_probabilities[edge] = prob
            
            # CALCULATING PROBABILTY VALUES FOR EDGES
            elif len(self.edges) > 1:
                sumX = sum(self.probabilities)
                # print(f"Debug - Sum of probabilities for Node {self.name}: {sumX}")
                self.X = len(self.edges) - len(self.probabilities)
                # print(f"Debug - Number of edges without assigned probabilities for Node {self.name}: {X}")
                self.leftover_probability_value = ((1 - sumX) / self.X) if self.X > 0 else 0
                # print(f"Debug - Node {self.name}: Leftover Probability Value = {self.leftover_probability_value[]}, reward ={self.reward}")
                # Assign primary probability to the first edge
                primary_prob = self.probabilities[0]
                self.edge_probabilities[self.edges[0]] = primary_prob

                # Assign leftover probability to the remaining edges
                for edge in self.edges[1:]:
                    self.edge_probabilities[edge] = float(self.leftover_probability_value)
                # print(f'Debug - Checking inital probabilities Node: {self.name} Edge :{self.edges} probability : {self.probabilities}')   
        else:
            if self.edges:
                if self.reward is None:
                    self.reward = 0
                    # print(f"Debug - Node {self.name} has reward = {self.reward} and probability = {self.probabilities}")

        for edge, prob in self.edge_probabilities.items():
            print(f" Node: {self.name} Edge '{edge}': Probability {prob}")
    # Chance Node, Decision Node, Terminal
    def NodeType(self):
        
        if self.probabilities is not None and len(self.probabilities) == len(self.edges):
            return 'Chance Node'
        # Check for Terminal node conditions
        elif self.edges == [] and self.reward == 0:
            return 'Terminal Node'
        # Check for Decision node conditions
        else:
            return 'Decision Node'
    
    # policy Iteration
    def PolicyIteration(self, node_types):
        end = False
        name_node_types = [x[0] for x in node_types]
        type_node_types = [x[1] for x in node_types]
        edges_node_types = [x[2] for x in node_types]
        reward_node_types = [x[3] for x in node_types]

        #intiiail values of each node = reward
        for index, node_name in enumerate(name_node_types):
            self.values[node_name] = reward_node_types[index]

        while not end:
            old_policy = self.policy.copy()

            for index, node_name in enumerate(name_node_types):
                if type_node_types[index] == 'Decision Node':
                    edges = edges_node_types[index]
                    reward = reward_node_types[index]
                    best_neighbor = self.find_best_edge(node_name, edges, reward, node_types)
                    self.policy[node_name] = best_neighbor

            # Apply Value Iteration with the updated policy
            self.ValueIteration(node_types)

            # Check if the policy has changed
            end = self.is_policy_converged(old_policy)

        # Print final policy and values (Assuming a print function is defined)
        # print(f"final policy: {self.policy} and Value of {edges_node_types}: {self.values}")
    
    # find decision edges
    def find_best_edge(self, node_name, edges, reward, node_types):
        name_node_types = [x[0] for x in node_types]
        edges_node_types = [x[2] for x in node_types]
        # Find the index of the node_name in the list of node names
        node_index = name_node_types.index(node_name)
        # Get the edges for this specific node
        new_edges = edges_node_types[node_index]
        # print(f"Finding best edge for node: {node_name}")
        # print(f"Available edges: {new_edges}")

        # Initialize the best edge
        if not new_edges:
            return None
        best_edge = self.policy.get(node_name, new_edges[0])
        for edge in new_edges:
            if self.is_minimize:
                if self.values[best_edge] > self.values[edge]:
                    best_edge = edge
            else:
                if self.values[best_edge] < self.values[edge]:
                    best_edge = edge
        # print(f"Selected best edge for {node_name}: {best_edge}")
        return best_edge

    def is_policy_converged(self, old_policy):
        for node_name in self.policy:
            if self.policy[node_name] != old_policy.get(node_name, None):
                return False
        return True

    # Value Iteration
    def ValueIteration(self, node_types):
        iter = 0
        tol = False
        name_node_types = [x[0] for x in node_types]
        type_node_types = [x[1] for x in node_types]
        edges_node_types = [x[2] for x in node_types]
        reward_node_types = [x[3] for x in node_types]
        prob_node_types = [x[4] for x in node_types]
        prob_node_types = [list(i.values()) for i in prob_node_types]

        while iter < self.iter and not tol:
            # print(f"Iteration {iter}")
            updated_values = self.values.copy()
            for index, node_name in enumerate(name_node_types):
                # print(f"Node {node_name} ({type_node_types[index]}) current value: {self.values[node_name]}")
                
                # CHANCE NODE 
                if type_node_types[index] == 'Chance Node':
                    reward = reward_node_types[index]
                    edge_value = 0
                    for edge_index, edge in enumerate(edges_node_types[index]):
                        edge_value += prob_node_types[index][edge_index] * self.values[edge]
                    reward += self.discount_factor * edge_value
                
                # DECISION NODE
                elif type_node_types[index] == 'Decision Node':
                    if not edges_node_types[index]:  # Check if the edges list is empty
                        continue
                    # derived_prob = float(self.leftover_probability_value) if self.leftover_probability_value is not None else 0.0
                    chosen_decision = self.policy.get(node_name, edges_node_types[index][0])
                    # given_prob = prob_node_types[index][0]  # Assuming primary probability is the first in the list
                    reward =reward_node_types[index]+ (self.discount_factor * self.values[chosen_decision])
                
                # Update the value of the node
                updated_values[node_name] = reward
                # print(f"Updated value for {node_name}: {updated_values[node_name]}")

            tol = self.is_tolerance(updated_values)
            self.values = updated_values
            iter += 1
        # print("Final values after Value Iteration:", self.values)
    
    # tolerance check
    def is_tolerance(self, updated_values):
        is_converged = True
        for node_name in updated_values:
            diff = abs(updated_values[node_name] - self.values[node_name])
            # print(f"Node {node_name}: Current Value = {self.values[node_name]}, Updated Value = {updated_values[node_name]}, Difference = {diff}")
            if diff > self.tolerance:
                is_converged = False
                # print(f"Tolerance check failed for node {node_name}: Difference {diff} is greater than tolerance {self.tolerance}")

        # if is_converged:
        #     print("All nodes have converged within the tolerance level.")
        # else:
        #     # print("Convergence not yet achieved.")
        
        return is_converged





def parse_input_file(input_file_path):
    input_data = {}
    current_node = None
    with open(input_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue

            if ':' in line:
                node, edges = line.split(':', 1)
                current_node = node.strip()
                input_data[current_node] = {"edges": [edge.strip() for edge in edges.strip().strip('[]').split(',')]}
            elif '%' in line:
                _, probability = line.split('%', 1)
                if current_node:
                    # Split the probability string by spaces and convert each to float
                    input_data[current_node]["probability"] = [float(p.strip()) for p in probability.split()]
            else:
                node, reward = line.split('=', 1)
                input_data[node.strip()] = {"reward": float(reward.strip())}

    return input_data

# Main Execution Function
def main(input_file_path, discount_factor, is_minimize, tolerance, iter):
    input_data = parse_input_file(input_file_path)

    nodes = {name: Node(name, data, discount_factor, is_minimize, tolerance, iter) for name, data in input_data.items()}
    node_types = []
    for node in nodes.values():
        if node.check_nodes_have_attributes():
            node.compute_probability()
            node_type = node.NodeType()
            node_types.append((node.name, node_type, node.get_edges(), node.get_reward(), node.edge_probabilities))
    for name, node in nodes.items():
        node.PolicyIteration(node_types)
    for name, node in nodes.items():
        chosen_edge = node.policy.get(name, "None")
        print(f"{name} -> {chosen_edge}")

    # Print final values for each node
    for name, node in nodes.items():
        value = node.values.get(name, 0)
        print(f"{name}={value:.3f}")

if __name__ == "__main__":
    # input_file_path = "D:/D Drive/college classes/ai/lab/lab3/data/maze example.txt"
    # input_file_path = "D:/D Drive/college classes/ai/lab/lab3/data/publisher decision tree.txt"
    # main(input_file_path)
    parser = argparse.ArgumentParser(description="Running MDP")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("-df", "--discount_factor", type=float, default=0.9, help="Discount factor [0, 1]")
    parser.add_argument("-min", "--minimize", action="store_true", help="Minimize values as costs")
    parser.add_argument("-tol", "--tolerance", type=float, default=0.001, help="Tolerance for exiting value iteration")
    parser.add_argument("-iter", "--iterations", type=int, default=100, help="Cutoff for value iteration")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    args = parser.parse_args()
    main(args.input_file, args.discount_factor, args.minimize, args.tolerance, args.iterations)
        
