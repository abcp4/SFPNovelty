import custom_mcts 
def default(node,state):
    return [custom_mcts.Node(node, a) for a in state.get_possible_actions()]
