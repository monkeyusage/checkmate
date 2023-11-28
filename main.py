from typing import Optional
from dataclasses import dataclass, field
from random import choice

import chess


def get_board_value(board: chess.Board, side: chess.Color) -> float:
    pieces = board.piece_map().values()
    if board.is_checkmate() and board.turn != side:
        return 1000
    elif board.is_checkmate() and board.turn == side:
        return 0
    own = sum([piece.piece_type for piece in pieces if piece.color == side])
    other = sum([piece.piece_type for piece in pieces if piece.color != side])
    return own / other

@dataclass
class Node:
    turn: chess.Color
    move: chess.Move
    value: float
    prev_node: Optional['Node']
    next_nodes: list['Node'] = field(default_factory=list)

    def __repr__(self) -> str:
        prev_move = None if self.prev_node is None else self.prev_node.move
        return f'Node(move:{self.move}, turn:{self.turn}, value:{self.value}, prev_move:{prev_move})'

    def optimize(self, side: chess.Color) -> None:
        # optimizing a node consists in getting the max potential value if its our turn
        # other pick the best move of the opponent thus the minimum value
        if not self.next_nodes:
            return
        func = min if self.turn != side else max
        self.value = func([node.value for node in self.next_nodes])

    def size(self, acc: int = 0) -> int:
        if not self.next_nodes:
            return acc
        for next_node in self.next_nodes:
            acc += next_node.size(len(next_node.next_nodes))
        return acc 


def play(board: chess.Board, side: chess.Color, max_depth: int, depth: int, prev_node: Node) -> chess.Move:
    """
    this function recursively fills the move tree
    when going down the leaves back to the root is 'optimizes' the node s values depending on the turn
    """
    if depth == max_depth:
        return chess.Move.null()

    for move in board.legal_moves:
        board.push(move)
        node_value = get_board_value(board, side)
        if node_value < prev_node.value and board.turn == side:
            # do not explore nodes that are just bad...
            board.pop()
            continue
        node = Node(turn=board.turn, move=move, value=node_value, prev_node=prev_node)
        prev_node.next_nodes.append(node)
        play(board, side, max_depth, depth+1, prev_node=node)
        board.pop()

    if depth != 0:
        # since we went through all next moves we can now optimize the node
        prev_node.optimize(side)
        return chess.Move.null()

    assert prev_node.prev_node is None, 'We should be at the start of the filled tree here'
    # if we have equal moves we chose one randomly
    max_value = max([node.value for node in prev_node.next_nodes])
    print(f"Evaluated {prev_node.size()} nodes, can you remove some? I'm exhausted!")
    return choice([node for node in prev_node.next_nodes if node.value == max_value]).move


def get_user_move(board: chess.Board) -> chess.Move:
    while True:
        move_str = input('> ')
        if move_str == 'quit': exit()
        try:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                raise chess.InvalidMoveError()
            return move
        except chess.InvalidMoveError:
            print('Invalid move, enter one of the legal moves')


def main() -> None:
    board = chess.Board()
    while board.outcome() is None:
        # debug, only use 'AI' for white player
        if board.turn == chess.BLACK: print(board)
        root = Node(board.turn, chess.Move.null(), 1, None)
        move = (
            play(board, board.turn, 4, 0, root)
            if board.turn == chess.WHITE
            else get_user_move(board)
        )
        board.push(move)
    outcome = board.outcome()
    assert outcome is not None
    print(f'Outcome of the game is: {outcome.winner if outcome.winner is not None else "no one"}')


if __name__ == '__main__':
    main()
