from typing import Optional
from dataclasses import dataclass, field
from random import choice
import heapq

import chess


def get_board_value(board: chess.Board, side: chess.Color) -> float:
    pieces = board.piece_map().values()
    if board.is_checkmate():
        out = 1000
    else:
        own = sum([piece.piece_type for piece in pieces if piece.color == side])
        other = sum([piece.piece_type for piece in pieces if piece.color != side])
        out = own / other
    return out if side == board.turn else -out

@dataclass(order=True)
class Node:
    value: float
    turn: chess.Color = field(compare=False)
    move: chess.Move = field(compare=False)
    next_nodes: list['Node'] = field(default_factory=list, compare=False)

    def __repr__(self) -> str:
        return f'Node(move:{self.move}, turn:{self.turn}, value:{self.value})'

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
        node = Node(turn=board.turn, move=move, value=node_value)
        heapq.heappush(prev_node.next_nodes, node)
        play(board, side, max_depth, depth+1, prev_node=node)
        board.pop()

    if depth != 0:
        # since we went through all next moves we can now optimize the node
        if prev_node.next_nodes:
            (best_node, *_) = heapq.nlargest(1, prev_node.next_nodes)
            prev_node.value = best_node.value
        return chess.Move.null()

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
    root = Node(board.turn, chess.Move.null(), 0)
    while board.outcome() is None:
        # debug, only use 'AI' for white player
        if board.turn == chess.BLACK: print(board)
        move = (
            play(board, board.turn, 4, 0, root)
            if board.turn == chess.WHITE
            else get_user_move(board)
        )
        board.push(move)
        for next_node in root.next_nodes:
            if next_node.move == move:
                root = next_node
                break

    outcome = board.outcome()
    assert outcome is not None
    print(f'Outcome of the game is: {outcome.winner if outcome.winner is not None else "no one"}')


if __name__ == '__main__':
    main()
