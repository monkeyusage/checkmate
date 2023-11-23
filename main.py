from random import choice
import chess
import numpy


def assign_value(board: chess.Board, side: chess.Color) -> int:
    return sum([piece.piece_type for piece in board.piece_map().values() if piece.color == side])

def play(board: chess.Board, side: chess.Color) -> chess.Move:
    if side == chess.BLACK:
        return choice([*board.legal_moves])
    # get current value of the board
    current_value = assign_value(board, side)
    next_move = chess.Move.null()
    # check all possible moves and the average value of the board when played
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        new_value = assign_value(board, side)
        board.pop()
        if new_value > current_value:
            next_move = move
        # we want to maximize the attack area and minimize the number of unprotected pieces
    if not next_move:
        return choice([*board.legal_moves])
    return next_move


def main() -> int:
    board = chess.Board()
    while board.outcome() is None:
        move = play(board, board.turn)
        board.push(move)
    outcome = board.outcome()
    assert outcome is not None
    if outcome.winner is None:
        return 0
    elif outcome.winner != chess.WHITE:
        return -1
    return 1



if __name__ == '__main__':
    main()
