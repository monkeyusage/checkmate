from dataclasses import dataclass, field
from typing import Optional
import random
import time
import heapq
import math
import sys
import io
import chess
import chess.svg
import cairosvg
import pygame
from numpy.typing import NDArray
from checkmate import play as rust_play


@dataclass
class Bot:
    id: int
    generation: int
    fitness: float
    brain: NDArray


PROMOTION_SQUARES = [*range(8), *range(64-8, 64)]


def maybe_promote(board: chess.Board, move: chess.Move) -> chess.Move:
    # auto promote to queen
    if (
        (piece := board.piece_at(move.from_square)) is not None
        and piece.piece_type == chess.PAWN
        and (move.to_square in PROMOTION_SQUARES)
    ):
        move.promotion = chess.QUEEN
    return move


def get_board_value(board: chess.Board, side: chess.Color) -> float:
    pieces = board.piece_map().values()
    if board.is_checkmate():
        out = 3 
    elif board.is_check():
        out = 2
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
    next_nodes: dict[chess.Move, 'Node'] = field(default_factory=dict, compare=False)
    queue: list['Node'] = field(default_factory=list, compare=False)

    def add_next(self, node: 'Node') -> None:
        self.next_nodes[node.move] = node
        heapq.heappush(self.queue, node)

    def __repr__(self) -> str:
        return f'Node(move:{self.move}, turn:{self.turn}, value:{self.value})'

    def size(self, acc: int = 0) -> int:
        if not self.next_nodes:
            return acc
        for next_node in self.next_nodes.values():
            acc += next_node.size(len([*next_node.next_nodes.values()]))
        return acc


def play(
        board: chess.Board,
        side: chess.Color,
        max_depth: int,
        depth: int,
        prev_node: Node
) -> chess.Move:
    """
    this function recursively fills the move tree
    when going down the leaves back to the root is 'optimizes'
    the node s values depending on the turn
    """
    start = time.perf_counter()
    if depth == max_depth:
        return chess.Move.null()

    for move in board.legal_moves:
        move = maybe_promote(board, move)
        # go further down the tree if
        # current_node value > prev_node value or
        # current node does not exist, create it and compute its value
        node = prev_node.next_nodes.get(move)
        if node is not None and node.value < prev_node.value:
            # the node exists and is useless
            continue
        board.push(move)
        # move has not yet been evaluated add it to the tree
        if node is None:
            node_value = get_board_value(board, side)
            node = Node(turn=board.turn, move=move, value=node_value)
            prev_node.add_next(node)

        # otherwise do not recompute
        play(board, side, max_depth, depth+1, prev_node=node)
        board.pop()

    if depth != 0:
        # since we went through all next moves we can now optimize the node
        if prev_node.next_nodes:
            (best_node, *_) = heapq.nlargest(1, prev_node.queue)
            prev_node.value = best_node.value
        return chess.Move.null()

    # if we have equal moves we chose one randomly
    stop = time.perf_counter()
    max_value = max([node.value for node in prev_node.next_nodes.values()])
    print(f"Evaluated {prev_node.size()} nodes in {stop-start} seconds")
    return random.choice(
        [
            node
            for node in prev_node.next_nodes.values()
            if node.value == max_value
        ]
    ).move


def to_screen(
    board: chess.Board,
    size: int,
    square: Optional[chess.Square] = None
) -> pygame.Surface:
    fillings = {}
    if square is not None and board.piece_at(square) is not None:
        destinations = [move.to_square for move in board.legal_moves if move.from_square == square]
        fillings = {destination: 'red' for destination in destinations}
    buffer = chess.svg.board(board=board, orientation=board.turn, size=size, fill=fillings).encode('utf-8')
    png_buffer: bytes = cairosvg.svg2png(
        bytestring=buffer,
        output_height=size,
        output_width=size
    )  # type: ignore
    return pygame.image.load(io.BytesIO(png_buffer))


def get_square(xpos: int, ypos: int, size: int, turn: chess.Color) -> chess.Square:
    letters = 'abcdefgh'
    letters = letters if turn == chess.WHITE else letters[::-1]
    letter = letters[math.ceil((xpos/size) * 8) - 1]
    number = '12345678'[math.ceil((ypos/size) * 8) - 1]
    return chess.parse_square(letter + number)


def get_user_move(board: chess.Board, clock: pygame.time.Clock, screen_size: int, screen: pygame.Surface) -> chess.Move:
    move_from: chess.Square | None = None
    while True:
        screen_board = to_screen(board, screen_size, move_from)
        screen.blit(screen_board, (0, 0))
        pygame.display.flip()
        events = pygame.event.get(eventtype=pygame.MOUSEBUTTONDOWN)
        quit_game = pygame.event.get(eventtype=pygame.QUIT)
        if quit_game:
            pygame.quit()
            sys.exit()
        if not events:
            clock.tick(60)
            continue
        event, *_ = events
        xpos, ypos = event.pos
        if move_from is None:
            move_from = get_square(xpos, ypos, screen_size, board.turn)
            continue
        move_to = get_square(xpos, ypos, screen_size, board.turn)
        move = chess.Move(move_from, move_to)
        if move not in board.legal_moves:
            move_from = None
            continue
        move = maybe_promote(board, move)
        return move


def main() -> None:
    pygame.init()
    size = 420
    screen = pygame.display.set_mode((size, size))
    clock = pygame.time.Clock()
    board = chess.Board()
    root = Node(0, board.turn, chess.Move.null())
    while board.outcome() is None:
        screen_board = to_screen(board, size)
        screen.blit(screen_board, (0, 0))
        pygame.display.flip()
        start = time.perf_counter()
        move = (
            play(board, board.turn, 4, 0, root)
            if board.turn == chess.WHITE
            else get_user_move(board, clock, size, screen)
        )
        stop = time.perf_counter()
        print(f"Played {move} in {stop-start} seconds")
        board.push(move)
        root = root.next_nodes[move]

    outcome = board.outcome()
    assert outcome is not None
    winner = outcome.winner if outcome.winner is not None else "no one"
    print(f'Outcome of the game is: {winner}')
    pygame.quit()


if __name__ == '__main__':
    main()
