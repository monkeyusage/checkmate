use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::time;
use rand::prelude::*;
use chess::{Board, Color, MoveGen};
use chess::movegen::SquareAndBitBoard;
use pyo3::prelude::*;


#[derive(Clone, Eq, PartialEq)]
struct Node {
    value: usize,
    turn: Color,
    _move: Move,
    next_nodes: HashMap<Move, Node>,
    queue: BinaryHeap<Box<Node>>,
}

impl Node {
    fn new(value: usize, turn: Color, _move: Move) -> Node {
        Node {
            value: value,
            turn: turn,
            _move: _move,
            next_nodes: HashMap::new(),
            queue: BinaryHeap::new(),
        }
    }

    fn add_next(&mut self, node: Node) {
        self.next_nodes.insert(node._move, node);
        self.queue.push(node);
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.turn == other.turn {
            return self.value.cmp(&other.value);
        }
        return self.turn.cmp(&other.turn);
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.turn == other.turn {
            return self.value.partial_cmp(&other.value);
        }
        return self.turn.partial_cmp(&other.turn);
    }
}


fn get_board_value(&board: Board, side: Color) -> f32 {
    let pieces = board.piece_map().values();
    if board.is_checkmate() {
        return 3.0;
    }
    if board.is_check() {
        return 2.0;
    }
    let own: f32 = pieces.clone().filter(|piece| piece.color() == side).map(|piece| piece.piece_type() as f32).sum();
    let other: f32 = pieces.clone().filter(|piece| piece.color() != side).map(|piece| piece.piece_type() as f32).sum();
    let out =  own / other;
    if side == board.side_to_move() { out } else { -out }
}


fn think(&board: Board, side: bool, max_depth: usize, depth: usize, prev_node: Node) -> Option<String> {
    let start = time::Instant::now();
    if depth == max_depth {
        return None;
    }
    let legal_moves = MoveGen::new_legal(&board);

    for _move in legal_moves {
        let new_board = board.make_move_new(_move);
        let node = prev_node.next_nodes.get(&_move);
        if node.is_some() && node.value < prev_node.value {
            continue;
        }
        let node_value = get_board_value(&new_board, side);
        let node = Node {
            value: node_value,
            turn: new_board.side_to_move(),
            _move: _move,
            next_nodes: HashMap::new(),
            queue: BinaryHeap::new(),
        };
        prev_node.add_next(node);
        think(new_board.clone(), side, max_depth, depth+1, prev_node=node);
    }

    if depth != 0 {
        if prev_node.next_nodes.len() > 0 {
            let best_node = prev_node.queue.peek().unwrap();
            prev_node.value = best_node.value;
        }
        return None;
    }

    let stop = time::Instant::now();
    let max_value = prev_node.next_nodes.values().map(|node| node.value).max().unwrap();
    println!("Evaluated {} nodes in {} seconds", prev_node.size(), stop.duration_since(start).as_secs());

    let mut best_moves = Vec::new();
    for node in prev_node.next_nodes.values() {
        if node.value == max_value {
            best_moves.push(node._move);
        }
    }
    let _move = best_moves.choose(&mut rand::thread_rng()).unwrap();
    return Some(_move);
}


#[pyfunction]
fn play(board_fen: String, side: bool, max_depth: usize, depth: usize, prev_node: Node) -> PyResult<String> {
    return think(Board::from_str(&board_fen), side, max_depth, depth, prev_node);
}

#[pymodule]
fn checkmate(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play, m)?)?;
    Ok(())
}
