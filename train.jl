import Chess
import Flux


struct Cerberus{T}
  paths::T
end

Cerberus(paths...) = Cerberus(paths)
(model::Cerberus)(x::AbstractArray) = map(f -> f(x), model.paths)

Flux.@layer Cerberus


mutable struct Bot
  neuralnet::Flux.Chain
  wins::Int
end


Bot() = Bot(
  Flux.Chain(
    Flux.Conv((8, 8), 12 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((8, 8), 64 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.Conv((5, 5), 64 => 64, Flux.relu, bias=false, pad=Flux.SamePad()),
    Flux.flatten,
    # output represents all possible moves in chess board for both white and black)
    Cerberus(
      Flux.Dense(64*64 => 64*64, Flux.sigmoid),
      Flux.Dense(64*64 => 1, Flux.tanh)
    )
  ), 0
)

function to_array(board::Chess.Board)::Array{Float16, 4}
  array = zeros(Float16, (8, 8, 12))
  # save board for each piece/color in an array
  for (index, (color, piece)) in enumerate(Iterators.product(
     [Chess.WHITE, Chess.BLACK], 
     [Chess.QUEEN, Chess.KING, Chess.BISHOP, Chess.KNIGHT, Chess.PAWN, Chess.ROOK])
  )
    array[:, :, index] = Chess.toarray(Chess.pieces(board, color, piece), Float16)
  end
  # reshapes for batshsize == 1
  reshape(array, (8, 8, 12, 1))
end

function pick_move(bot::Bot, board::Chess.Board)::Chess.Move
  # get the highest ranked legal prediction
  (policy, (board_value,)) = bot.neuralnet(to_array(board))
  # cancel impossible moves such as a1a1
  predictions[[1:65:4096]...] .= 0
  # sort by highest probability
  moves = sortperm(vec(predictions))
  max_index = length(moves)
  move_index = 1 
  selected_move = Chess.Move(moves[move_index])
  # iterate over moves stop at first highest ranked legal move
  legal_moves = Chess.moves(board)
  while !(selected_move in legal_moves)
    move_index += 1
    @inbounds selected_move = Chess.Move(moves[move_index])
    @assert move_index < max_index "legal_moves: $(legal_moves)"
  end
  selected_move
end

function play!(whites::Bot, blacks::Bot)
  game = Chess.Game()
  while !Chess.isterminal(game)
    board = game.node.board
    move = Chess.sidetomove(board) == Chess.WHITE ? pick_move(whites, board) : pick_move(blacks, board)
    Chess.domove!(game, move)
  end
  if Chess.ischeckmate(game.node.board)
    winner = Chess.sidetomove(game.node.board) == Chess.WHITE ? blacks : whites
    winner.wins += 1
  end
end


function round_robin!(bots::Vector{Bot})
  for (idx, bot) in enumerate(bots)
    for other_idx in 1:length(bots)
      # dont play with yourself
      if idx == other_idx
        continue
      end
      play!(bot, bots[other_idx])
    end
  end
end


function train()
  bots = [Bot() for _ in 1:10]
  while sum(map((b) -> b.wins, bots)) == 0
    round_robin!(bots)
  end
end
