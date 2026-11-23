"""Othello, ported from Pgx (github.com/sotetsuk/pgx, pgx/othello.py).

Pgx plays on a flat `bool[64]` bitboard: a direction is a `jnp.roll` by 1, 7,
8 or 9 plus a hand-written mask against wrapping around the edges, the eight
directions are a `lax.fori_loop` over a table of shifts and masks, and every
decision is a `lax.cond` over lambdas. This port keeps the observations,
rewards and legal-action masks identical (checked in compare.py), and
writes one game on an `i32[Side, Side]` board with plain Python:

* a move walks from the new stone in each direction with a `while` loop and
  turns the opponent's stones it brackets with `board[r, c] = MINE`;
* the legal moves of the next player are rays over the whole board, one `for`
  per direction, with an edge-aware `shift(x, dr, dc)`;
* `if action == PASS:` / `if terminated:` instead of `lax.cond(..., lambda: ...)`.

    python examples/pgx/othello.py            # play random games, print a board
    python -m expex examples/pgx/othello.py   # prove the shapes
"""

import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Batch, Bool, Dim, Key, f32, i32


class Side(Dim, size=8): ...
class Actions(Dim, size=65): ...  # a square (row * 8 + col), or PASS
class Players(Dim, size=2): ...


PASS = 64
MINE, THEIRS, EMPTY = 1, -1, 0  # the board is always seen by the player to move
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1, -1))


@expex.struct
class Othello:
    current_player: i32  # the player id (0 or 1) to move; not the colour
    observation: Bool[Side, Side, Players]  # [..., 0]: stones of the player to move
    rewards: f32[Players]
    terminated: Bool
    truncated: Bool
    legal_action_mask: Bool[Actions]
    step_count: i32
    turn: i32  # the colour to move: 0 (black, moves first) or 1
    board: i32[Side, Side]  # MINE / THEIRS / EMPTY for the player to move
    passed: Bool  # the last move was a pass

    @staticmethod
    def new(key: Key) -> "Othello":
        """The starting position; the player who moves first is drawn at random."""
        board = jnp.zeros((Side, Side), jnp.int32)
        board[3, 4] = MINE
        board[4, 3] = MINE
        board[3, 3] = THEIRS
        board[4, 4] = THEIRS
        game = Othello(
            current_player=jnp.int32(jr.bernoulli(key)),
            observation=jnp.zeros((Side, Side, Players), bool),
            rewards=jnp.zeros(Players),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=legal_moves(board),
            step_count=jnp.int32(0),
            turn=jnp.int32(0),
            board=board,
            passed=jnp.bool_(False),
        )
        return game.replace(observation=game.observe())

    def step(self, action: i32) -> "Othello":
        """Pgx's `Env.step`: a finished game stays finished, an illegal move loses."""
        if self.terminated:
            game = self.replace(rewards=jnp.zeros(Players))
        else:
            game = self.replace(step_count=self.step_count + 1).play(action)
        if not self.legal_action_mask[action]:
            rewards = jnp.ones(Players)
            rewards[self.current_player] = -1.0
            game = game.replace(rewards=rewards, terminated=jnp.bool_(True))
        if game.terminated:  # any action is "legal" once the game is over
            game = game.replace(legal_action_mask=jnp.ones(Actions, bool))
        return game.replace(observation=game.observe())

    def play(self, action: i32) -> "Othello":
        """Place a stone (or pass), turn what it brackets, and hand the board over."""
        board = self.board
        if action != PASS:
            row, col = action // Side, action % Side
            board[row, col] = MINE
            for dr, dc in DIRECTIONS:
                r, c = row + dr, col + dc
                while on_board(r, c) and board[r, c] == THEIRS:
                    r, c = r + dr, c + dc
                if on_board(r, c) and board[r, c] == MINE:  # bracketed: turn them
                    r, c = r - dr, c - dc
                    while r != row or c != col:
                        board[r, c] = MINE
                        r, c = r - dr, c - dc

        mine, theirs = (board == MINE).sum(), (board == THEIRS).sum()
        rewards = jnp.zeros(Players)
        terminated = mine + theirs == Side * Side or theirs == 0 or (self.passed and action == PASS)
        if terminated and mine != theirs:
            rewards = -jnp.ones(Players)
            winner = self.current_player if mine > theirs else 1 - self.current_player
            rewards[winner] = 1.0
        legal = legal_moves(-board)
        return self.replace(
            current_player=1 - self.current_player,
            turn=1 - self.turn,
            board=-board,  # the opponent's view
            passed=action == PASS,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=legal,
        )

    def observe(self) -> Bool[Side, Side, Players]:
        """Two planes: the stones of the player to move, then the opponent's."""
        return jnp.stack([self.board == MINE, self.board == THEIRS], -1)


@expex.fn
def on_board(r: i32, c: i32) -> Bool:
    return 0 <= r < Side and 0 <= c < Side


@expex.fn
def shift(x: Bool[Side, Side], dr: int, dc: int) -> Bool[Side, Side]:
    """`x` moved by (dr, dc) squares; what moves off the board is dropped."""
    padded = jnp.pad(x, 1)
    return padded[1 - dr : 1 - dr + Side, 1 - dc : 1 - dc + Side]


@expex.fn
def legal_moves(board: i32[Side, Side]) -> Bool[Actions]:
    """Where the player to move can play: empty squares that bracket a line of opponent stones."""
    theirs, empty = board == THEIRS, board == EMPTY
    legal = jnp.zeros((Side, Side), bool)
    for dr, dc in DIRECTIONS:
        line = shift(board == MINE, dr, dc) & theirs  # opponent stones next to one of mine
        for _ in range(Side - 3):  # ... and the ones behind them (at most 6 in a row)
            line |= shift(line, dr, dc) & theirs
        legal |= shift(line, dr, dc) & empty
    return jnp.append(legal.ravel(), not legal.any())


@expex.fn(jit=True)
def random_games(key: Key, games: int, moves: int) -> Batch[Othello, games]:
    """`games` games after `moves` uniformly random legal moves each."""
    key, start = jr.split(key)
    state = Othello.new.vectorize(jr.split(start, games))
    for _ in range(moves):
        key, pick = jr.split(key)
        action = jr.categorical(pick, jnp.log(state.legal_action_mask), axis=-1)
        state = state.step.vectorize(action)
    return state


if __name__ == "__main__":
    state = random_games(jr.key(0), 8, 70)
    print(state.board[0], state.rewards, state.terminated, sep="\n")
    print(Othello.play.flow, legal_moves.flow, sep="\n")
