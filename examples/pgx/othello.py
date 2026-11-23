"""Othello, ported from Pgx (github.com/sotetsuk/pgx, pgx/othello.py).

Pgx's rules are a bitboard program: each of the eight directions is a
`jnp.roll` by 1, 7, 8 or 9 squares guarded by one of three hand-written
64-entry masks against wrapping around the edges, the directions are a
`lax.fori_loop` over a table of shifts and masks, and every decision is a
`lax.cond` over lambdas. This port keeps the observations, rewards and
legal-action masks identical (checked in compare.py) and the same algorithm,
written for one game with plain Python:

* `for dr, dc in DIRECTIONS:` (staged as one `lax.scan`, like Pgx's
  `fori_loop`) and `if (...).any(): flips |= line` instead of `lax.cond`;
* a direction is a pair (dr, dc) and `shift` works out the edges from it,
  so there are no mask tables;
* sizes by name (`jnp.arange(Cells) // Side`), and a batch of games is
  `Othello.new(keys)` (mapped over the keys), stepped by
  `state.step.vectorize(actions)` (mapping over `self` is explicit).

The board is kept flat, as in Pgx: on CPU, XLA runs the shifts of a
`[games, 64]` array about three times faster than those of a `[games, 8, 8]` one.

    python examples/pgx/othello.py            # play random games, print one
    python -m expex examples/pgx/othello.py   # prove the shapes
"""

import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Batch, Bool, Dim, Key, f32, i32


class Side(Dim, size=8): ...
class Cells(Dim, size=64): ...  # square row * Side + col
class Actions(Dim, size=65): ...  # a square, or PASS
class Players(Dim, size=2): ...


PASS = 64
MINE, THEIRS, EMPTY = 1, -1, 0  # the board is always seen by the player to move
DIRECTIONS = jnp.array([(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1, -1)])


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
    board: i32[Cells]  # MINE / THEIRS / EMPTY, for the player to move
    passed: Bool  # the last move was a pass

    @staticmethod
    def new(key: Key) -> "Othello":
        """The starting position; the player who moves first is drawn at random."""
        board = jnp.zeros((Side, Side), jnp.int32)
        board[3, 4] = MINE
        board[4, 3] = MINE
        board[3, 3] = THEIRS
        board[4, 4] = THEIRS
        board = board.ravel()
        game = Othello(
            current_player=jnp.int32(jr.bernoulli(key)),
            observation=jnp.zeros((Side, Side, Players), bool),
            rewards=jnp.zeros(Players),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=legal_moves(board == MINE, board == THEIRS),
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
        """Place a stone (or pass), turn the lines it closes, and hand the board over."""
        mine, theirs = self.board == MINE, self.board == THEIRS
        placed = jnp.arange(Cells) == action  # nothing for PASS
        flips = jnp.zeros(Cells, bool)
        for dr, dc in DIRECTIONS:
            line = ray(placed, theirs, dr, dc)
            if (shift(line, dr, dc) & mine).any():  # closed by one of mine
                flips |= line
        mine, theirs = mine | placed | flips, theirs & ~flips

        n_mine, n_theirs = mine.sum(), theirs.sum()
        terminated = n_mine + n_theirs == Cells or n_theirs == 0 or (self.passed and action == PASS)
        rewards = jnp.zeros(Players)
        if terminated and n_mine != n_theirs:
            rewards = -jnp.ones(Players)
            winner = self.current_player if n_mine > n_theirs else 1 - self.current_player
            rewards[winner] = 1.0
        return self.replace(
            current_player=1 - self.current_player,
            turn=1 - self.turn,
            board=jnp.int32(theirs) - jnp.int32(mine),  # the opponent's view
            passed=action == PASS,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=legal_moves(theirs, mine),
        )

    def observe(self) -> Bool[Side, Side, Players]:
        """Two planes: the stones of the player to move, then the opponent's."""
        board = self.board.reshape(Side, Side)
        return jnp.stack([board == MINE, board == THEIRS], -1)


@expex.fn
def shift(x: Bool[Cells], dr: i32, dc: i32) -> Bool[Cells]:
    """Every square's value moved (dr, dc) squares; what moves off the board is dropped."""
    row, col = jnp.arange(Cells) // Side, jnp.arange(Cells) % Side
    return jnp.roll(x, dr * Side + dc) & (0 <= row - dr < Side) & (0 <= col - dc < Side)


@expex.fn
def ray(start: Bool[Cells], through: Bool[Cells], dr: i32, dc: i32) -> Bool[Cells]:
    """The squares of `through` reached from `start` by going (dr, dc) over `through` only."""
    line = shift(start, dr, dc) & through
    for _ in range(Side - 3):  # at most 6 stones in a row
        line |= shift(line, dr, dc) & through
    return line


@expex.fn
def legal_moves(mine: Bool[Cells], theirs: Bool[Cells]) -> Bool[Actions]:
    """Empty squares from which a line of `theirs` is closed by one of `mine`; PASS if none."""
    empty = ~(mine | theirs)
    legal = jnp.zeros(Cells, bool)
    for dr, dc in DIRECTIONS:
        legal |= shift(ray(mine, theirs, dr, dc), dr, dc) & empty
    return jnp.append(legal, not legal.any())


@expex.fn(jit=True)
def random_games(key: Key, games: int, moves: int) -> tuple[Batch[Othello, games], f32[games, Players]]:
    """`games` games after `moves` uniformly random legal moves each, and each player's total reward."""
    key, start = jr.split(key)
    state = Othello.new(jr.split(start, games))
    returns = jnp.zeros((games, Players))
    for _ in range(moves):
        key, pick = jr.split(key)
        action = jr.categorical(pick, jnp.log(state.legal_action_mask), axis=-1)
        state = state.step.vectorize(action)
        returns += state.rewards
    return state, returns


if __name__ == "__main__":
    state, returns = random_games(jr.key(0), 1000, 80)
    first = returns[:, 0]
    print(f"1000 random games: player 0 won {(first > 0).sum()}, lost {(first < 0).sum()}, drew {(first == 0).sum()}")
    print("game 0 at the end (x: the player to move, o: the other):")
    for row in state.board[0].reshape(8, 8):
        print(" ", " ".join(".xo"[int(v)] for v in row))
    print("how play() ran:", Othello.play.flow)
