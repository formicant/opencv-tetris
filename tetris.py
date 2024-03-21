from types import SimpleNamespace
from dataclasses import dataclass
from random import randrange
from time import time
import numpy as np
from numpy import ndarray as Array
import cv2 as cv


FIELD_WIDTH = 10
FIELD_HEIGHT = 20

DISPLAY_SCALE = 32

DROP_INTERVAL = 1 / 60
LINE_REMOVING_INTERVAL = 1 / 20

ROTATIONS = (
    cv.ROTATE_90_COUNTERCLOCKWISE,
    cv.ROTATE_180,
    cv.ROTATE_90_CLOCKWISE,
)

PIECE_SHAPES = (
    np.array([
        [1, 1, 1, 1],
    ]),
    np.array([
        [1, 1, 1],
        [0, 0, 1],
    ]),
    np.array([
        [1, 1, 1],
        [1, 0, 0],
    ]),
    np.array([
        [1, 1, 1],
        [0, 1, 0],
    ]),
    np.array([
        [1, 1, 0],
        [0, 1, 1],
    ]),
    np.array([
        [0, 1, 1],
        [1, 1, 0],
    ]),
    np.array([
        [1, 1],
        [1, 1],
    ]),
)


class Color(SimpleNamespace):
    BORDER = 0.5
    FROZEN = 0.8
    LINE   = 1.0


class Key(SimpleNamespace):
    """ Keyboard keys. """
    NONE   = {-1}
    EXIT   = {27}  # Esc
    LEFT   = {ord('A'), ord('a'), 81}  # Left arrow
    RIGHT  = {ord('D'), ord('d'), 83}  # Right arrow
    ROTATE = {ord('W'), ord('w'), 82}  # Up arrow
    DROP   = {ord('S'), ord('s'), 84}  # Down arrow


@dataclass(frozen=True)
class Piece:
    """ Represents a piece with its position on the field. """
    shape_index: int
    rotation: int  # 0..4, counter-clockwise
    x: int
    y: int


def get_piece_shape(shape_index: int, rotation: int) -> Array:
    """ Returns rotated shape of the piece. """
    shape = PIECE_SHAPES[shape_index]
    if rotation > 0:
        shape = cv.rotate(shape, ROTATIONS[rotation - 1])
    return shape


def compose(field: Array, piece: Piece, color: float=1) -> Array:
    """ Places the piece onto the field and returns the result. """
    piece_shape = get_piece_shape(piece.shape_index, piece.rotation)
    start = np.array([piece.y, piece.x])
    end = start + piece_shape.shape
    composed = field.copy()
    composed[start[0]:end[0], start[1]:end[1]] += piece_shape * color
    return composed


def is_enough_space(field: Array, piece: Piece) -> bool:
    """ Checks whether the piece can be placed onto the field at its position. """
    composed = compose(field, piece)
    return np.max(composed) <= 1


class Tetris:
    fall_interval: float
    field: Array
    piece: Piece|None
    
    def __init__(self, fall_interval: float):
        self.fall_interval = fall_interval
        
        # empty field with a border
        self.field = np.full((FIELD_HEIGHT + 2, FIELD_WIDTH + 2), Color.BORDER)
        self.field[1:FIELD_HEIGHT + 1, 1:FIELD_WIDTH + 1] = 0
        
        self._place_new_piece()
    
    
    def _place_new_piece(self) -> bool:
        """ Creates a new random piece at the top.
            Returns `True` if there is space for the new piece,
            or `False` otherwise.
        """
        shape_index = randrange(len(PIECE_SHAPES))
        rotation = randrange(4)
        _, piece_width = get_piece_shape(shape_index, rotation).shape
        x = FIELD_WIDTH // 2 - (piece_width - 1) // 2
        self.piece = Piece(shape_index, rotation, x, 1)
        return is_enough_space(self.field, self.piece)
    
    
    def _move_piece(self, dx: int=0, dy: int=0, rotate: bool=False) -> bool:
        """ Moves or rotates the piece if it's possible and returns `True`,
            or returns `False` if impossible.
        """
        assert self.piece is not None
        rotation, x, y = self.piece.rotation, self.piece.x, self.piece.y
        if rotate:
            height, width = get_piece_shape(self.piece.shape_index, rotation).shape
            rotation = (rotation + 1) % 4
            shift = (width - 1) // 2 - (height - 1) // 2
            dx += shift
            dy -= shift
        
        piece = Piece(self.piece.shape_index, rotation, x + dx, max(1, y + dy))
        if is_enough_space(self.field, piece):
            self.piece = piece
            return True
        return False
    
    
    def _freeze_piece(self) -> None:
        """ Makes the fallen piece a part of the field. """
        assert self.piece is not None
        self.field = compose(self.field, self.piece, Color.FROZEN)
        self.piece = None
    
    
    def _remove_full_lines(self) -> None:
        """ Finds full lines, highlights them,
            then, moves down everything above them.
        """
        line_minimums = np.min(self.field[1:-1, 1:-1], axis=1)
        full_line_indices = 1 + np.nonzero(line_minimums)[0]
        if len(full_line_indices) == 0:
            return
        
        # highlight full lines
        for y in full_line_indices:
            self.field[y, 1:-1] = Color.LINE
        self._draw_and_wait(LINE_REMOVING_INTERVAL * 2)
        
        # remove full lines one by one
        for y in full_line_indices:
            self.field[2:y + 1, 1:-1] = self.field[1:y, 1:-1]
            self.field[1, 1:-1] = 0
            self._draw_and_wait(LINE_REMOVING_INTERVAL)
    
    
    def _draw_and_wait(self, interval: float) -> int:
        """ Draws a frame.
            Waits for the given time interval or a key press.
            Returns the pressed key code (or `Key.None` if no key has been pressed).
        """
        if self.piece is not None:
            composed = compose(self.field, self.piece)
        else:
            composed = self.field
        scaled = cv.resize(composed, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv.INTER_NEAREST)
        cv.imshow('Tetris', scaled)
        return cv.waitKey(max(1, round(interval * 1000)))
    
    
    def play(self) -> None:
        """ Starts the game. """
        print(f'Game started (Fall interval: {self.fall_interval} s)')
        old_time = time()
        interval = self.fall_interval
        
        while(True):
            key_code = self._draw_and_wait(interval)
            new_time = time()
            interval -= new_time - old_time
            old_time = new_time
            freeze_piece = False
            
            if key_code in Key.LEFT:
                self._move_piece(dx=-1)
            elif key_code in Key.RIGHT:
                self._move_piece(dx=1)
            elif key_code in Key.ROTATE:
                self._move_piece(rotate=True)
            elif key_code in Key.DROP:
                while self._move_piece(dy=1):
                    self._draw_and_wait(DROP_INTERVAL)
                freeze_piece = True
            elif key_code in Key.NONE:
                freeze_piece = not self._move_piece(dy=1)
                interval = self.fall_interval
            elif key_code in Key.EXIT:
                print('Game aborted')
                break
            
            if freeze_piece:
                self._freeze_piece()
                self._remove_full_lines()
                if not self._place_new_piece():
                    print('Game over')
                    break
                old_time = time()
                interval = self.fall_interval


if __name__ == '__main__':
    Tetris(fall_interval=0.5).play()
