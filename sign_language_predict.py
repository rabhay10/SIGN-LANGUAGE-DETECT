"""
Shared prediction logic for ASL fingerspelling (8-group CNN + landmark heuristics).
Used by both final_pred.py (Tkinter) and app_streamlit.py.
"""
import math
import numpy as np


def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def predict_single(pts, white, model):
    """
    Given hand landmarks pts (list of [x,y] or [x,y,z]), skeleton image white (400x400x3),
    and the loaded CNN model, returns the detected symbol for this frame.
    Returns: str — one of A-Z, " ", "Speak", "next", "Backspace", or ""
    """
    white_flat = np.array(white, dtype=np.uint8)
    white_flat = white_flat.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(white_flat, verbose=0)[0], dtype="float32")
    ch1 = int(np.argmax(prob, axis=0))
    prob[ch1] = 0
    ch2 = int(np.argmax(prob, axis=0))
    prob[ch2] = 0
    ch3 = int(np.argmax(prob, axis=0))
    prob[ch3] = 0

    # Thumb up = Speak
    if (
        pts[4][1] < pts[3][1]
        and pts[4][1] < pts[5][1] - 30
        and pts[4][1] < pts[9][1] - 30
        and pts[8][1] > pts[6][1]
        and pts[12][1] > pts[10][1]
        and pts[16][1] > pts[14][1]
        and pts[20][1] > pts[18][1]
    ):
        ch1 = "Speak"

    pl = [ch1, ch2]

    # condition for [Aemnst]
    l = [
        [5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
        [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
        [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2],
    ]
    if pl in l:
        if pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]:
            ch1 = 0

    l = [[2, 2], [2, 1]]
    if pl in l:
        if pts[5][0] < pts[4][0]:
            ch1 = 0

    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[0][0] > pts[8][0]
            and pts[0][0] > pts[4][0]
            and pts[0][0] > pts[12][0]
            and pts[0][0] > pts[16][0]
            and pts[0][0] > pts[20][0]
            and pts[5][0] > pts[4][0]
        ):
            ch1 = 2

    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[0][0] < pts[8][0]
            and pts[0][0] < pts[12][0]
            and pts[0][0] < pts[16][0]
            and pts[0][0] < pts[20][0]
        ):
            ch1 = 3

    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            distance(pts[4], pts[11]) > 50
            and pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = 4

    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] < pts[0][0]:
            ch1 = 4

    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[1][0] < pts[12][0]:
            ch1 = 4

    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[10][1]
        ):
            ch1 = 5

    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[4][1] + 17 > pts[8][1]
            and pts[4][1] + 17 > pts[12][1]
            and pts[4][1] + 17 > pts[16][1]
            and pts[4][1] + 17 > pts[20][1]
        ):
            ch1 = 5

    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[0][0] < pts[8][0]
            and pts[0][0] < pts[12][0]
            and pts[0][0] < pts[16][0]
            and pts[0][0] < pts[20][0]
        ):
            ch1 = 5

    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    l = [
        [5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2],
        [5, 0], [6, 3], [6, 4], [7, 5], [7, 2],
    ]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    l = [
        [6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
        [7, 4], [6, 6], [7, 2], [7, 5], [7, 2],
    ]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[2][0] < pts[0][0]
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            distance(pts[4], pts[11]) < 50
            and pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[2][0] < pts[0][0]
            and pts[14][1] < pts[4][1]
        ):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 1

    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = 7

    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[14][1]
        ):
            ch1 = 1

    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (
            pts[0][0] + fg < pts[8][0]
            and pts[0][0] + fg < pts[12][0]
            and pts[0][0] + fg < pts[16][0]
            and pts[0][0] + fg < pts[20][0]
        ) and not (
            pts[0][0] > pts[8][0]
            and pts[0][0] > pts[12][0]
            and pts[0][0] > pts[16][0]
            and pts[0][0] > pts[20][0]
        ) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
        ):
            ch1 = 1

    # --- Map group index to letter ---
    if ch1 == 0:
        ch1 = "S"
        if (
            pts[4][0] < pts[6][0]
            and pts[4][0] < pts[10][0]
            and pts[4][0] < pts[14][0]
            and pts[4][0] < pts[18][0]
        ):
            ch1 = "A"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] < pts[10][0]
            and pts[4][0] < pts[14][0]
            and pts[4][0] < pts[18][0]
            and pts[4][1] < pts[14][1]
            and pts[4][1] < pts[18][1]
        ):
            ch1 = "T"
        if (
            pts[4][1] > pts[8][1]
            and pts[4][1] > pts[12][1]
            and pts[4][1] > pts[16][1]
            and pts[4][1] > pts[20][1]
        ):
            ch1 = "E"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] > pts[10][0]
            and pts[4][0] > pts[14][0]
            and pts[4][1] < pts[18][1]
        ):
            ch1 = "M"
        if (
            pts[4][0] > pts[6][0]
            and pts[4][0] > pts[10][0]
            and pts[4][1] < pts[18][1]
            and pts[4][1] < pts[14][1]
        ):
            ch1 = "N"

    if ch1 == 2:
        ch1 = "C" if distance(pts[12], pts[4]) > 42 else "O"

    if ch1 == 3:
        ch1 = "G" if distance(pts[8], pts[12]) > 72 else "H"

    if ch1 == 7:
        ch1 = "Y" if distance(pts[8], pts[4]) > 42 else "J"

    if ch1 == 4:
        ch1 = "L"

    if ch1 == 6:
        ch1 = "X"

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            ch1 = "Z" if pts[8][1] < pts[5][1] else "Q"
        else:
            ch1 = "P"

    if ch1 == 1:
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "B"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "D"
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "F"
        if (
            pts[6][1] < pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "I"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "W"
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] < pts[9][1]
        ):
            ch1 = "K"
        if (
            (distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8
            and pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "U"
        if (
            (distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8
            and pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
            and pts[4][1] > pts[9][1]
        ):
            ch1 = "V"
        if (
            pts[8][0] > pts[12][0]
            and pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] < pts[20][1]
        ):
            ch1 = "R"

    if ch1 in (1, "E", "S", "X", "Y", "B"):
        if (
            pts[6][1] > pts[8][1]
            and pts[10][1] < pts[12][1]
            and pts[14][1] < pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = " "

    # Next gesture
    if ch1 in ("E", "Y", "B", "Speak", "A", "S", "T", "M", "N"):
        if (pts[4][0] < pts[5][0]) and (
            pts[6][1] > pts[8][1]
            and pts[10][1] > pts[12][1]
            and pts[14][1] > pts[16][1]
            and pts[18][1] > pts[20][1]
        ):
            ch1 = "next"

    # Backspace
    if (
        pts[0][0] > pts[8][0]
        and pts[0][0] > pts[12][0]
        and pts[0][0] > pts[16][0]
        and pts[0][0] > pts[20][0]
        and pts[4][1] < pts[8][1]
        and pts[4][1] < pts[12][1]
        and pts[4][1] < pts[16][1]
        and pts[4][1] < pts[20][1]
        and pts[4][1] < pts[6][1]
        and pts[4][1] < pts[10][1]
        and pts[4][1] < pts[14][1]
        and pts[4][1] < pts[18][1]
    ):
        ch1 = "Backspace"

    if not isinstance(ch1, str):
        ch1 = ""

    return ch1
