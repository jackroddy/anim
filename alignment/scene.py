import manim as mn
from manim import *
import numpy as np
import random

EMIT_TIP_LENGTH = 0.25
EMIT_STROKE_WIDTH = 3.0
EMIT_TIP_SHAPE = mn.ArrowTriangleTip

P_F_TO_B = 0.1
P_F_TO_F = 1.0 - P_F_TO_B
P_B_TO_F = 0.1
P_B_TO_B = 1.0 - P_B_TO_F

P_FAIR_H = 0.5
P_FAIR_T = 0.5

P_BIAS_H = 0.2
P_BIAS_T = 0.8


class CoinHmm(mn.Scene):
    def construct(self):

        fair_pos = [-2.0, 0.0, 0.0]
        bias_pos = [2.0, 0.0, 0.0]

        fair_state = mn.Circle().shift(fair_pos)
        bias_state = mn.Circle().shift(bias_pos)

        fair_text = mn.Text("Fair").scale(0.5).shift(fair_pos)
        bias_text = mn.Text("Biased").scale(0.5).shift(bias_pos)

        fair_emit_table = mn.MathTable(
            [[f"P(H|Fair) = {P_FAIR_H}"],
             [f"P(T|Fair) = {P_FAIR_T}"]],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": mn.YELLOW}
        )\
            .scale(0.5)\
            .next_to(fair_state, mn.DOWN)

        bias_emit_table = mn.MathTable(
            [[f"P(H|Biased) = {P_BIAS_H}"],
             [f"P(T|Biased) = {P_BIAS_T}"]],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": mn.YELLOW}
        )\
            .scale(0.5)\
            .next_to(bias_state, mn.DOWN)

        fair_to_fair = mn.CurvedArrow(
            fair_state.get_left(),
            fair_state.get_top(),
            angle=-180 * mn.DEGREES
        )

        fair_to_bias = mn.CurvedArrow(
            fair_state.get_right(),
            bias_state.get_left()
        )

        bias_to_bias = mn.CurvedArrow(
            bias_state.get_right(),
            bias_state.get_top(),
            angle=180 * mn.DEGREES
        )

        bias_to_fair = mn.CurvedArrow(
            bias_state.get_left(),
            fair_state.get_right()
        )

        fair_to_fair_text = mn.MathTex(f"{P_F_TO_F}")\
            .scale(0.5)\
            .next_to(fair_to_fair, mn.UL, buff=-0.1)

        fair_to_bias_text = mn.MathTex(f"{P_F_TO_B}")\
            .scale(0.5)\
            .next_to(fair_to_bias, mn.DOWN)

        bias_to_bias_text = mn.MathTex(f"{P_B_TO_B}")\
            .scale(0.5)\
            .next_to(bias_to_bias, mn.UR, buff=-0.1)

        bias_to_fair_text = mn.MathTex(f"{P_B_TO_F}")\
            .scale(0.5)\
            .next_to(bias_to_fair, mn.UP)

        state_group = mn.Group(
            fair_state,
            bias_state,
            fair_text,
            bias_text
        )

        trans_arrows_group = mn.Group(
            fair_to_fair,
            fair_to_bias,
            bias_to_bias,
            bias_to_fair,
        )

        trans_probs_group = mn.Group(
            fair_to_fair_text,
            fair_to_bias_text,
            bias_to_bias_text,
            bias_to_fair_text
        )

        emit_table_group = mn.Group(
            fair_emit_table,
            bias_emit_table
        )

        hmm_group = mn.Group(
            state_group,
            trans_arrows_group,
            trans_probs_group,
            emit_table_group
        )
        self.play(mn.FadeIn(hmm_group))


class CoinFlip(mn.ThreeDScene):
    def construct(self):
        coin = mn.Circle(
            radius=2,
            fill_color=mn.BLUE,
            fill_opacity=1.0,
            stroke_color=mn.BLUE,
            stroke_opacity=1.0,
        )\
            .set_shade_in_3d(True)\


        self.set_camera_orientation(phi=0 * mn.DEGREES, theta=0 * mn.DEGREES)

        top_text = mn.Text("H")\
            .scale(2.0)\
            .shift([0.0, 0.0, -0.051])\
            .rotate(90 * mn.DEGREES)\
            .set_shade_in_3d(True)

        bottom_text = mn.Text("T")\
            .scale(2.0)\
            .shift([0.0, 0.0, 0.051])\
            .rotate(90 * mn.DEGREES)\
            .set_shade_in_3d(True)

        heads_text = mn.Text("H")\
            .scale(2.0)\
            .shift([0.0, 0.0, 0.051])\
            .rotate(90 * mn.DEGREES)\
            .set_shade_in_3d(True)

        tails_text = mn.Text("T")\
            .scale(2.0)\
            .shift([0.0, 0.0, 0.051])\
            .rotate(90 * mn.DEGREES)\
            .set_shade_in_3d(True)

        coin_group = mn.VGroup(
            coin, top_text, bottom_text, heads_text, tails_text)

        self.add(coin_group)

        p1 = 0.5
        p2 = 0.2

        n_flips = 10
        random.seed(420)

        for _ in range(n_flips):
            roll = random.random()

            angle = mn.PI * 16.0

            tails_text.set_opacity(0.0)
            heads_text.set_opacity(0.0)
            top_text.set_opacity(1.0)
            bottom_text.set_opacity(1.0)

            self.play(
                mn.Rotate(
                    coin_group,
                    angle,
                    about_point=mn.ORIGIN,
                    axis=mn.RIGHT,
                    rate_func=mn.rush_into
                ),
                run_time=0.5
            )

            top_text.set_opacity(0.0)
            bottom_text.set_opacity(0.0)

            if roll >= p1:
                tails_text.set_opacity(1.0)
                heads_text.set_opacity(0.0)
            else:
                tails_text.set_opacity(0.0)
                heads_text.set_opacity(1.0)

            self.pause(1)


def needleman_wunsch(seq1, seq2, match_score=1, gap_cost=1):
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    traceback_matrix = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(1, m + 1):
        score_matrix[i, 0] = score_matrix[i - 1, 0] - gap_cost
    for j in range(1, n + 1):
        score_matrix[0, j] = score_matrix[0, j - 1] - gap_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1, j - 1] + \
                (match_score if seq1[i - 1] == seq2[j - 1] else -match_score)
            delete = score_matrix[i - 1, j] - gap_cost
            insert = score_matrix[i, j - 1] - gap_cost
            score_matrix[i, j] = max(match, delete, insert)

            if score_matrix[i, j] == match:
                traceback_matrix[i, j] = 1
            elif score_matrix[i, j] == delete:
                traceback_matrix[i, j] = 2
            elif score_matrix[i, j] == insert:
                traceback_matrix[i, j] = 3

    aligned_seq1 = []
    aligned_seq2 = []
    i, j = m, n
    while i > 0 or j > 0:
        if traceback_matrix[i, j] == 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback_matrix[i, j] == 2:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append('-')
            i -= 1
        elif traceback_matrix[i, j] == 3:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))

    mid = []
    for (a, b) in zip(aligned_seq1, aligned_seq2):
        if a == b:
            mid.append("|")
        else:
            mid.append(".")

    mid = "".join(mid)

    return aligned_seq1, aligned_seq2, mid, score_matrix


def seq_group(seq: str):
    g = []
    for char in seq:
        s = mn.Square(
            side_length=0.25,
            stroke_width=0.0,
        )

        if char != ".":
            t = mn.Text(char, font_size=24, font="monospace")
        else:
            t = mn.Text(char, font_size=24, font="monospace", color=mn.BLACK)

        g.append(mn.VGroup(s, t))

    group = mn.VGroup(*g)\
        .arrange(mn.RIGHT, buff=0.0)\
        .center()

    return group


def seqs():
    distance = 0.35
    seq1 = "GATCATTATCTTCTATTTGA"
    seq2 = "GGTCATCACTGAAGATGTGC"

    ali1, ali2, mid, mat = needleman_wunsch(seq1, seq2)

    s1 = seq_group(seq1).shift(mn.UP * distance)
    s2 = seq_group(seq2).shift(mn.DOWN * distance)

    a1 = seq_group(ali1).shift(mn.UP * distance)
    a2 = seq_group(ali2).shift(mn.DOWN * distance)
    m = seq_group(mid)

    return s1, s2, a1, a2, m, mat


class Alignment(mn.Scene):
    def construct(self):
        s1, s2, a1, a2, m, _ = seqs()

        g1 = mn.Group(s1, s2)
        self.play(mn.FadeIn(g1))

        g2 = mn.AnimationGroup(
            mn.TransformMatchingShapes(s1, a1),
            mn.TransformMatchingShapes(s2, a2),
            mn.FadeIn(m),
        )

        self.play(g2)


class AlignmentDp(mn.Scene):
    def construct(self):

        s1, s2, _, _, _, mat = seqs()

        g1 = mn.Group(s1, s2)
        self.play(mn.FadeIn(g1))

        h = self.camera.frame_height * 0.75

        n_cols = len(s1) + 1
        n_rows = len(s2) + 1

        z = h / n_cols

        rows = [[] for _ in range(n_rows)]
        cols = [[] for _ in range(n_cols)]
        cells = []

        for row in range(n_rows):
            for col in range(n_cols):
                group = mn.VGroup(
                    mn.Square(
                        side_length=z,
                        stroke_width=3.0,
                        stroke_opacity=1.0
                    ),
                    mn.Text(str(mat[row][col]), font_size=8)
                )

                rows[row].append(group)
                cols[col].append(group)
                cells.append(group)

        boxes = mn.VGroup(*cells)
        boxes.arrange_in_grid(rows=n_cols, buff=0.0)

        for (cell1, char1, cell2, char2) in zip(rows[0][1:], s1, cols[0][1:], s2):

            self.play(
                mn.AnimationGroup(
                    mn.ApplyMethod(
                        char1.next_to,
                        cell1,
                        mn.UP
                    ),
                    mn.ApplyMethod(
                        char2.next_to,
                        cell2,
                        mn.LEFT
                    )
                ),
                run_time=0.1
            )

        self.play(mn.FadeIn(boxes))


class Msa(mn.Scene):
    def construct(self):
        pass
