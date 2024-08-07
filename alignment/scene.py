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


def seq_text(seq: str):
    s = mn.Text(seq, font_size=24, font="monospace", t2c={".": mn.BLACK})

    return s


def seq_group(seq: str):
    g = []
    for char in seq:
        s = mn.Square(
            side_length=0.35,
            stroke_width=0.0,
            z_index=0,
        )

        t = mn.Text(
            char,
            font_size=24,
            font="monospace",
            z_index=1
        )

        if char == ".":
            t.set_color(mn.BLACK)

        t.move_to(s.get_center())

        if char == "Q":
            t.shift(mn.DOWN * 0.01)
        elif char == "L":
            t.shift(mn.RIGHT * 0.01)
        # elif char == "G":
        #     t.shift(mn.RIGHT * 0.01)

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


msa_colors = {
    "A": "#b7bc97",
    "C": "#ff5701",
    "D": "#019fb6",
    "E": "#01a47f",
    "F": "#d3aec4",
    "G": "#36c6fe",
    "H": "#2194f5",
    "I": "#faa311",
    "K": "#34d098",
    "L": "#f3aa58",
    "M": "#e1af84",
    "N": "#05cbe5",
    "P": "#80cd02",
    "Q": "#5a9e7e",
    "R": "#03cfb7",
    "S": "#a9beaf",
    "T": "#969357",
    "V": "#bf8609",
    "W": "#ff42c8",
    "Y": "#a687a1",
}

msa_colors = {k: mn.ManimColor.from_hex(msa_colors[k]) for k in msa_colors}
msa_colors["-"] = mn.BLACK


def protein_seqs():
    seqs = [
        "SNDSLCTKCKNNLLVNTDQSYCVCKECECSQEG",
        "MEQYLCLCRHMGLFNAKDSGEGCIDCGSSFPF",
        "ANLRKCKVCNKGKIFNREKMYRRCMFCESVAQY",
        "LDEKICHGCNREILGNWTNQSYRVCQFCGAVFPL",
        "ALNHCICNERASTIVLQQKIDDGQCQDCQSINPK",
        "NDDRHCSGCGGDGLYMTADFYEVCLDCGATFPY",
        "GNASACIVRCHHEEVLNEDRGYLACIECEYSEPT",
        "LALSKCGNCSYDWIIILRDDDREVCSNCGAIFSY",
        "DYWICRDGNHPGLLAEDGSMFCRFCGISHQV",
        "SQDSTCQKCRSNLVMHTTGSYEVCEFCEISQPV",
    ]
    group = mn.VGroup(*[seq_group(seq) for seq in seqs]).arrange(mn.DOWN)

    return group


def protein_ali():
    seqs = [
        "-SNDSLC-TKCKNNLL-VNTDQSYCVCKECECSQEG",
        "-MEQYLC-L-CRHMGL-FNAKDSGEGCIDCGSSFPF",
        "-ANLRKC-KVCNKGKI-FNREKMYRRCMFCESVAQY",
        "-LDEKIC-HGCNREILGNWTNQSYRVCQFCGAVFPL",
        "ALNHCICNERASTIVLQQKIDDGQ--CQDCQSINPK",
        "-NDDRHC-SGCGGDGL-YMTADFYEVCLDCGATFPY",
        "-GNASACIVRCHHEEV-LNEDRGYLACIECEYSEPT",
        "-LALSKC-GNCSYDWIIILRDDDREVCSNCGAIFSY",
        "--DYWIC-RDGNHPGL--LAEDGSMFCRFCGISHQV",
        "-SQDSTC-QKCRSNLV-MHTTGSYEVCEFCEISQPV",
    ]
    group = mn.VGroup(*[seq_group(seq) for seq in seqs]).arrange(mn.DOWN)

    ali_2d = [list(seq) for seq in seqs]

    return group, ali_2d


protein_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I",
                    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]


def residue_axes():
    axes = mn.Axes(
        x_range=[0, 21, 1],
        y_range=[0.0, 1.0, 0.2],
        axis_config={"color": mn.BLUE},
        y_axis_config={
            "include_numbers": True,
            "include_tip": False,
        },
        x_axis_config={
            "include_tip": False,
            "include_ticks": False,
        }
    )

    z = {float(i + 1): mn.Text(r, font="monospace")
         for (i, r) in enumerate(protein_alphabet)}

    axes.get_x_axis().add_labels(z)

    return axes


def residue_distribution():
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5,
            0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9, 0.5, 0.4]

    axes = residue_axes()

    a = axes.get_y_axis().number_to_point(0.0)

    bars = mn.VGroup()
    for i, value in enumerate(data):
        x = i + 1
        b = axes.get_y_axis().number_to_point(value)
        height = b[1] - a[1]
        bar = mn.Rectangle(
            width=0.4,
            height=height,
            fill_color=mn.GREEN,
            fill_opacity=0.7,
            stroke_color=mn.WHITE,
            stroke_width=1,
        ).move_to(axes.c2p(x, value / 2))
        bars.add(bar)

    g = mn.VGroup(axes, bars)

    return g


class Dist(mn.Scene):
    def construct(self):
        plot = residue_distribution()
        self.play(mn.FadeIn(plot))


SECTION = 0


class Msa(mn.Scene):
    section_count = 0

    def section(self):
        self.next_section(skip_animations=self.section_count < SECTION)
        self.section_count += 1

    def construct(self):
        # SECTION 0
        self.section()

        seqs = protein_seqs()
        ali_rows, ali_2d = protein_ali()

        n_seqs = len(ali_2d)
        cols = list(zip(*ali_2d))
        counts = [(n_seqs - col.count("-")) for col in cols]

        self.add(seqs)

        # SECTION 1
        self.section()

        g = mn.AnimationGroup(
            *[mn.TransformMatchingShapes(s, a)
              for (s, a) in zip(seqs, ali_rows)],
        )

        self.play(g, run_time=0.25)
        self.play(mn.ApplyMethod(ali_rows.arrange, mn.DOWN, 0.0))

        # SECTION 2
        self.section()

        g = []
        for seq in ali_rows:
            for char in seq:
                color = msa_colors[char[1].text]
                g.append(mn.ApplyMethod(
                    char[0].set_stroke, mn.WHITE, 1.0, 1.0)
                )

        self.play(mn.AnimationGroup(*g), run_time=0.1)

        for seq in ali_rows:
            for char in seq:
                color = msa_colors[char[1].text]
                g.append(mn.ApplyMethod(char[0].set_fill, color, 0.5))

        self.play(mn.AnimationGroup(*g), run_time=0.1)

        # SECTION 3
        self.section()

        cons_cols = [i for i, c in enumerate(counts) if c / n_seqs >= 0.5]
        nons_cols = [i for i, c in enumerate(counts) if c / n_seqs <= 0.5]

        ali_cols = list(zip(*ali_rows))

        cons_cols = [ali_cols[i] for i in cons_cols]
        nons_cols = [ali_cols[i] for i in nons_cols]

        g = []
        g2 = []
        for col in nons_cols:
            for (s, c) in col:
                g.append(mn.ApplyMethod(s.set_fill, mn.BLACK, 0.0))
                g.append(mn.ApplyMethod(c.set_opacity, 0.25))
                g2.append(mn.FadeOut(s, c))

        self.play(mn.AnimationGroup(*g), run_time=0.1)
        self.play(mn.AnimationGroup(*g2), run_time=0.1)

        # SECTION 4
        self.section()

        cons_cols = mn.VGroup(*[mn.VGroup(*col) for col in cons_cols])

        self.play(mn.ApplyMethod(cons_cols.arrange,
                  mn.RIGHT, 0.0), run_time=0.1)
        self.play(mn.ApplyMethod(cons_cols.to_edge, mn.UP), run_time=0.1)

        # SECTION 5
        self.section()

        axes = residue_axes().scale(0.5).shift(mn.DOWN * 2)
        self.play(mn.FadeIn(axes), run_time=0.1)

        # SECTION 6
        self.section()

        for col in cons_cols[:1]:
            others = mn.VGroup(*[other for other in cons_cols if other != col])
            self.play(
                mn.AnimationGroup(
                    mn.ApplyMethod(col.set_opacity, 1.0),
                    mn.ApplyMethod(others.set_opacity, 0.1),
                ),
                run_time=0.1
            )

            mn.Square().move_to
            last_at = [None for _ in range(21)]
            for row in col:
                residue = row[1].text

                if residue == "-":
                    continue

                idx = protein_alphabet.index(residue) + 1

                last = last_at[idx]

                if last is not None:
                    a = mn.ApplyMethod(row.next_to, last, mn.UP, 0.0)
                else:
                    point = axes.get_x_axis().number_to_point(idx)
                    a = mn.ApplyMethod(row.move_to, point + mn.UP * 0.25)

                self.play(
                    a,
                    run_time=0.1
                )

                last_at[idx] = row
