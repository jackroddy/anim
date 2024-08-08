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


X_LEN = 12
Y_LEN = 6


def residue_axes(ylim):
    y_step = ylim / 5
    axes = mn.Axes(
        x_length=X_LEN,
        y_length=Y_LEN,
        x_range=[0, 21, 1],
        y_range=[0.0, ylim, y_step],
        axis_config={"color": mn.BLUE},
        y_axis_config={
            "include_numbers": False,
            "include_tip": False,
        },
        x_axis_config={
            "include_tip": False,
            "include_ticks": False,
        }
    )

    s = mn.Rectangle(
        height=Y_LEN + 0.5,
        width=X_LEN + 1.1,
        stroke_width=0.25,
        stroke_opacity=0.0,
    )
    axes.move_to(s.get_center()).shift(mn.RIGHT * 0.55)

    y_ticks = [i * y_step for i in range(1, 6)]

    y_labels = {
        y: mn.MathTex(f"{y:3.2f}")
        for y in y_ticks
    }

    x_labels = {
        float(i + 1): mn.Text(r, font="monospace")
        for (i, r) in enumerate(protein_alphabet)
    }

    axes.get_x_axis().add_labels(x_labels)
    axes.get_y_axis().add_labels(y_labels)

    return mn.VGroup(axes, s)


def background_distribution(ylim=0.12):
    freqs = [
        0.0787945,  # A
        0.0151600,  # C
        0.0535222,  # D
        0.0668298,  # E
        0.0397062,  # F
        0.0695071,  # G
        0.0229198,  # H
        0.0590092,  # I
        0.0594422,  # K
        0.0963728,  # L
        0.0237718,  # M
        0.0414386,  # N
        0.0482904,  # P
        0.0395639,  # Q
        0.0540978,  # R
        0.0683364,  # S
        0.0540687,  # T
        0.0673417,  # V
        0.0114135,  # W
        0.0304133,  # Y
    ]
    axes = residue_axes(ylim)

    y_axis = axes[0].get_y_axis()
    a = y_axis.number_to_point(0.0)

    bars = mn.VGroup()
    for i, value in enumerate(freqs):
        x = i + 1
        b = y_axis.number_to_point(value)
        height = b[1] - a[1]
        bar = mn.Rectangle(
            width=0.4,
            height=height,
            fill_color=mn.GREEN,
            fill_opacity=0.7,
            stroke_color=mn.WHITE,
            stroke_width=1,
        ).move_to(axes[0].c2p(x, value / 2))
        bars.add(bar)

    g = mn.VGroup(axes, bars)

    return g


def background_count_distribution(ylim=0.12):
    freqs = [
        0.0787945,  # A
        0.0151600,  # C
        0.0535222,  # D
        0.0668298,  # E
        0.0397062,  # F
        0.0695071,  # G
        0.0229198,  # H
        0.0590092,  # I
        0.0594422,  # K
        0.0963728,  # L
        0.0237718,  # M
        0.0414386,  # N
        0.0482904,  # P
        0.0395639,  # Q
        0.0540978,  # R
        0.0683364,  # S
        0.0540687,  # T
        0.0673417,  # V
        0.0114135,  # W
        0.0304133,  # Y
    ]
    axes = residue_axes(ylim)

    y_axis = axes[0].get_y_axis()

    a = y_axis.number_to_point(0.0)

    bars = mn.VGroup()
    for i, value in enumerate(freqs):
        x = i + 1
        b = y_axis.number_to_point(value)
        height = b[1] - a[1]
        color = msa_colors[protein_alphabet[i]]
        bar = mn.Rectangle(
            width=0.4,
            height=height,
            fill_color=color,
            fill_opacity=0.7,
            stroke_color=mn.WHITE,
            stroke_width=1,
        ).move_to(axes[0].c2p(x, value / 2))
        bars.add(bar)

    g = mn.VGroup(axes, bars)

    return g


class Dist(mn.Scene):
    def construct(self):
        plot = background_distribution()
        self.play(mn.FadeIn(plot))


SECTION_START = 7
SECTION_END = 100


class Msa(mn.Scene):
    section_count = 0

    def s_play(self, *args, **kwargs):
        self.section()
        self.play(*args, **kwargs)

    def section(self):
        if self.section_count < SECTION_START:
            skip = True
        elif self.section_count > SECTION_END:
            skip = True
        else:
            skip = False

        self.next_section(skip_animations=skip)
        self.section_count += 1

    def construct(self):
        seqs = protein_seqs()
        ali_rows, ali_2d = protein_ali()

        n_seqs = len(ali_2d)
        cols = list(zip(*ali_2d))
        counts = [(n_seqs - col.count("-")) for col in cols]

        # fade in sequences
        self.s_play(mn.FadeIn(seqs))

        # align the sequences
        self.s_play(
            *[
                mn.TransformMatchingShapes(s, a)
                for (s, a) in zip(seqs, ali_rows)
            ],
            run_time=0.25
        )
        self.play(ali_rows.animate.arrange(mn.DOWN, buff=0.0))
        # ---

        squares = mn.VGroup(*[char[0] for row in ali_rows for char in row])
        chars = mn.VGroup(*[char[1] for row in ali_rows for char in row])

        # set ali squares to MSA color
        self.s_play(
            *[
                square.animate.set_stroke(mn.WHITE, width=1.0, opacity=1.0)
                .set_fill(msa_colors[chars[i].text], opacity=0.5)
                for (i, square) in enumerate(squares)
            ],
            run_time=0.1
        )

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

        # uncolor nonsensus cols
        self.s_play(mn.AnimationGroup(*g), run_time=0.1)

        # fadeout nonsensus cols
        self.s_play(mn.AnimationGroup(*g2), run_time=0.1)

        cons_cols = mn.VGroup(*[mn.VGroup(*col) for col in cons_cols])

        # rearrange consensus cols
        self.s_play(
            mn.ApplyMethod(
                cons_cols.arrange,
                mn.RIGHT, 0.0),
            run_time=0.1
        )

        # slide up consensus cols
        self.s_play(mn.ApplyMethod(cons_cols.to_edge, mn.UP), run_time=0.1)

        counts_plot = residue_axes(10)\
            .scale(0.5)\
            .shift(mn.DOWN * 2)

        # add in the distribution plot
        self.s_play(mn.FadeIn(counts_plot), run_time=0.1)

        def lol(col, plot):
            x_axis = plot[0].get_x_axis()
            y_axis = plot[0].get_y_axis()
            [_, size, _] = y_axis.number_to_point(1) -\
                y_axis.number_to_point(0)

            others = mn.VGroup(*[other for other in cons_cols if other != col])
            self.play(
                mn.AnimationGroup(
                    mn.ApplyMethod(col.set_opacity, 1.0),
                    mn.ApplyMethod(others.set_opacity, 0.1),
                ),
                run_time=0.1
            )

            moves = []
            counts = [0 for _ in range(21)]
            for char in col:
                residue = char[1].text

                if residue == "-":
                    moves.append(mn.FadeOut(char))
                    continue

                idx = protein_alphabet.index(residue) + 1

                x = x_axis.number_to_point(idx)[0]
                y = y_axis.number_to_point(counts[idx])[1]

                moves.append(
                    char.animate.scale_to_fit_height(size)
                    .move_to([x, y, 0] + mn.UP * 0.15)
                )

                # MAKE BARS ON THE AXES 0 TALL
                # grow them as we add a letter

                # moves.append(
                #     mn.Transform(char, )
                # )

                counts[idx] += 1

            # move the residues onto the counts plot
            self.s_play(
                mn.Succession(
                    *moves
                ),
                run_time=1.0)

            zero_points = [
                x_axis.number_to_point(i) for (i, cnt)
                in enumerate(counts)
                if cnt == 0
            ]

            arrows = mn.VGroup(
                *[
                    mn.Arrow(
                        start=p + mn.UP * 0.5,
                        end=p,
                        buff=0.1,
                        tip_length=0.5,
                    ).set_color(mn.RED)
                    for p in zero_points[1:]
                ]
            )

            self.s_play(mn.FadeIn(arrows))
            self.s_play(mn.FadeOut(arrows))

            remaining = mn.VGroup(*[c for c in col if c[1].text != "-"])

            return mn.VGroup(remaining, plot)

        plot = lol(cons_cols[0], counts_plot)

        background_probs = background_distribution()\
            .scale(0.5)\
            .shift(mn.DOWN * 2)\
            .to_edge(mn.LEFT)

        self.s_play(
            plot.animate.to_edge(mn.RIGHT),
            mn.FadeIn(background_probs)
        )

        background_counts_a = background_count_distribution(ylim=1)\
            .scale(0.5)\
            .shift(mn.DOWN * 2)\
            .to_edge(mn.LEFT)

        self.s_play(
            mn.Transform(
                background_probs,
                background_counts_a,
                replace_mobject_with_target_in_scene=True,
            )
        )

        background_counts_b = background_count_distribution(ylim=10)\
            .scale(0.5)\
            .shift(mn.DOWN * 2)\
            .to_edge(mn.LEFT)

        self.s_play(
            mn.Transform(
                background_counts_a,
                background_counts_b,
                replace_mobject_with_target_in_scene=True,
            )
        )

        self.wait(2)
