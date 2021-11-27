from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

def get_vect_rect(x, y, n, scale, color):
    if n % 2 == 0:
        n = n + 1
    else:
        n = n
    s = scale
    w = 0.9 * s
    h = 0.9 * s
    x0 = x - w / 2
    y0 = y - h / 2
    l = []
    for i in range(n):
        # three dots
        if i == n // 2:
            c1 = Circle((x0 + i * s + w / 2, y0 + h / 2), radius=0.05 * s, facecolor='black')
            c2 = Circle((x0 + i * s + w / 2 - 0.3 * s, y0 + h / 2), radius=0.05 * s, facecolor='black')
            c3 = Circle((x0 + i * s + w / 2 + 0.3 * s, y0 + h / 2), radius=0.05 * s, facecolor='black')
            l.append(c1)
            l.append(c2)
            l.append(c3)
        else:
            rect = Rectangle((x0 + i * s, y0), w, h, facecolor=color, alpha=0.8, edgecolor=None)
            l.append(rect)
    # left bracket
    x1 = x - w / 2 - 0.3 * s
    y1 = y - 0.75 * h
    b1 = Rectangle((x1, y1), 0.15 * w, h * 1.5, facecolor='black', edgecolor=None)
    b2 = Rectangle((x1, y1), 0.35 * w, h * 0.1, facecolor='black', edgecolor=None)
    b3 = Rectangle((x1, y1 + 1.5 * h - 0.1 * h), 0.35 * w, h * 0.1, facecolor='black', edgecolor=None)
    bracket_left = [b1, b2, b3]
    # right bracket
    x2 = x + w / 2 + 0.3 * s - 0.15 * w + (n - 1) * s
    y2 = y - 0.75 * h
    b1 = Rectangle((x2, y2), 0.15 * w, h * 1.5, facecolor='black', edgecolor=None)
    b2 = Rectangle((x2 + 0.15 * w - 0.35 * w, y2), 0.35 * w, h * 0.1, facecolor='black', edgecolor=None)
    b3 = Rectangle((x2 + 0.15 * w - 0.35 * w, y2 + 1.5 * h - 0.1 * h), 0.35 * w, h * 0.1, facecolor='black',
                   edgecolor=None)
    bracket_right = [b1, b2, b3]

    return l + bracket_left + bracket_right



def get_matrix_rect(x, y, rows, cols, scale, color, rows_hide, cols_hide, alpha):
    if rows_hide is None:
        rows_hide = []
    if cols_hide is None:
        cols_hide = []

    s = scale
    w = 0.85 * s
    h = 0.85 * s
    x0 = x - w / 2
    y0 = y - h / 2
    l = []
    for i in range(rows):
        for j in range(cols):
            if i in rows_hide or j in cols_hide:
                rect = Rectangle((x0 + j * s, y0 + i * s), w, h, alpha=0)
            else:
                rect = Rectangle((x0 + j * s, y0 + i * s), w, h, facecolor=color, edgecolor=None, alpha=alpha)
            l.append(rect)
    # dots
    dots1 = []
    dots2 = []
    dots3 = []
    ys1 = []
    if rows_hide:
        x1 = x
        x2 = x + (cols-1)*s
        y1 = y + s*min(rows_hide)
        y2 = y + s*max(rows_hide)
        xs1 = [x1, x2]
        ys1 = [y1, (y1 + y2) / 2, y2]
        dots1 = [Circle((i, j), 0.1*s, facecolor='black', edgecolor=None) for i in xs1 for j in ys1]

    xs2 = []
    if cols_hide:
        x1 = x + s * min(cols_hide)
        x2 = x + s * max(cols_hide)
        y1 = y
        y2 = y + (rows-1)*s
        xs2 = [x1, (x1+x2)/2,  x2]
        ys2 = [y1, y2]
        dots2 = [Circle((i, j), 0.1*s, facecolor='black', edgecolor=None) for i in xs2 for j in ys2]

    if ys1 and xs2:
        dots3 = [Circle((i, j), 0.1*s, facecolor='black', edgecolor=None) for i, j in zip(xs2, ys1)]

    return l + dots1 + dots2 + dots3

class Feature_Matrix:
    def __init__(self, x, y, rows, cols, scale=1, color='#cee4cc', rows_hide=None, cols_hide=None, alpha=0.9):
        self.x = x
        self.y = y
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self.color = color
        if rows_hide is None:
            self.rows_hide = []
        else:
            self.rows_hide = rows_hide
        if cols_hide is None:
            self.cols_hide = []
        else:
            self.cols_hide = cols_hide
        self.data = get_matrix_rect(x, y, rows, cols, scale, color, rows_hide, cols_hide, alpha)

    def set_row_color(self, ind, color):
        start = ind * self.cols
        end = (1 + ind) * self.cols

        for i in range(start, end):
            if i not in self.cols_hide:
                self.data[i].set_facecolor(color)

    def plot(self, ax):
        for e in self.data:
            ax.add_patch(e)


class Feature_Vector:
    def __init__(self, x, y, n=7, scale=1, color='#1f77b4'):
        self.n = n
        self.color = color
        self.position = (x, y)
        self.x = x
        self.y = y
        self.scale = scale
        self.data = get_vect_rect(x, y, n, scale, color)

    def plot(self, ax):
        for e in self.data:
            ax.add_patch(e)


