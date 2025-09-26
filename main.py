import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
import copy
import matplotlib.cm as cm

# ---------- Geometry helpers ----------
def regular_ngon_vertices(n):
    angles = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(angles), np.sin(angles)]

def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def in_arc(start, end, pos, n):
    start, end, pos = start % n, end % n, pos % n
    if start <= end:
        return start <= pos <= end
    else:
        return pos >= start or pos <= end

# ---------- Game class ----------
class PolygonGame:
    def __init__(self, n):
        if n < 6:
            raise ValueError("n must be at least 6")
        self.n = n
        self.vertices = regular_ngon_vertices(n)
        self.counters = [0, 1, 2, 3]
        self.selected = None
        self.player = 1
        self.winner = None
        self.history = []  # move history stack

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect("equal")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_plot()
        plt.show()

    # --- State saving & undo --- #
    def save_state(self):
        self.history.append((
            copy.deepcopy(self.counters),
            self.player,
            self.winner
        ))

    def undo(self):
        if not self.history:
            print("Nothing to undo.")
            return
        self.counters, self.player, self.winner = self.history.pop()
        self.selected = None
        print("Undo performed.")
        self.update_plot()

    # --- Helpers --- #
    def cyclic_order(self, counters):
        return sorted(counters)

    def current_area(self):
        cyc = self.cyclic_order(self.counters)
        pts = self.vertices[cyc]
        return polygon_area(pts)

    def is_legal_move(self, counter_vertex, target):
        if target in self.counters:
            return False, "Target already occupied."
        cyc = self.cyclic_order(self.counters)
        pos = cyc.index(counter_vertex)
        left = cyc[(pos - 1) % 4]
        right = cyc[(pos + 1) % 4]
        if not in_arc(left, right, target, self.n):
            return False, "Leapfrog not allowed."
        old_area = self.current_area()
        new_counters = self.counters.copy()
        new_counters[self.counters.index(counter_vertex)] = target
        new_area = polygon_area(self.vertices[np.array(new_counters)])
        if new_area <= old_area + 1e-9:
            return False, "Area must increase."
        return True, ""

    def legal_targets(self, counter_vertex):
        return [v for v in range(self.n) if self.is_legal_move(counter_vertex, v)[0]]

    def has_any_legal_move(self):
        for cv in self.counters:
            if self.legal_targets(cv):
                return True
        return False

    # --- Plotting --- #
    def update_plot(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        verts = self.vertices

        # Draw polygon
        self.ax.plot(np.r_[verts[:,0], verts[0,0]], np.r_[verts[:,1], verts[0,1]], "k-", lw=1)
        self.ax.scatter(verts[:,0], verts[:,1], c="lightgray", s=20, zorder=1)

        # Quadrilateral edges
        quad_order = self.cyclic_order(self.counters)
        quad_coords = verts[quad_order]
        self.ax.plot(np.r_[quad_coords[:,0], quad_coords[0,0]],
                     np.r_[quad_coords[:,1], quad_coords[0,1]], "b-", lw=2, zorder=2)

        # Counters
        for i, v_idx in enumerate(self.counters):
            self.ax.scatter(verts[v_idx,0], verts[v_idx,1], c="red", s=100, zorder=4)
            self.ax.text(verts[v_idx,0]*1.08, verts[v_idx,1]*1.08, f"C{i}\n({v_idx})",
                         ha="center", fontsize=9, zorder=5)

        # If a counter selected
        if self.selected is not None and self.winner is None:
            sel = self.selected
            legal = self.legal_targets(sel)

            # Compute resulting areas for all legal moves
            area_map = {}
            for mv in legal:
                new_counters = self.counters.copy()
                new_counters[self.counters.index(sel)] = mv
                new_area = polygon_area(self.vertices[np.array(new_counters)])
                area_map[mv] = new_area
            unique_areas = sorted(set(area_map.values()))
            cmap = cm.get_cmap("tab10", len(unique_areas))
            area_to_color = {a: cmap(i) for i, a in enumerate(unique_areas)}

            # Plot legal moves with color by resulting area
            for mv in legal:
                color = area_to_color[area_map[mv]]
                self.ax.scatter(verts[mv,0], verts[mv,1],
                                color=color, s=90, zorder=3)

            # Triangle splitting
            cyc = self.cyclic_order(self.counters)
            pos = cyc.index(sel)
            left = cyc[(pos - 1) % 4]
            right = cyc[(pos + 1) % 4]
            opposite = cyc[(pos + 2) % 4]
            triA = verts[[left, sel, right]]
            triB = verts[[left, right, opposite]]
            self.ax.add_patch(MplPolygon(triA, closed=True, facecolor="C0", alpha=0.18, zorder=1))
            self.ax.add_patch(MplPolygon(triB, closed=True, facecolor="C1", alpha=0.18, zorder=1))
            self.ax.plot([verts[left,0], verts[right,0]],
                         [verts[left,1], verts[right,1]], "g--", lw=2, zorder=2)
            areaA = polygon_area(triA)
            areaB = polygon_area(triB)
            self.ax.set_title(f"Player {self.player}'s turn | Total area={areaA+areaB:.4f} | Tri areas=({areaA:.4f},{areaB:.4f})")
        else:
            if self.winner is None:
                self.ax.set_title(f"Player {self.player}'s turn | Area={self.current_area():.4f}")
            else:
                self.ax.set_title(f"Game Over! Player {self.winner} wins | Final area={self.current_area():.4f}")

        self.fig.canvas.draw()

    # --- Interaction --- #
    def on_click(self, event):
        if event.inaxes != self.ax or self.winner is not None:
            return
        click = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.vertices - click, axis=1)
        thresh = max(0.06, 0.5 * 2 * np.sin(np.pi / self.n))
        chosen = int(np.argmin(dists))
        if dists[chosen] > thresh:
            return
        if self.selected is None:
            if chosen in self.counters:
                self.selected = chosen
                self.update_plot()
            return
        legal, reason = self.is_legal_move(self.selected, chosen)
        if legal:
            self.save_state()  # save before move
            idx = self.counters.index(self.selected)
            self.counters[idx] = chosen
            self.player = 2 if self.player == 1 else 1
            if not self.has_any_legal_move():
                self.winner = 2 if self.player == 1 else 1
        else:
            print("Illegal move:", reason)
        self.selected = None
        self.update_plot()

    def on_key(self, event):
        if event.key == "u":
            self.undo()

# ---------- Run ----------
if __name__ == "__main__":
    n = int(input("Enter n (>=6): "))
    PolygonGame(n)
