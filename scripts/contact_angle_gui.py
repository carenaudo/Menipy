"""
Contact-angle measurement GUI (Tkinter).

A small desktop app that:
  * loads a sessile-drop image,
  * runs the automatic detection (substrate line, droplet contour, contact
    points) from contact_angle_detect_v2,
  * computes the LEFT and RIGHT contact angles by all four methods from
    contact_angle_methods,
  * lets the operator REDRAW THE SUBSTRATE LINE by hand (the one manual
    override allowed) and recomputes everything downstream from it.

Requirements:  python3, numpy, opencv-python, pillow, scipy
    pip install numpy opencv-python pillow scipy

Run:  python contact_angle_gui.py
"""

import os
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

import contact_angle_detect_v2 as det
import contact_angle_methods as cam


MAX_DISP = 720          # max displayed image dimension (px)


class ContactAngleApp:
    def __init__(self, root):
        self.root = root
        root.title("Sessile-drop contact-angle measurement")

        # state
        self.path = None
        self.img_bgr = None          # original image (BGR)
        self.scale = 1.0             # display scale (disp = orig * scale)
        self.result = None           # detector result dict
        self.methods = None          # computed angles dict
        self.baseline_override = None
        self.draw_mode = False       # True while redrawing the substrate
        self.first_pt = None         # first clicked baseline point (orig px)
        self.tkimg = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        left = ttk.Frame(self.root, padding=6)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # canvas
        self.canvas = tk.Canvas(left, bg="#202020", highlightthickness=0,
                                width=MAX_DISP, height=MAX_DISP)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # controls
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Load image…", command=self.load_image).pack(fill="x", pady=2)
        self.redraw_btn = ttk.Button(btns, text="Redraw substrate line",
                                     command=self.toggle_redraw, state="disabled")
        self.redraw_btn.pack(fill="x", pady=2)
        ttk.Button(btns, text="Auto substrate (reset)",
                   command=self.reset_substrate).pack(fill="x", pady=2)
        ttk.Button(btns, text="Save annotated…",
                   command=self.save_annotated).pack(fill="x", pady=2)

        # polynomial degree option
        opt = ttk.Frame(right)
        opt.pack(fill="x", pady=(0, 8))
        ttk.Label(opt, text="Polynomial degree:").pack(side="left")
        self.poly_deg = tk.IntVar(value=2)
        ttk.Spinbox(opt, from_=1, to=5, width=4, textvariable=self.poly_deg,
                    command=self.recompute_methods).pack(side="left", padx=4)

        # results table
        ttk.Label(right, text="Contact angles (deg)",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        cols = ("method", "left", "right")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=5)
        for c, w in zip(cols, (150, 70, 70)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", pady=4)

        # extra info
        self.info = tk.Text(right, width=42, height=12, wrap="word",
                            state="disabled", font=("TkFixedFont", 9))
        self.info.pack(fill="both", expand=True, pady=4)

        # status bar
        self.status = tk.StringVar(value="Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w").grid(row=1, column=0, columnspan=2, sticky="ew")

    # -------------------------------------------------------------- actions
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Open drop image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        self.path = path
        self.img_bgr = img
        self.baseline_override = None
        self.draw_mode = False
        self.first_pt = None
        h, w = img.shape[:2]
        self.scale = min(1.0, MAX_DISP / max(h, w))
        self.redraw_btn.config(state="normal")
        self.status.set(f"Loaded {os.path.basename(path)} ({w}x{h}). Detecting…")
        self.root.update_idletasks()
        self.run_pipeline()

    def toggle_redraw(self):
        self.draw_mode = not self.draw_mode
        self.first_pt = None
        if self.draw_mode:
            self.redraw_btn.config(text="Click 2 points… (cancel)")
            self.status.set("Click the LEFT then RIGHT end of the substrate line.")
        else:
            self.redraw_btn.config(text="Redraw substrate line")
            self.status.set("Redraw cancelled.")
            self.render()

    def reset_substrate(self):
        if self.img_bgr is None:
            return
        self.baseline_override = None
        self.draw_mode = False
        self.first_pt = None
        self.redraw_btn.config(text="Redraw substrate line")
        self.status.set("Substrate reset to automatic detection. Recomputing…")
        self.root.update_idletasks()
        self.run_pipeline()

    # ------------------------------------------------------------ pipeline
    def run_pipeline(self):
        if self.img_bgr is None:
            return
        try:
            self.result = det.analyze(self.path, draw=True,
                                      baseline_override=self.baseline_override)
            self.recompute_methods(rerender=False)
            self.render()
            bl = self.result["baseline"]
            tag = "manual" if bl.get("manual") else f"auto (inliers {bl['inlier_ratio']*100:.0f}%)"
            self.status.set(f"Done. Baseline {bl['angle_deg']:+.2f}deg [{tag}].")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Detection failed", str(e))
            self.status.set("Detection failed - try redrawing the substrate line.")

    def recompute_methods(self, rerender=True):
        if self.result is None:
            return
        try:
            self.methods = cam.compute_all(self.result, poly_degree=self.poly_deg.get())
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Angle computation failed", str(e))
            return
        self.fill_results()
        if rerender:
            self.render()

    # -------------------------------------------------------------- display
    def _orig_to_disp(self, x, y):
        return x * self.scale, y * self.scale

    def _disp_to_orig(self, x, y):
        return x / self.scale, y / self.scale

    def render(self):
        """Draw the annotated image (with angle overlays) on the canvas."""
        if self.result is None:
            return
        try:
            annotated = cam.visualize_angles(self.result, self.methods, out_path=None) \
                if self.methods is not None else self.result["vis"]
        except Exception:
            annotated = self.result["vis"]
        if annotated is None:
            annotated = self.img_bgr
        disp = annotated
        if self.scale != 1.0:
            disp = cv2.resize(annotated, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(width=self.tkimg.width(), height=self.tkimg.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg, tags="img")

    def on_canvas_motion(self, event):
        if not self.draw_mode or self.first_pt is None:
            return
        self.canvas.delete("rubber")
        x0, y0 = self._orig_to_disp(*self.first_pt)
        self.canvas.create_line(x0, y0, event.x, event.y, fill="#ff3bd0",
                                width=2, tags="rubber")

    def on_canvas_click(self, event):
        if not self.draw_mode or self.img_bgr is None:
            return
        ox, oy = self._disp_to_orig(event.x, event.y)
        if self.first_pt is None:
            self.first_pt = (ox, oy)
            self.canvas.create_oval(event.x - 4, event.y - 4, event.x + 4, event.y + 4,
                                    outline="#ff3bd0", width=2, tags="rubber")
        else:
            p1, p2 = self.first_pt, (ox, oy)
            w = self.img_bgr.shape[1]
            self.baseline_override = det.baseline_from_points(p1, p2, w)
            self.draw_mode = False
            self.first_pt = None
            self.canvas.delete("rubber")
            self.redraw_btn.config(text="Redraw substrate line")
            self.status.set("Substrate updated. Recomputing…")
            self.root.update_idletasks()
            self.run_pipeline()

    # -------------------------------------------------------------- results
    def fill_results(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        if not self.methods:
            return
        m = self.methods
        rows = [
            ("Circle (cap)", m["circle"]["left"], m["circle"]["right"]),
            ("Tangent", m["tangent"]["left"], m["tangent"]["right"]),
            ("Polynomial", m["polynomial"]["left"], m["polynomial"]["right"]),
        ]
        yl = m["young_laplace"]
        if yl.get("available"):
            rows.append(("Young-Laplace", yl["left"], yl["right"]))
        else:
            rows.append(("Young-Laplace", None, None))
        for name, l, r in rows:
            ls = f"{l:.2f}" if l is not None else "n/a"
            rs = f"{r:.2f}" if r is not None else "n/a"
            self.tree.insert("", "end", values=(name, ls, rs))

        # info panel
        s = m["shape_factors"]
        yl = m["young_laplace"]
        lines = []
        allL = [m["circle"]["left"], m["tangent"]["left"], m["polynomial"]["left"]]
        allR = [m["circle"]["right"], m["tangent"]["right"], m["polynomial"]["right"]]
        if yl.get("available"):
            allL.append(yl["left"]); allR.append(yl["right"])
        lines.append(f"mean   L={np.mean(allL):6.2f}   R={np.mean(allR):6.2f}")
        lines.append(f"spread L={np.ptp(allL):6.2f}   R={np.ptp(allR):6.2f}")
        lines.append("")
        if yl.get("available"):
            lines.append(f"Bond number (shape): {yl['bond_number']:.3f}")
            lines.append(f"apex radius R0     : {yl['R0_px']:.1f} px")
            lines.append(f"YL fit rmse        : {yl['rmse_px']:.2f} px")
            if "bond_number_physical" in yl:
                lines.append(f"Bond (physical)    : {yl['bond_number_physical']:.3f}")
        lines.append("")
        lines.append("Shape factors:")
        lines.append(f"  base width   : {s['base_width']:.1f} px")
        lines.append(f"  apex height  : {s['apex_height']:.1f} px")
        lines.append(f"  height/base  : {s['height_to_base']:.3f}")
        lines.append(f"  cap 2atan(h/a): {s['cap_angle_2atan']:.2f} deg")
        lines.append(f"  area         : {s['area_px2']:.0f} px^2")
        lines.append(f"  volume(revol): {s['volume_px3']:.0f} px^3")
        d = self.result["diagnostics"]
        lines.append("")
        lines.append(f"needle: {'attached' if d.get('needle_attached') else 'detached/none'}")
        lines.append(f"contrast score: {d.get('contrast', 0):.2f}")

        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.config(state="disabled")

    def save_annotated(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            cam.visualize_angles(self.result, self.methods, out_path=path)
            self.status.set(f"Saved {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    ContactAngleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
