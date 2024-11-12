import os
import tkinter as tk
import numpy as np

from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedStyle
import csv

from shtmbss2.common.config import PATH_MAPS


class Location:
    N = "n"
    S = "s"
    W = "w"
    E = "e"


def char_to_num(char):
    if len(char) > 1:
        return int(char)
    ord_id = ord(char.lower()) - 95
    if 2 <= ord_id <= 27:
        return ord_id
    else:
        return int(char)


class GridCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid world creator")
        self.root.configure(bg="black")  # Set background color of root window
        self.root.geometry("1000x1000")
        # configure grid columns/rows which should be resized
        self.root.columnconfigure(4, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create a frame to hold all widgets including scrollbars and canvas
        main_frame = tk.Frame(self.root, bg="black")
        main_frame.grid(column=0, row=0, columnspan=5, sticky="nsew", padx=0, pady=0)

        # Create a canvas with scrollbars
        self.canvas = tk.Canvas(main_frame, bg="black")

        self.v_scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.v_scrollbar.config(command=self.canvas.yview)

        self.h_scrollbar = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.h_scrollbar.config(command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame on the canvas to hold the grid
        self.grid_frame = tk.Frame(self.canvas, bg="black")
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")

        # Bind scrollbar movements to canvas scrolling
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<Button-4>", lambda e: self.on_mouse_wheel(1))  # Bind mouse wheel event
        self.canvas.bind_all("<Button-5>", lambda e: self.on_mouse_wheel(-1))  # Bind mouse wheel event

        # Labels and input fields
        self.label_x = ttk.Label(self.root, text="Enter height (x):", background="black", foreground="white")
        self.label_x.grid(column=0, row=1, sticky="nsew", padx=5, pady=5)

        self.entry_x = ttk.Entry(self.root)
        self.entry_x.grid(column=0, row=2, sticky="nsew", padx=5, pady=5)
        self.entry_x.insert(0, "5")

        self.label_y = ttk.Label(self.root, text="Enter width (y):", background="black", foreground="white")
        self.label_y.grid(column=1, row=1, sticky="nsew", padx=5, pady=5)

        self.entry_y = ttk.Entry(self.root)
        self.entry_y.grid(column=1, row=2, sticky="nsew", padx=5, pady=5)
        self.entry_y.insert(0, "5")

        # Generate buttons for adding/removing columns
        self.button_add_left = ttk.Button(self.root, text="L+", width=5,
                                          command=lambda: self.add_cells(col=True, location=Location.W))
        self.button_add_left.grid(column=0, row=3, sticky="nsw", padx=15, pady=5)

        self.button_add_right = ttk.Button(self.root, text="R+", width=5,
                                           command=lambda: self.add_cells(col=True, location=Location.E))
        self.button_add_right.grid(column=0, row=3, sticky="nse", padx=15, pady=5)

        self.button_add_top = ttk.Button(self.root, text="T+", width=5,
                                         command=lambda: self.add_cells(row=True, location=Location.N))
        self.button_add_top.grid(column=1, row=3, sticky="nsw", padx=15, pady=5)

        self.button_add_down = ttk.Button(self.root, text="B+", width=5,
                                          command=lambda: self.add_cells(row=True, location=Location.S))
        self.button_add_down.grid(column=1, row=3, sticky="nse", padx=15, pady=5)

        self.button_remove_left = ttk.Button(self.root, text="L-", width=5,
                                             command=lambda: self.remove_cells(col=True, location=Location.W))
        self.button_remove_left.grid(column=0, row=4, sticky="nsw", padx=15, pady=5)

        self.button_remove_right = ttk.Button(self.root, text="R-", width=5,
                                              command=lambda: self.remove_cells(col=True, location=Location.E))
        self.button_remove_right.grid(column=0, row=4, sticky="nse", padx=15, pady=5)

        self.button_remove_top = ttk.Button(self.root, text="T-", width=5,
                                            command=lambda: self.remove_cells(row=True, location=Location.N))
        self.button_remove_top.grid(column=1, row=4, sticky="nsw", padx=15, pady=5)

        self.button_remove_down = ttk.Button(self.root, text="B-", width=5,
                                             command=lambda: self.remove_cells(row=True, location=Location.S))
        self.button_remove_down.grid(column=1, row=4, sticky="nse", padx=15, pady=5)

        # Generate remaining buttons
        self.generate_button = ttk.Button(self.root, text="Generate Grid", command=self.generate_grid)
        self.generate_button.grid(column=2, row=2, sticky="nsew", padx=5, pady=5)

        self.assign_button = ttk.Button(self.root, text="Assign Unique Values", command=self.assign_unique_values)
        self.assign_button.grid(column=2, row=3, sticky="nsew", padx=5, pady=5)

        # slider for updating the scale of the grid
        self.resize_slider = ttk.Scale(self.root, from_=0, to=2, value=1, orient=tk.HORIZONTAL,
                                       command=self.resize_grid)
        self.resize_slider.grid(column=2, row=4, sticky="nsew", padx=5, pady=5)

        # Buttons for loading/saving map
        map_files = [map_file for map_file in os.listdir(PATH_MAPS) if map_file.endswith('.csv')]
        map_files = sorted(map_files)

        self.load_map_var = tk.StringVar()
        self.load_map_var.set(map_files[0])

        self.select_map_dropdown = ttk.Combobox(self.root, textvariable=self.load_map_var,
                                              values=map_files)
        self.select_map_dropdown.grid(column=3, row=2, sticky="nsew", padx=5, pady=5)

        self.load_map_button = ttk.Button(self.root, text="Load map", command=self.load_map)
        self.load_map_button.grid(column=3, row=3, sticky="nsew", padx=5, pady=5)

        self.save_button = ttk.Button(self.root, text="Save as CSV", command=self.save_as_csv)
        self.save_button.grid(column=3, row=4, sticky="nsew", padx=5, pady=5)

        self.grid_values = None  # 2D list to hold grid cell values
        self.labels = []  # 2D list to hold label widgets
        self.is_mouse_pressed = False
        self.prev_row = None
        self.prev_col = None
        self.prev_label = None

        self.label_size = (10, 6)
        self.label_size_scale = 1.0

    def get_label_size(self, dim):
        return int(np.round(self.label_size[dim] * self.label_size_scale))

    def create_cell(self, x, y, cell_color=None):
        if cell_color is None:
            if self.grid_values[x, y] >= 1:
                cell_color = "white"
            elif self.grid_values[x, y] < 0:
                cell_color = "black"
            else:
                cell_color = "grey"

        label = tk.Label(self.grid_frame, text="", relief=tk.RIDGE, width=self.get_label_size(0),
                         height=self.get_label_size(1), bg=cell_color)
        label.grid(row=x, column=y, padx=1, pady=1)
        label.bind("<Button-1>", lambda event, row=x, col=y: self.on_cell_click(event, row, col))
        label.bind("<Button-3>", lambda event, row=x, col=y: self.on_cell_click(event, row, col, right=True))
        label.bind("<<B1-Enter>>", lambda event, row=x, col=y: self.on_cell_enter(event, row, col))
        label.bind("<<B3-Enter>>", lambda event, row=x, col=y: self.on_cell_enter(event, row, col, right=True))

        return label

    def update_cell_location(self, x, y, new_x, new_y):
        label: tk.Label = self.labels[x][y]

        # Remove old bindings
        label.unbind("<Button-1>")
        label.unbind("<Button-3>")
        label.unbind("<<B1-Enter>>")
        label.unbind("<<B3-Enter>>")

        # Set new location
        label.grid(row=new_x, column=new_y, padx=1, pady=1)

        # Create new bindings
        label.bind("<Button-1>", lambda event, row=new_x, col=new_y: self.on_cell_click(event, row, col))
        label.bind("<Button-3>", lambda event, row=new_x, col=new_y: self.on_cell_click(event, row, col, right=True))
        label.bind("<<B1-Enter>>", lambda event, row=new_x, col=new_y: self.on_cell_enter(event, row, col))
        label.bind("<<B3-Enter>>", lambda event, row=new_x, col=new_y: self.on_cell_enter(event, row, col, right=True))

        self.labels[x][y] = label

    def add_cells(self, row=False, col=False, location=None):
        if self.labels is None or len(self.labels) <= 0 or len(self.labels[0]) <= 0 or self.grid_values is None:
            self.generate_grid(x=1, y=1)
            return
        if not row and not col:
            messagebox.showerror("Error", "Either row or column have to be set.")
            return
        elif row and col:
            messagebox.showerror("Error", "Only row or column can be set to 'true'.")
            return
        if type(location) is not str or location not in "nsew":
            messagebox.showerror("Error", f"The location has to be a cardinal direction string (nsew): {location}")
            return
        if row and location not in "ns" or col and location not in "ew":
            messagebox.showerror("Error", "Wrong combination of column/row and location (nsew).")
            return

        # Create new row/column
        if row:
            size = len(self.labels[0])
            new_cells = list()
            for i_cell in range(size):
                x = 0 if location == Location.N else len(self.labels)
                y = i_cell
                new_cell = self.create_cell(x, y, cell_color="grey")
                new_cells.append(new_cell)

                if location == Location.N:
                    for j_cell in range(len(self.labels)):
                        self.update_cell_location(j_cell, i_cell, j_cell + 1, i_cell)

            new_arr = np.zeros((1, self.grid_values.shape[1]), dtype=np.int8)
            if location == Location.N:

                self.labels = [new_cells] + self.labels
                self.grid_values = np.concatenate([new_arr, self.grid_values], axis=0, dtype=np.int8)
            else:
                self.labels = self.labels + [new_cells]
                self.grid_values = np.concatenate([self.grid_values, new_arr], axis=0, dtype=np.int8)
        else:
            size = len(self.labels)
            y = 0 if location == Location.W else len(self.labels[0])
            for i_cell in range(size):
                x = i_cell
                new_cell = self.create_cell(x, y, cell_color="grey")

                if location == Location.W:
                    for j_cell in range(len(self.labels[i_cell])):
                        self.update_cell_location(i_cell, j_cell, i_cell, j_cell + 1)
                    self.labels[i_cell] = [new_cell] + self.labels[i_cell]
                else:
                    self.labels[i_cell] = self.labels[i_cell] + [new_cell]

            new_arr = np.zeros((self.grid_values.shape[0], 1), dtype=np.int8)
            if location == Location.W:
                self.grid_values = np.concatenate(
                    [np.zeros((self.grid_values.shape[0], 1), dtype=np.int8), self.grid_values], axis=1, dtype=np.int8)
            else:
                self.grid_values = np.concatenate([self.grid_values, new_arr], axis=1, dtype=np.int8)

        # Update the canvas scroll region
        self.canvas.update_idletasks()  # Ensure updates are applied before configuring scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def remove_cells(self, row=False, col=False, location=None):
        if self.labels is None or len(self.labels) <= 0 or len(self.labels[0]) <= 0 or self.grid_values is None:
            return
        if not row and not col:
            messagebox.showerror("Error", "Either row or column have to be set.")
            return
        elif row and col:
            messagebox.showerror("Error", "Only row or column can be set to 'true'.")
            return
        if type(location) is not str or location not in "nsew":
            messagebox.showerror("Error", f"The location has to be a cardinal direction string (nsew): {location}")
            return
        if row and location not in "ns" or col and location not in "ew":
            messagebox.showerror("Error", "Wrong combination of column/row and location (nsew).")
            return

        # Create new row/column
        if row:
            size = len(self.labels[0])
            x = 0 if location == Location.N else len(self.labels) - 1

            # Update old cells
            if location == Location.N:
                for i_x in range(1, len(self.labels)):
                    for i_y in range(size):
                        self.update_cell_location(i_x, i_y, i_x + 1, i_y)

            # Remove row from labels and grid values
            labels = self.labels.pop(x)
            for label in labels:
                label.destroy()
            self.grid_values = np.delete(self.grid_values, x, axis=0)
        else:
            size = len(self.labels)
            y = 0 if location == Location.W else len(self.labels[0]) - 1

            # Update old cells
            for i_x in range(size):
                if location == Location.W:
                    for i_y in range(1, len(self.labels[i_x])):
                        self.update_cell_location(i_x, i_y, i_x + 1, i_y)
                # Remove row from labels
                label = self.labels[i_x].pop(y)
                label.destroy()

            # Remove row from grid values
            self.grid_values = np.delete(self.grid_values, y, axis=1)

        # Update the canvas scroll region
        self.canvas.update_idletasks()  # Ensure updates are applied before configuring scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def generate_grid(self, x=None, y=None, grid_values=None):
        try:
            if x is None:
                x = int(self.entry_x.get())
            if y is None:
                y = int(self.entry_y.get())

            if x <= 0 or y <= 0:
                messagebox.showerror("Error", "Dimensions must be positive integers.")
                return

            # Clear any previous grid
            for widget in self.grid_frame.winfo_children():
                widget.destroy()

            # Initialize grid_values and labels lists
            if grid_values is None:
                self.grid_values = np.zeros((x, y), dtype=np.int8)
            else:
                self.grid_values = grid_values
            self.labels = [[None for _ in range(y)] for _ in range(x)]

            self.root.bind("<B1-Motion>", lambda event: self.on_move(event))
            self.root.bind("<B3-Motion>", lambda event: self.on_move(event))
            self.root.bind("<ButtonRelease-1>", lambda event: self.on_release(event))
            self.root.bind("<ButtonRelease-3>", lambda event: self.on_release(event))

            # Create the grid
            for i in range(x):
                for j in range(y):
                    # Create new label (grid cell)
                    label = self.create_cell(i, j)

                    # Store label in 2D list
                    self.labels[i][j] = label
                    if self.grid_values[i][j] > 1:
                        self.labels[i][j].config(text=chr(self.grid_values[i][j] + 63))

            # Update the canvas scroll region
            self.canvas.update_idletasks()  # Ensure updates are applied before configuring scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for x and y.")

    def resize_grid(self, scale):
        self.label_size_scale = float(scale)
        for labels_i in self.labels:
            for label_i in labels_i:
                label_i.config(width=self.get_label_size(0),
                               height=self.get_label_size(1))

    def on_release(self, event):
        self.is_mouse_pressed = False

    def on_cell_click(self, event, row, col, right=False):
        self.is_mouse_pressed = True
        self.right = right
        self.update_cell(row, col)

        widget = event.widget.winfo_containing(event.x_root, event.y_root)
        if self.prev_label != widget:
            self.prev_label = widget

    def on_move(self, event):
        if self.is_mouse_pressed:  # Check if left mouse button is pressed (bitwise check)
            widget = event.widget.winfo_containing(event.x_root, event.y_root)
            if widget is None:
                return
            if self.prev_label != widget:
                self.prev_label = widget
                widget.event_generate("<<B1-Enter>>")

    def on_cell_enter(self, event, row, col, right=False):
        self.update_cell(row, col)

    def update_cell(self, row, col):
        if self.right:
            if self.grid_values[row, col] < 0:
                self.grid_values[row, col] = 0
            else:
                self.grid_values[row, col] = -1
        else:
            if self.grid_values[row, col] != 0:
                self.grid_values[row, col] = 0
            else:
                self.grid_values[row, col] = 1

        cell_value = self.grid_values[row, col]
        if cell_value >= 1:
            cell_color = "white"
        elif cell_value == 0:
            cell_color = "grey"
        else:
            cell_color = "black"
        self.labels[row][col].config(bg=cell_color, text="")  # Update background color of the label

    def load_map(self):
        try:
            map_path = os.path.join(PATH_MAPS, self.load_map_var.get())
            if not os.path.exists(map_path):
                messagebox.showerror("Error", f"Failed to load map, path does not exist: {map_path}")

            grid_values = list()
            with open(map_path) as csvfile:
                map_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in map_reader:
                    row = [char_to_num(char) for char in row]
                    grid_values.append(row)

            grid_values = np.array(grid_values, dtype=np.int8)

            self.generate_grid(x=grid_values.shape[0], y=grid_values.shape[1], grid_values=grid_values)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load map: {str(e)}")

    def cell_conversion(self, content):
        if content < 2:
            return content

        return chr(content+63)

    def save_as_csv(self):
        try:
            if not self.grid_values.any():
                messagebox.showwarning("Warning", "No grid data to save.")
                return

            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                                    initialdir=PATH_MAPS)
            if filename:
                with open(filename, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for row in self.grid_values:
                        row_final = [self.cell_conversion(char) for char in row]
                        csvwriter.writerow(row_final)

                messagebox.showinfo("Save Successful", f"Grid saved as {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save grid: {str(e)}")

    def on_mouse_wheel(self, delta):
        # Vertical scrolling
        if delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif delta < 0:
            self.canvas.yview_scroll(1, "units")

    def assign_unique_values(self, x_start=0, assigned_values=None):
        unique_value = np.max(self.grid_values) + 1
        if assigned_values is None:
            assigned_values = {}  # Dictionary to store already assigned values

        # find horizontal walls
        x_max = len(self.grid_values)
        for i in range(x_start, len(self.grid_values)):
            if (self.grid_values[i, :] < 0).all():
                x_max = i
                break

        for i in range(x_start, x_max):
            for j in range(len(self.grid_values[0])):
                if self.grid_values[i, j] == 1:
                    if (i - x_start, j) not in assigned_values:
                        self.grid_values[i, j] = unique_value
                        assigned_values[(i, j)] = unique_value
                        unique_value += 1
                    elif (i - x_start, j) in assigned_values and x_start > 0:
                        self.grid_values[i, j] = assigned_values[(i - x_start, j)]

        # Update labels based on updated grid_values
        for i in range(x_start, x_max):
            for j in range(len(self.grid_values[0])):
                if self.grid_values[i, j] == 0:
                    continue
                cell_value = self.grid_values[i, j]
                cell_color = "white" if cell_value >= 1 else "grey"
                self.labels[i][j].config(bg=cell_color, text=chr(cell_value + 63))

        if x_max < len(self.grid_values):
            self.assign_unique_values(x_start=x_max + 1, assigned_values=assigned_values)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()

    style = ThemedStyle(root)
    style.set_theme('equilux')

    app = GridCreator(root)
    app.run()
