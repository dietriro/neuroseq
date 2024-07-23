import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ttkthemes import ThemedStyle
import csv


class GridCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid world creator")
        self.root.configure(bg="black")  # Set background color of root window

        # Create a frame to hold all widgets including scrollbars and canvas
        main_frame = tk.Frame(self.root, bg="black")
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

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
        self.label_x.pack()

        self.entry_x = ttk.Entry(self.root)
        self.entry_x.insert(0, "5")
        self.entry_x.pack()

        self.label_y = ttk.Label(self.root, text="Enter width (y):", background="black", foreground="white")
        self.label_y.pack()

        self.entry_y = ttk.Entry(self.root)
        self.entry_y.insert(0, "5")
        self.entry_y.pack()

        self.generate_button = ttk.Button(self.root, text="Generate Grid", command=self.generate_grid)
        self.generate_button.pack()

        self.label_type = ttk.Label(self.root, text="Select cell type:", background="black", foreground="white")
        self.label_type.pack()

        self.cell_type_var = tk.StringVar()
        self.cell_type_var.set("traversable")  # Default value

        self.cell_type_dropdown = ttk.Combobox(self.root, textvariable=self.cell_type_var,
                                               values=["traversable", "occupied"])
        self.cell_type_dropdown.pack()

        self.save_button = ttk.Button(self.root, text="Save as CSV", command=self.save_as_csv)
        self.save_button.pack()

        self.grid_values = []  # 2D list to hold grid cell values
        self.labels = []  # 2D list to hold label widgets
        self.is_mouse_pressed = False
        self.prev_row = None
        self.prev_col = None
        self.prev_label = None

    def generate_grid(self):
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())

            if x <= 0 or y <= 0:
                messagebox.showerror("Error", "Dimensions must be positive integers.")
                return

            # Clear any previous grid
            for widget in self.grid_frame.winfo_children():
                widget.destroy()

            # Initialize grid_values and labels lists
            self.grid_values = [[0 for _ in range(y)] for _ in range(x)]
            self.labels = [[None for _ in range(y)] for _ in range(x)]

            self.root.bind("<B1-Motion>", lambda event: self.on_move(event))
            self.root.bind("<ButtonRelease-1>", lambda event: self.on_release(event))

            # Create the grid
            for i in range(x):
                for j in range(y):
                    cell_value = self.grid_values[i][j]
                    cell_color = "white" if cell_value == 1 else "grey"

                    label = tk.Label(self.grid_frame, text="", relief=tk.RIDGE, width=10, height=6, bg=cell_color)
                    label.grid(row=i, column=j, padx=1, pady=1)
                    label.bind("<Button-1>", lambda event, row=i, col=j: self.on_cell_click(event, row, col))
                    label.bind("<<B1-Enter>>", lambda event, row=i, col=j: self.on_cell_enter(event, row, col))

                    self.labels[i][j] = label  # Store label in 2D list

            # Update the canvas scroll region
            self.canvas.update_idletasks()  # Ensure updates are applied before configuring scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for x and y.")

    def on_release(self, event):
        self.is_mouse_pressed = False

    def on_cell_click(self, event, row, col):
        self.is_mouse_pressed = True
        self.update_cell(row, col)

    def on_move(self, event):
        if self.is_mouse_pressed:  # Check if left mouse button is pressed (bitwise check)
            widget = event.widget.winfo_containing(event.x_root, event.y_root)
            if self.prev_label != widget:
                self.prev_label = widget
                widget.event_generate("<<B1-Enter>>")

    def on_cell_enter(self, event, row, col):
        self.update_cell(row, col)

    def update_cell(self, row, col):
        selected_type = self.cell_type_var.get()
        if selected_type == "traversable":
            self.grid_values[row][col] = 1
        elif selected_type == "occupied":
            self.grid_values[row][col] = 0

        cell_value = self.grid_values[row][col]
        cell_color = "white" if cell_value == 1 else "grey"
        self.labels[row][col].config(bg=cell_color)  # Update background color of the label

    def save_as_csv(self):
        try:
            if not self.grid_values:
                messagebox.showwarning("Warning", "No grid data to save.")
                return

            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if filename:
                with open(filename, "w", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    for row in self.grid_values:
                        csvwriter.writerow(row)

                messagebox.showinfo("Save Successful", f"Grid saved as {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save grid: {str(e)}")

    def on_mouse_wheel(self, delta):
        # Vertical scrolling
        if delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif delta < 0:
            self.canvas.yview_scroll(1, "units")

        # # Horizontal scrolling
        # if delta > 0:
        #     self.canvas.xview_scroll(-1, "units")
        # elif delta < 0:
        #     self.canvas.xview_scroll(1, "units")


if __name__ == "__main__":
    root = tk.Tk()

    style = ThemedStyle(root)
    style.set_theme('equilux')

    app = GridCreator(root)
    root.mainloop()
