import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from typing import List, Dict, Set, Tuple
import re

class StateVisualizer:
    def __init__(self, states):
        self.states = states
        self.current_idx = 0
        self.max_idx = len(states) - 1
        
        # Store highlight rectangles and current highlighted cell
        self.highlight_rects = []
        self.hover_annotations = []
        self.current_highlighted_cell = None
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(14, 10))
        
        # Create a 3-row, 2-column grid
        self.gs = gridspec.GridSpec(3, 2, figure=self.fig)
        
        # Input tensors on top row
        self.ax_in0 = self.fig.add_subplot(self.gs[0, 0])
        self.ax_in1 = self.fig.add_subplot(self.gs[0, 1])
        
        # Input circular buffers on second row - equal width
        self.ax_in0_cb = self.fig.add_subplot(self.gs[1, 0])
        self.ax_in1_cb = self.fig.add_subplot(self.gs[1, 1])
        
        # Output circular buffer and destination register on bottom row with equal width
        self.ax_out_cb = self.fig.add_subplot(self.gs[2, 0])
        self.ax_dest_reg = self.fig.add_subplot(self.gs[2, 1])
        
        # Navigation buttons
        self.ax_prev = plt.axes([0.3, 0.01, 0.1, 0.05])
        self.ax_next = plt.axes([0.6, 0.01, 0.1, 0.05])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_state)
        self.btn_next.on_clicked(self.next_state)
        
        # State counter and operation info text
        self.state_text = self.fig.text(0.5, 0.03, f"State: {self.current_idx}/{self.max_idx}", 
                                        ha='center', va='center', fontsize=12)
        self.operation_text = self.fig.text(0.03, 0.03, "", ha='left', va='center', fontsize=10, color='red')
        self.next_op_text = self.fig.text(0.97, 0.03, "", ha='right', va='center', fontsize=10, color='blue')
        
        # Connect keyboard events and click events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial plot
        self.update_plot()
        
    def on_key(self, event):
        if event.key == 'right':
            self.next_state(event)
        elif event.key == 'left':
            self.prev_state(event)
    
    def prev_state(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_plot()
    
    def next_state(self, event):
        if self.current_idx < self.max_idx:
            self.current_idx += 1
            self.update_plot()
    
    def _clear_highlights(self):
        """Clear all highlights and annotations."""
        for rect in self.highlight_rects:
            rect.remove()
        self.highlight_rects = []
        
        for annotation in self.hover_annotations:
            annotation.remove()
        self.hover_annotations = []
    
    def update_plot(self):
        # Clear all axes
        for ax in [self.ax_in0, self.ax_in1, 
                  self.ax_in0_cb, self.ax_in1_cb, self.ax_out_cb, 
                  self.ax_dest_reg]:
            ax.clear()
            
        # Clear any existing highlights
        self._clear_highlights()
        
        # Reset current highlighted cell
        self.current_highlighted_cell = None
        
        state = self.states[self.current_idx]
        
        # Find tensor elements that are in circular buffers
        in0_cb_elements = set()
        in1_cb_elements = set()
        
        # Get elements from in0_cb
        for i in range(state.in0_cb.size):
            if state.in0_cb.buffer[i] is not None and state.in0_cb.buffer[i] != 0:
                in0_cb_elements.add(str(state.in0_cb.buffer[i]))
        
        # Get elements from in1_cb
        for i in range(state.in1_cb.size):
            if state.in1_cb.buffer[i] is not None and state.in1_cb.buffer[i] != 0:
                in1_cb_elements.add(str(state.in1_cb.buffer[i]))
        
        # Plot input tensors with highlighting
        self._plot_tensor(self.ax_in0, state.in0, "Input 0", in0_cb_elements)
        self._plot_tensor(self.ax_in1, state.in1, "Input 1", in1_cb_elements)
        
        # Plot circular buffers
        self._plot_circular_buffer(self.ax_in0_cb, state.in0_cb, "Input 0 CB")
        self._plot_circular_buffer(self.ax_in1_cb, state.in1_cb, "Input 1 CB")
        self._plot_circular_buffer(self.ax_out_cb, state.out_cb, "Output CB")
        
        # Plot destination register
        self._plot_dest_register(self.ax_dest_reg, state.dest_register)
        
        # Update state counter
        self.state_text.set_text(f"State: {self.current_idx}/{self.max_idx}")
        
        # Update operation text
        if self.current_idx == 0:
            self.operation_text.set_text("")
        else:
            current_op = f"Previous Instruction: {state.operation}"
            self.operation_text.set_text(current_op)
        
        # Show next operation if available
        if self.current_idx < self.max_idx:
            next_state = self.states[self.current_idx + 1]
            next_op = f"Next Instruction: {next_state.operation}"
            self.next_op_text.set_text(next_op)
        else:
            self.next_op_text.set_text("")
        
        self.fig.canvas.draw_idle()
    
    def _plot_tensor(self, ax, tensor, title, highlighted_elements=None):
        """Plot a tensor as a grid with values inside cells."""
        rows, cols = tensor.shape
        state = self.states[self.current_idx]
        
        # Create a table-like visualization
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_title(title)
        
        # Remove default ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw regular grid lines
        for i in range(rows+1):
            ax.axhline(i, color='gray', linewidth=0.5)
        for j in range(cols+1):
            ax.axvline(j, color='gray', linewidth=0.5)
        
        # Draw colored block boundary lines based on tensor type
        if title == "Input 0":  # M x K matrix
            # M block boundaries (horizontal lines) in red
            for i in range(state.M_block_size, rows, state.M_block_size):
                if i < rows:  # Skip the boundary
                    ax.axhline(i, color='red', linewidth=2)
            # K block boundaries (vertical lines) in blue
            for j in range(state.K_block_size, cols, state.K_block_size):
                if j < cols:  # Skip the boundary
                    ax.axvline(j, color='blue', linewidth=2)
                    
        elif title == "Input 1":  # K x N matrix
            # K block boundaries (horizontal lines) in blue
            for i in range(state.K_block_size, rows, state.K_block_size):
                if i < rows:  # Skip the boundary
                    ax.axhline(i, color='blue', linewidth=2)
            # N block boundaries (vertical lines) in green
            for j in range(state.N_block_size, cols, state.N_block_size):
                if j < cols:  # Skip the boundary
                    ax.axvline(j, color='green', linewidth=2)
                    
        elif title == "Output":  # M x N matrix
            # No colored block boundaries for output
            # Just use regular grid lines (already drawn)
            pass
        
        # Add text for each cell
        for i in range(rows):
            for j in range(cols):
                val = tensor[i, j]
                val_str = str(val)
                
                # Check if this element is in a circular buffer
                is_highlighted = False
                if highlighted_elements is not None and val_str in highlighted_elements:
                    # Add background shading for elements in circular buffer
                    rect = plt.Rectangle((j, rows - i - 1), 1, 1, fill=True, alpha=0.3, facecolor='yellow')
                    ax.add_patch(rect)
                    is_highlighted = True
                
                ax.text(j + 0.5, rows - i - 0.5, val_str,
                        ha='center', va='center', fontsize=10, 
                        weight='bold' if is_highlighted else 'normal')
    
    def _plot_circular_buffer(self, ax, cb, title):
        """Plot a circular buffer with read and write pointers."""
        size = cb.size
        if size == 0:
            ax.text(0.5, 0.5, "Empty Buffer", ha='center', va='center')
            ax.set_title(title)
            ax.set_yticks([])
            return
            
        # Store cell info for hover functionality
        if title == "Output CB":
            ax.cb_cells = []
            
        # Create a horizontal buffer visualization
        ax.set_xlim(0, size)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(range(size+1))
        ax.set_title(f"{title} (read_ptr={cb.read_ptr}, write_ptr={cb.write_ptr})")
        
        # Draw cells
        for i in range(size):
            rect = plt.Rectangle((i, 0), 1, 1, fill=False, edgecolor='black')
            ax.add_patch(rect)
            val = cb.buffer[i]
            if val is None or val == 0:
                continue
            text_val = str(val)
            if ' + ' in text_val:  # If it contains + signs, display vertically
                parts = text_val.split(' + ')
                text_val = '\n+'.join(parts)
            
            # Store cell info for hover functionality if this is the output CB
            if title == "Output CB":
                ax.cb_cells.append({
                    'x': i,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'value': str(val)
                })
                
            ax.text(i + 0.5, 0.5, text_val, ha='center', va='center', fontsize=8)
        
        # Highlight read and write pointers
        if cb.read_ptr < size:
            read_rect = plt.Rectangle((cb.read_ptr, 0), 1, 1, fill=True, alpha=0.2, facecolor='green')
            ax.add_patch(read_rect)
            ax.text(cb.read_ptr + 0.5, 0.2, "R", ha='center', va='center', color='green', fontweight='bold')
        
        if cb.write_ptr < size:
            write_rect = plt.Rectangle((cb.write_ptr, 0), 1, 1, fill=True, alpha=0.2, facecolor='red')
            ax.add_patch(write_rect)
            ax.text(cb.write_ptr + 0.5, 0.8, "W", ha='center', va='center', color='red', fontweight='bold')
    
    def _plot_dest_register(self, ax, dest_reg):
        """Plot the destination register."""
        size = dest_reg.size
        
        # Store cell info for hover functionality
        ax.dest_cells = []
        
        # Create a horizontal register visualization
        ax.set_xlim(0, size)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(range(size+1))
        ax.set_title("Destination Register")
        
        # Draw cells
        for i in range(size):
            rect = plt.Rectangle((i, 0), 1, 1, fill=False, edgecolor='black')
            ax.add_patch(rect)
            val = dest_reg.buffer[i]
            if val is None or val == 0:
                continue
            text_val = str(val)
            if ' + ' in text_val:  # If it contains + signs, display vertically
                parts = text_val.split(' + ')
                text_val = '\n+'.join(parts)
                
            # Store cell info for hover functionality
            ax.dest_cells.append({
                'x': i,
                'y': 0,
                'width': 1,
                'height': 1,
                'value': str(val)
            })
            
            ax.text(i + 0.5, 0.5, text_val, ha='center', va='center', fontsize=8)
    
    def on_click(self, event):
        """Handle click events to highlight contributing elements."""
        if event.inaxes is None:
            # Clear highlights when clicking outside any axes
            self._clear_highlights()
            self.fig.canvas.draw_idle()
            return
            
        found_cell = False
        cell_value = None
        cell_id = None
            
        # Check if clicking on output CB
        if event.inaxes == self.ax_out_cb and hasattr(self.ax_out_cb, 'cb_cells'):
            for i, cell in enumerate(self.ax_out_cb.cb_cells):
                if (cell['x'] <= event.xdata < cell['x'] + cell['width'] and 
                    cell['y'] <= event.ydata < cell['y'] + cell['height']):
                    found_cell = True
                    cell_value = cell['value']
                    cell_id = f"out_cb_{i}"
                    break
                    
        # Check if clicking on destination register
        elif event.inaxes == self.ax_dest_reg and hasattr(self.ax_dest_reg, 'dest_cells'):
            for i, cell in enumerate(self.ax_dest_reg.dest_cells):
                if (cell['x'] <= event.xdata < cell['x'] + cell['width'] and 
                    cell['y'] <= event.ydata < cell['y'] + cell['height']):
                    found_cell = True
                    cell_value = cell['value']
                    cell_id = f"dest_reg_{i}"
                    break
        
        # Clear highlights if clicking on a different cell or outside any cell
        self._clear_highlights()
        
        # If we found a cell and it's not the same as the currently highlighted one, highlight it
        if found_cell:
            if cell_id != self.current_highlighted_cell:
                self._highlight_contributing_elements(cell_value)
                self.current_highlighted_cell = cell_id
            else:
                # If clicking the same cell again, just clear the highlight
                self.current_highlighted_cell = None
        else:
            # If clicking elsewhere, clear the current highlighted cell
            self.current_highlighted_cell = None
                    
        self.fig.canvas.draw_idle()
    
    def _highlight_contributing_elements(self, value_str):
        """Highlight the input tensor elements that contributed to this value."""
        if not value_str or value_str == '0':
            return
            
        # Parse the value string to find contributing elements
        # Format could be like "0d + 1l + 2u"
        contributing_elements = []
        
        # Split by + if present
        if ' + ' in value_str:
            parts = value_str.split(' + ')
        else:
            parts = [value_str]
            
        # Extract numbers and letters
        for part in parts:
            # Match patterns like "0d", "1l", "2u" - number followed by letter
            matches = re.findall(r'(\d+)([a-zA-Z]+)', part)
            for match in matches:
                number, letter = match
                contributing_elements.append((number, letter))
        
        # Highlight elements in input tensors
        if contributing_elements:
            state = self.states[self.current_idx]
            
            # Find positions of elements in input tensors
            for i in range(state.in0.shape[0]):
                for j in range(state.in0.shape[1]):
                    val = str(state.in0[i, j])
                    if any(val == num for num, _ in contributing_elements):
                        # Highlight this element in input 0
                        rect = plt.Rectangle((j, state.in0.shape[0] - i - 1), 1, 1, 
                                           fill=True, alpha=0.4, facecolor='lightblue', 
                                           transform=self.ax_in0.transData)
                        self.ax_in0.add_patch(rect)  # Add to axes, not figure
                        self.highlight_rects.append(rect)
            
            for i in range(state.in1.shape[0]):
                for j in range(state.in1.shape[1]):
                    val = str(state.in1[i, j])
                    if any(val == letter for _, letter in contributing_elements):
                        # Highlight this element in input 1
                        rect = plt.Rectangle((j, state.in1.shape[0] - i - 1), 1, 1, 
                                           fill=True, alpha=0.4, facecolor='lightgreen', 
                                           transform=self.ax_in1.transData)
                        self.ax_in1.add_patch(rect)  # Add to axes, not figure
                        self.highlight_rects.append(rect)
            
            # Add annotation explaining the highlight
            annotation = self.fig.text(0.5, 0.97, f"Highlighting contributors to: {value_str}", 
                                     ha='center', va='top', fontsize=10,
                                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            self.hover_annotations.append(annotation)
    
    def show(self):
        # Adjust layout without using tight_layout to avoid warnings
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.2, hspace=0.3)
        plt.show()


def visualize_states(states):
    """
    Create and show a visualization of the simulation states.
    
    Args:
        states: List of MatmulState objects
    """
    visualizer = StateVisualizer(states)
    visualizer.show()
