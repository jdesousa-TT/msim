import numpy as np
from typing import Tuple, List, Optional
from copy import deepcopy

class DestRegister:
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=object)
    def __repr__(self):
        return (f"DestRegister(size={self.size}, buffer={self.buffer})")
    def clear(self):
        self.buffer = np.zeros(self.size, dtype=object)

class CircularBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=object)
        self.read_ptr = 0
        self.write_ptr = 0
    def __repr__(self):
        return (f"CircularBuffer(size={self.size}, buffer={self.buffer}, read_ptr={self.read_ptr}, write_ptr={self.write_ptr})")

class MatmulState:
    def __init__(self,
                 in0_shape: Tuple[int, int],
                 in1_shape: Tuple[int, int],
                 M_block: int = 8,
                 N_block: int = 8,
                 K_block: int = 8,
                 dest_register_size: int = 8,
                 operation: str = "initial_state"):

        self.in0_shape = in0_shape
        self.in1_shape = in1_shape
        self.M_block = M_block  # Number of blocks along M
        self.N_block = N_block  # Number of blocks along N
        self.K_block = K_block  # Number of blocks along K
        self.dest_register_size = dest_register_size
        self.M = in0_shape[0]
        self.N = in1_shape[1]
        self.K = in0_shape[1]

        assert in0_shape[1] == in1_shape[0], "K dim doesn't match"
        self.output_shape = (in0_shape[0], in1_shape[1])

        # Compute block sizes from number of blocks
        self.M_block_size = self.M // self.M_block if self.M_block else self.M
        self.N_block_size = self.N // self.N_block if self.N_block else self.N
        self.K_block_size = self.K // self.K_block if self.K_block else self.K

        # Tensors (tiles)
        num_tiles1 = in0_shape[0] * in0_shape[1]
        self.in0 = np.array([str(i) for i in range(num_tiles1)]).reshape(in0_shape)

        def repeated_letter(idx):
            letter = chr(ord('a') + (idx % 26))
            reps = (idx // 26) + 1
            return letter * reps
        num_tiles2 = in1_shape[0] * in1_shape[1]
        alphabet = [repeated_letter(i) for i in range(num_tiles2)]
        self.in1 = np.array(alphabet).reshape(in1_shape)

        self.output = np.zeros(self.output_shape, dtype=int)
        
        # Circular buffers (for tiles), sized by block size
        self.in0_cb = CircularBuffer(self.M_block_size * self.K_block_size)
        self.in1_cb = CircularBuffer(self.K_block_size * self.N_block_size)
        self.out_cb = CircularBuffer(self.M * self.N)
        
        # Destination register (max 8 tiles)
        self.dest_register = DestRegister(dest_register_size)
        
        # Track the operation that created this state
        self.operation = operation

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return (f"TileSystemState(\n"
                f"  operation={self.operation}\n"
                f"  in0=\n{self.in0}\n"
                f"  in1=\n{self.in1}\n"
                f"  output=\n{self.output}\n"
                f"  in0_cb={self.in0_cb}\n"
                f"  in1_cb={self.in1_cb}\n"
                f"  out_cb={self.out_cb}\n"
                f"  dest_register={self.dest_register}\n"
                f")")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Matmul State Simulator")
    parser.add_argument('--M', type=int, default=2, help='Rows of input0 (A)')
    parser.add_argument('--N', type=int, default=8, help='Cols of input1 (B)')
    parser.add_argument('--K', type=int, default=4, help='Cols of input0/rows of input1')
    parser.add_argument('--bm', type=int, default=2, help='Number of blocks along M')
    parser.add_argument('--bn', type=int, default=1, help='Number of blocks along N')
    parser.add_argument('--bk', type=int, default=4, help='Number of blocks along K')
    parser.add_argument('--dest-size', type=int, default=8, help='Destination register size (max 8)')
    args = parser.parse_args()


    init_M = args.M
    init_N = args.N
    init_K = args.K
    in0_shape = (init_M, init_K)
    in1_shape = (init_K, init_N)
    M_block = args.bm
    N_block = args.bn
    K_block = args.bk
    M_block_size = init_M // M_block if M_block else init_M
    N_block_size = init_N // N_block if N_block else init_N
    K_block_size = init_K // K_block if K_block else init_K
    out_block_size = M_block_size * N_block_size
    dest_register_size = args.dest_size

    # Initialize the simulation state
    sim_states = [MatmulState((init_M, init_K), (init_K, init_N), M_block, N_block, K_block, dest_register_size, "initial_state")]

    print("Initialized state:\n", sim_states[0])

    # --- cb_* API boilerplate using cb_op_wrapper ---
    def cb_op_wrapper(cb_name, op_fn, op_name, *args, **kwargs):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = op_name
        cb = getattr(new_state, cb_name)
        op_fn(cb, *args, **kwargs)
        save_state(new_state)
        return new_state

    # Example cb_* implementations (user should fill in logic)
    def cb_wait_front_impl(cb, num_tiles):
        pass  # user logic here
    def cb_push_back_impl(cb, num_tiles):
        cb.write_ptr += num_tiles
        cb.write_ptr = cb.write_ptr % cb.size
        pass  # user logic here
    def cb_pop_front_impl(cb, num_tiles):
        cb.read_ptr += num_tiles
        cb.read_ptr = cb.read_ptr % cb.size
        pass  # user logic here 
    def cb_reserve_back_impl(cb, num_tiles):
        pass  # user logic here

    # Public cb_* APIs
    def cb_push_back(cb_name, num_items):
        return cb_op_wrapper(cb_name, cb_push_back_impl, f"cb_push_back({cb_name}, {num_items})", num_items)

    def cb_pop_front(cb_name, num_items):
        return cb_op_wrapper(cb_name, cb_pop_front_impl, f"cb_pop_front({cb_name}, {num_items})", num_items)

    def cb_wait_front(cb_name, num_items):
        return cb_op_wrapper(cb_name, cb_wait_front_impl, f"cb_wait_front({cb_name}, {num_items})", num_items)

    def cb_reserve_back(cb_name, num_items):
        return cb_op_wrapper(cb_name, cb_reserve_back_impl, f"cb_reserve_back({cb_name}, {num_items})", num_items)

    # --- copy_tile API with wrapper ---
    def copy_tile_op_wrapper(src_cb_name, cb_idx, dst_idx):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = f"copy_tile({src_cb_name}, {cb_idx}, {dst_idx})"
        src_cb = getattr(new_state, src_cb_name)
        dest_reg = new_state.dest_register
        copy_tile_impl(src_cb, dest_reg, cb_idx, dst_idx)
        save_state(new_state)
        return new_state

    def copy_tile_impl(src_cb, dest_reg, cb_idx, dst_idx):
        # Copy a tile from src_cb.buffer at (read_ptr + cb_idx) % size to dest_reg.buffer[dst_idx]
        buf_idx = (src_cb.read_ptr + cb_idx) % src_cb.size
        dest_reg.buffer[dst_idx] = src_cb.buffer[buf_idx]

    def copy_tile(src_cb_name, cb_idx, dst_idx):
        return copy_tile_op_wrapper(src_cb_name, cb_idx, dst_idx)

    # --- pack_tile API with wrapper ---
    def pack_tile_op_wrapper(dst_idx, cb_name, cb_idx):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = f"pack_tile({dst_idx}, {cb_name}, {cb_idx})"
        cb = getattr(new_state, cb_name)
        dest_reg = new_state.dest_register
        pack_tile_impl(dest_reg, dst_idx, cb, cb_idx)
        save_state(new_state)
        return new_state

    def pack_tile_impl(dest_reg, dst_idx, cb, cb_idx):
        # Copy from dest_reg.buffer[dst_idx] to cb.buffer at (write_ptr + cb_idx) % cb.size
        buf_idx = (cb.write_ptr + cb_idx) % cb.size
        cb.buffer[buf_idx] = dest_reg.buffer[dst_idx]

    def pack_tile(dst_idx, cb_name, cb_idx):
        return pack_tile_op_wrapper(dst_idx, cb_name, cb_idx)

    def matmul_block_op_wrapper(in0_cb_name, in1_cb_name):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = f"matmul_block({in0_cb_name}, {in1_cb_name})"
        in0_cb = getattr(new_state, in0_cb_name)
        in1_cb = getattr(new_state, in1_cb_name)
        dest_reg = new_state.dest_register
        matmul_block_impl(in0_cb, in1_cb, dest_reg)
        save_state(new_state)
        return new_state

    def matmul_block_impl(in0_cb, in1_cb, dest_reg):
        # Perform matrix multiplication: for each (m, n), accumulate the string product over k
        for m in range(M_block_size):
            for n in range(N_block_size):
                acc = ""
                for k in range(K_block_size):
                    # Compute flat indices into the buffers, relative to read_ptr
                    in0_idx = (in0_cb.read_ptr + m * K_block_size + k) % in0_cb.size
                    in1_idx = (in1_cb.read_ptr + k * N_block_size + n) % in1_cb.size
                    # Get tile values as strings
                    tile0 = str(in0_cb.buffer[in0_idx])
                    tile1 = str(in1_cb.buffer[in1_idx])
                    acc += tile0 + tile1  # e.g. '1a'
                dest_idx = m * N_block_size + n
                dest_reg.buffer[dest_idx] = str(dest_reg.buffer[dest_idx]) + " + " + acc if dest_reg.buffer[dest_idx] else acc

    def matmul_block(in0_cb_name, in1_cb_name):
        return matmul_block_op_wrapper(in0_cb_name, in1_cb_name)

    # --- Tile loading APIs with wrappers ---
    def load_in0_tiles_op_wrapper(m_start, k_start):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = f"load_in0_tiles({m_start}, {k_start})"
        load_in0_tiles_impl(new_state, m_start, k_start)
        save_state(new_state)
        return new_state

    def load_in0_tiles_impl(state, m_start, k_start):
        # Loads a block of tiles from in0 into in0_cb buffer
        for m in range(state.M_block_size):
            for k in range(state.K_block_size):
                tile = state.in0[m_start + m, k_start + k]
                buf_idx = (state.in0_cb.write_ptr + m * state.K_block_size + k) % state.in0_cb.size
                state.in0_cb.buffer[buf_idx] = tile
        state.in0_cb.write_ptr = (state.in0_cb.write_ptr + state.M_block_size * state.K_block_size) % state.in0_cb.size

    def load_in0_tiles(m_start, k_start):
        return load_in0_tiles_op_wrapper(m_start, k_start)

    def load_in1_tiles_op_wrapper(k_start, n_start):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = f"load_in1_tiles({k_start}, {n_start})"
        load_in1_tiles_impl(new_state, k_start, n_start)
        save_state(new_state)
        return new_state

    def load_in1_tiles_impl(state, k_start, n_start):
        # Loads a block of tiles from in1 into in1_cb buffer
        for k in range(state.K_block_size):
            for n in range(state.N_block_size):
                tile = state.in1[k_start + k, n_start + n]
                buf_idx = (state.in1_cb.write_ptr + k * state.N_block_size + n) % state.in1_cb.size
                state.in1_cb.buffer[buf_idx] = tile
        state.in1_cb.write_ptr = (state.in1_cb.write_ptr + state.K_block_size * state.N_block_size) % state.in1_cb.size

    def load_in1_tiles(k_start, n_start):
        return load_in1_tiles_op_wrapper(k_start, n_start)

    def save_state(state: MatmulState):
        sim_states.append(state)

    # --- Tile register operations ---
    def tile_regs_op_wrapper(op_fn, op_name, *args, **kwargs):
        prev_state = sim_states[-1]
        new_state = prev_state.copy()
        new_state.operation = op_name
        op_fn(new_state, *args, **kwargs)
        save_state(new_state)
        return new_state
        
    def tile_regs_acquire_impl(state):
        # Implementation to be provided by user
        pass
        
    def tile_regs_acquire():
        return tile_regs_op_wrapper(tile_regs_acquire_impl, "tile_regs_acquire()")

    def tile_regs_commit_impl(state):
        # Implementation to be provided by user
        pass
        
    def tile_regs_commit():
        return tile_regs_op_wrapper(tile_regs_commit_impl, "tile_regs_commit()")

    def tile_regs_wait_impl(state):
        # Implementation to be provided by user
        pass
        
    def tile_regs_wait():
        return tile_regs_op_wrapper(tile_regs_wait_impl, "tile_regs_wait()")

    def tile_regs_release_impl(state):
        state.dest_register.clear()
        pass
        
    def tile_regs_release():
        return tile_regs_op_wrapper(tile_regs_release_impl, "tile_regs_release()")

    def kernel0():
        in0_cb = "in0_cb"
        in1_cb = "in1_cb"
        out_cb = "out_cb"

        cb_reserve_back(out_cb, out_block_size)
        for M in range(M_block):
            for N in range(N_block):
                for K in range(K_block):
                    # Compute starting indices for this block
                    m_start = M * M_block_size
                    n_start = N * N_block_size
                    k_start = K * K_block_size

                    # Load tiles for this block into buffers
                    load_in0_tiles(m_start, k_start)
                    load_in1_tiles(k_start, n_start) # sim datamovement

                    # Copy partial tiles from previous K block
                    tile_regs_acquire()
                    block_offset = M * (N_block * N_block_size) + (N * N_block_size)
                    if K != 0:
                        # cb_wait_front(out_cb, out_block_size)
                        for i in range(out_block_size):
                            # stall wait
                            copy_tile(out_cb, i + block_offset, i)
                        # cb_pop_front(out_cb, out_block_size)

                    matmul_block(in0_cb, in1_cb)

                    tile_regs_commit()
                    tile_regs_wait()
                    
                    for i in range(out_block_size):
                        pack_tile(i, out_cb, i + block_offset)
                    tile_regs_release()
        cb_push_back(out_cb, out_block_size)

    kernel0()

    # Print all simulation states
    print("\nSimulation state history:")
    for i, st in enumerate(sim_states):
        print(f"--- State {i} ---\n{st}\n")
    
    # Visualize the simulation states
    try:
        from visualizer import StateVisualizer
        print("Launching interactive visualization...")
        StateVisualizer(sim_states).show()
    except ImportError:
        print("Could not import visualizer. Make sure matplotlib is installed.")
        print("Try: pip install matplotlib")
