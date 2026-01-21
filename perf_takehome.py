"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        """Add a VLIW instruction bundle (dict of engine -> list of slots)"""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        """Allocate a vector constant (broadcast scalar to 8 elements)"""
        if val not in self.vconst_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.vconst_map[val] = vec_addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Pipelined kernel: overlap loads of group N with VALU of group N-1.
        """
        tmp1 = self.alloc_scratch("tmp1")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)

        # Pre-allocate all scalar constants needed for hash (load phase)
        hash_scalar_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            sc1 = self.scratch_const(val1)
            sc3 = self.scratch_const(val3)
            hash_scalar_consts.append((sc1, sc3))

        # Pre-allocate multiply constants
        hash_mult_scalars = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                hash_mult_scalars.append(self.scratch_const(mult))
            else:
                hash_mult_scalars.append(None)

        # Pre-allocate other scalar constants
        zero_scalar = self.scratch_const(0)
        one_scalar = self.scratch_const(1)
        two_scalar = self.scratch_const(2)

        # Allocate vector destinations (no instructions yet)
        hash_consts_v = []
        hash_mult_v = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v1 = self.alloc_scratch(None, VLEN)
            v3 = self.alloc_scratch(None, VLEN)
            hash_consts_v.append((v1, v3))
            if hash_mult_scalars[hi] is not None:
                vm = self.alloc_scratch(None, VLEN)
                hash_mult_v.append(vm)
            else:
                hash_mult_v.append(None)

        zero_v = self.alloc_scratch(None, VLEN)
        one_v = self.alloc_scratch(None, VLEN)
        two_v = self.alloc_scratch(None, VLEN)
        n_nodes_v = self.alloc_scratch("n_nodes_v", VLEN)

        # Collect all vbroadcast ops (will be overlapped with initial data loads)
        pending_vbroadcasts = []
        for hi in range(len(HASH_STAGES)):
            v1, v3 = hash_consts_v[hi]
            sc1, sc3 = hash_scalar_consts[hi]
            pending_vbroadcasts.append(("vbroadcast", v1, sc1))
            pending_vbroadcasts.append(("vbroadcast", v3, sc3))
            if hash_mult_v[hi] is not None:
                pending_vbroadcasts.append(("vbroadcast", hash_mult_v[hi], hash_mult_scalars[hi]))
        pending_vbroadcasts.append(("vbroadcast", zero_v, zero_scalar))
        pending_vbroadcasts.append(("vbroadcast", one_v, one_scalar))
        pending_vbroadcasts.append(("vbroadcast", two_v, two_scalar))
        pending_vbroadcasts.append(("vbroadcast", n_nodes_v, self.scratch["n_nodes"]))

        n_vectors = batch_size // VLEN
        all_idx = [self.alloc_scratch(f"idx_{vi}", VLEN) for vi in range(n_vectors)]
        all_val = [self.alloc_scratch(f"val_{vi}", VLEN) for vi in range(n_vectors)]

        GROUP_SIZE = 6
        BUFFERS = 2
        s_addr = [[[self.alloc_scratch() for _ in range(VLEN)] for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]
        s_node = [[[self.alloc_scratch() for _ in range(VLEN)] for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]
        v_node = [[self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]
        v_tmp1 = [[self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]
        v_tmp2 = [[self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]

        # Overlap address computation AND vbroadcasts with loads for initial data
        idx_addrs = [self.alloc_scratch() for _ in range(n_vectors)]
        val_addrs = [self.alloc_scratch() for _ in range(n_vectors)]

        # Compute first address
        base0 = self.scratch_const(0)
        self.add_bundle({"alu": [
            ("+", idx_addrs[0], self.scratch["inp_indices_p"], base0),
            ("+", val_addrs[0], self.scratch["inp_values_p"], base0)
        ]})

        # Pipeline: load[vi] while computing addr[vi+1] AND doing vbroadcasts
        vb_idx = 0
        for vi in range(n_vectors):
            bundle = {"load": [
                ("vload", all_idx[vi], idx_addrs[vi]),
                ("vload", all_val[vi], val_addrs[vi])
            ]}
            if vi + 1 < n_vectors:
                base_next = self.scratch_const((vi + 1) * VLEN)
                bundle["alu"] = [
                    ("+", idx_addrs[vi+1], self.scratch["inp_indices_p"], base_next),
                    ("+", val_addrs[vi+1], self.scratch["inp_values_p"], base_next)
                ]
            # Overlap vbroadcasts with loads (up to 6 per cycle)
            if vb_idx < len(pending_vbroadcasts):
                bundle["valu"] = pending_vbroadcasts[vb_idx:vb_idx+6]
                vb_idx += 6
            self.add_bundle(bundle)

        # Handle remaining vbroadcasts if any (shouldn't happen with 32 vectors and ~19 vbroadcasts)
        while vb_idx < len(pending_vbroadcasts):
            self.add_bundle({"valu": pending_vbroadcasts[vb_idx:vb_idx+6]})
            vb_idx += 6

        self.add("flow", ("pause",))

        forest_p = self.scratch["forest_values_p"]
        total_iters = rounds * n_vectors
        n_groups = (total_iters + GROUP_SIZE - 1) // GROUP_SIZE

        def get_group_iters(group_idx, buf):
            iters = []
            for slot in range(GROUP_SIZE):
                iter_idx = group_idx * GROUP_SIZE + slot
                if iter_idx < total_iters:
                    vi = iter_idx % n_vectors
                    iters.append((slot, all_idx[vi], all_val[vi]))
            return iters

        def emit_loads_for_group(buf, iters):
            loads = []
            for slot, v_idx, v_val in iters:
                for j in range(VLEN):
                    loads.append(("load", s_node[buf][slot][j], s_addr[buf][slot][j]))
            return loads

        def emit_valu_for_group(buf, iters, phase):
            ops = []
            if phase == "xor":
                for slot, v_idx, v_val in iters:
                    ops.append(("^", v_val, v_val, v_node[buf][slot]))
            elif phase.startswith("hash_"):
                hi = int(phase.split("_")[1])
                part = int(phase.split("_")[2])
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                v1, v3 = hash_consts_v[hi]
                mult_v = hash_mult_v[hi]

                # Use multiply_add for stages with pattern (a + const) + (a << shift)
                if mult_v is not None:
                    if part == 0:
                        # multiply_add: val = val * mult + const
                        for slot, v_idx, v_val in iters:
                            ops.append(("multiply_add", v_val, v_val, mult_v, v1))
                    # Part 1 is not needed - multiply_add does it all in one op
                else:
                    if part == 0:
                        for slot, v_idx, v_val in iters:
                            ops.append((op1, v_tmp1[buf][slot], v_val, v1))
                            ops.append((op3, v_tmp2[buf][slot], v_val, v3))
                    else:
                        for slot, v_idx, v_val in iters:
                            ops.append((op2, v_val, v_tmp1[buf][slot], v_tmp2[buf][slot]))
            elif phase.startswith("idx_"):
                step = int(phase.split("_")[1])
                for slot, v_idx, v_val in iters:
                    # Use multiply_add to combine idx*2 + tmp in one op
                    if step == 0: ops.append(("&", v_tmp1[buf][slot], v_val, one_v))  # tmp1 = val & 1
                    elif step == 1: ops.append(("+", v_tmp1[buf][slot], v_tmp1[buf][slot], one_v))  # tmp1 = tmp1 + 1
                    elif step == 2: ops.append(("multiply_add", v_idx, v_idx, two_v, v_tmp1[buf][slot]))  # idx = idx*2 + tmp
                    elif step == 3: ops.append(("<", v_tmp1[buf][slot], v_idx, n_nodes_v))  # tmp1 = idx < n_nodes
                    elif step == 4: ops.append(("*", v_idx, v_idx, v_tmp1[buf][slot]))  # idx = idx * tmp1 (wrap)
            return ops

        valu_phases = ["xor"]
        for hi in range(6):
            valu_phases.append(f"hash_{hi}_0")
            # Skip part 1 for multiply_add stages (they do it all in part 0)
            if hash_mult_v[hi] is None:
                valu_phases.append(f"hash_{hi}_1")
        for step in range(5):  # Reduced from 6 to 5 steps using multiply_add
            valu_phases.append(f"idx_{step}")

        prev_buf = None
        prev_iters = None
        prev_phase_idx = 0
        prev_gather_alu = []  # Gather ALU ops from previous group

        # Precompute address ALU ops for first group
        first_iters = get_group_iters(0, 0)
        addr_alu_ops = []
        for slot, v_idx, v_val in first_iters:
            for j in range(VLEN):
                addr_alu_ops.append(("+", s_addr[0][slot][j], forest_p, v_idx + j))
        for i in range(0, len(addr_alu_ops), 12):
            self.add_bundle({"alu": addr_alu_ops[i:i+12]})

        for group in range(n_groups):
            buf = group % BUFFERS
            iters = get_group_iters(group, buf)

            # Precompute next group's address ALU (will be overlapped with this group's tail loads)
            next_addr_alu = []
            if group + 1 < n_groups:
                next_buf = (group + 1) % BUFFERS
                next_iters = get_group_iters(group + 1, next_buf)
                for slot, v_idx, v_val in next_iters:
                    for j in range(VLEN):
                        next_addr_alu.append(("+", s_addr[next_buf][slot][j], forest_p, v_idx + j))

            all_loads = emit_loads_for_group(buf, iters)

            load_idx = 0
            gather_idx = 0
            valu_op_offset = 0
            next_addr_idx = 0

            # Phase 1: Overlap loads with gather ALU (until gather ALU completes)
            while load_idx < len(all_loads) and gather_idx < len(prev_gather_alu):
                bundle = {}
                bundle["load"] = all_loads[load_idx:load_idx+2]
                load_idx += 2
                bundle["alu"] = prev_gather_alu[gather_idx:gather_idx+12]
                gather_idx += 12
                self.add_bundle(bundle)

            # Phase 2: Overlap loads with VALU
            while load_idx < len(all_loads) and prev_buf is not None and prev_phase_idx < len(valu_phases):
                bundle = {}
                bundle["load"] = all_loads[load_idx:load_idx+2]
                load_idx += 2

                valu_ops = emit_valu_for_group(prev_buf, prev_iters, valu_phases[prev_phase_idx])
                if valu_ops:
                    chunk = valu_ops[valu_op_offset:valu_op_offset+6]
                    if chunk:
                        bundle["valu"] = chunk
                    valu_op_offset += 6
                    if valu_op_offset >= len(valu_ops):
                        prev_phase_idx += 1
                        valu_op_offset = 0

                self.add_bundle(bundle)

            # Phase 3: Overlap remaining loads with next group's address ALU
            while load_idx < len(all_loads):
                bundle = {}
                bundle["load"] = all_loads[load_idx:load_idx+2]
                load_idx += 2

                if next_addr_idx < len(next_addr_alu):
                    bundle["alu"] = next_addr_alu[next_addr_idx:next_addr_idx+12]
                    next_addr_idx += 12

                self.add_bundle(bundle)

            # Handle remaining VALU after loads complete
            while prev_buf is not None and prev_phase_idx < len(valu_phases):
                valu_ops = emit_valu_for_group(prev_buf, prev_iters, valu_phases[prev_phase_idx])
                if valu_ops:
                    for i in range(0, len(valu_ops), 6):
                        self.add_bundle({"valu": valu_ops[i:i+6]})
                prev_phase_idx += 1

            # Handle remaining address ALU if not all overlapped
            while next_addr_idx < len(next_addr_alu):
                self.add_bundle({"alu": next_addr_alu[next_addr_idx:next_addr_idx+12]})
                next_addr_idx += 12

            # Prepare gather ALU for this group
            prev_gather_alu = []
            for slot, v_idx, v_val in iters:
                for j in range(VLEN):
                    prev_gather_alu.append(("+", v_node[buf][slot] + j, s_node[buf][slot][j], zero_const))

            prev_buf = buf
            prev_iters = iters
            prev_phase_idx = 0

        # Handle final group's gather ALU and VALU
        for i in range(0, len(prev_gather_alu), 12):
            self.add_bundle({"alu": prev_gather_alu[i:i+12]})

        while prev_buf is not None and prev_phase_idx < len(valu_phases):
            valu_ops = emit_valu_for_group(prev_buf, prev_iters, valu_phases[prev_phase_idx])
            if valu_ops:
                for i in range(0, len(valu_ops), 6):
                    self.add_bundle({"valu": valu_ops[i:i+6]})
            prev_phase_idx += 1

        # Overlap address computation with stores for final data
        store_idx_addrs = [self.alloc_scratch() for _ in range(n_vectors)]
        store_val_addrs = [self.alloc_scratch() for _ in range(n_vectors)]

        # Compute first address
        base0 = self.scratch_const(0)
        self.add_bundle({"alu": [
            ("+", store_idx_addrs[0], self.scratch["inp_indices_p"], base0),
            ("+", store_val_addrs[0], self.scratch["inp_values_p"], base0)
        ]})

        # Pipeline: store[vi] while computing addr[vi+1]
        for vi in range(n_vectors):
            bundle = {"store": [
                ("vstore", store_idx_addrs[vi], all_idx[vi]),
                ("vstore", store_val_addrs[vi], all_val[vi])
            ]}
            if vi + 1 < n_vectors:
                base_next = self.scratch_const((vi + 1) * VLEN)
                bundle["alu"] = [
                    ("+", store_idx_addrs[vi+1], self.scratch["inp_indices_p"], base_next),
                    ("+", store_val_addrs[vi+1], self.scratch["inp_values_p"], base_next)
                ]
            self.add_bundle(bundle)

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
