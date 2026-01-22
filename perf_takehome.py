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

    def batch_scratch_consts(self, values):
        """Batch load multiple constants using 2 load slots per cycle"""
        new_vals = [v for v in values if v not in self.const_map]
        for v in new_vals:
            self.const_map[v] = self.alloc_scratch()
        for i in range(0, len(new_vals), 2):
            if i + 1 < len(new_vals):
                self.add_bundle({"load": [
                    ("const", self.const_map[new_vals[i]], new_vals[i]),
                    ("const", self.const_map[new_vals[i + 1]], new_vals[i + 1])
                ]})
            else:
                self.add("load", ("const", self.const_map[new_vals[i]], new_vals[i]))
        return [self.const_map[v] for v in values]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with specialized handling for rounds 0-1.
        Round 0: All indices are 0 - use single tree[0] load
        Round 1: Indices are 1 or 2 - use arithmetic: node = A + idx*B
        Rounds 2+: Pipelined scatter-gather
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        for i in range(0, len(init_vars), 2):
            if i + 1 < len(init_vars):
                self.add_bundle({"load": [
                    ("const", tmp1, i),
                    ("const", tmp2, i + 1)
                ]})
                self.add_bundle({"load": [
                    ("load", self.scratch[init_vars[i]], tmp1),
                    ("load", self.scratch[init_vars[i + 1]], tmp2)
                ]})
            else:
                self.add("load", ("const", tmp1, i))
                self.add("load", ("load", self.scratch[init_vars[i]], tmp1))

        # Collect all scalar constants
        all_const_values = [0, 1, 2]
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            all_const_values.append(val1)
            all_const_values.append(val3)
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                all_const_values.append(mult)

        self.batch_scratch_consts(all_const_values)

        zero_const = self.const_map[0]
        hash_scalar_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            sc1 = self.const_map[val1]
            sc3 = self.const_map[val3]
            hash_scalar_consts.append((sc1, sc3))

        hash_mult_scalars = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mult = 1 + (1 << val3)
                hash_mult_scalars.append(self.const_map[mult])
            else:
                hash_mult_scalars.append(None)

        zero_scalar = self.const_map[0]
        one_scalar = self.const_map[1]
        two_scalar = self.const_map[2]

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

        # Allocate tree values for rounds 0-1
        tree_addr_0 = self.alloc_scratch()
        tree_addr_1 = self.alloc_scratch()
        tree_addr_2 = self.alloc_scratch()
        tree_val_0 = self.alloc_scratch()  # scalar
        tree_val_1 = self.alloc_scratch()
        tree_val_2 = self.alloc_scratch()
        tree_v_0 = self.alloc_scratch(None, VLEN)  # broadcast version
        # For round 1: A = 2*tree[1] - tree[2], B = tree[2] - tree[1]
        round1_A = self.alloc_scratch(None, VLEN)
        round1_B = self.alloc_scratch(None, VLEN)

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
        v_tmp1 = [[self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]
        v_tmp2 = [[self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)] for _ in range(BUFFERS)]

        # Temp vectors for specialized rounds - REUSE v_tmp1[0], v_tmp2[0] to save scratch
        # spec_tmp1/spec_tmp2 are only used during rounds 0-1, v_tmp1/v_tmp2 during rounds 2+
        spec_tmp1 = v_tmp1[0]  # Reuse buffer 0's tmp1
        spec_tmp2 = v_tmp2[0]  # Reuse buffer 0's tmp2
        spec_node = [self.alloc_scratch(None, VLEN) for _ in range(GROUP_SIZE)]

        idx_addrs = [self.scratch["inp_indices_p"]] + [self.alloc_scratch() for _ in range(n_vectors - 1)]
        val_addrs = [self.scratch["inp_values_p"]] + [self.alloc_scratch() for _ in range(n_vectors - 1)]

        offset_consts = [vi * VLEN for vi in range(1, n_vectors)]
        self.batch_scratch_consts(offset_consts)
        if 0 not in self.const_map:
            self.scratch_const(0)

        forest_p = self.scratch["forest_values_p"]

        # Compute tree addresses for indices 0, 1, 2
        self.add_bundle({"alu": [
            ("+", tree_addr_0, forest_p, zero_scalar),
            ("+", tree_addr_1, forest_p, one_scalar),
            ("+", tree_addr_2, forest_p, two_scalar)
        ]})

        # Load tree values (will overlap with data loading)
        # First load tree[0] and tree[1]
        self.add_bundle({"load": [
            ("load", tree_val_0, tree_addr_0),
            ("load", tree_val_1, tree_addr_1)
        ]})
        # Then tree[2]
        self.add_bundle({"load": [("load", tree_val_2, tree_addr_2)]})

        # Load input data and overlap with vbroadcasts
        vb_idx = 0
        for vi in range(n_vectors):
            bundle = {"load": [
                ("vload", all_idx[vi], idx_addrs[vi]),
                ("vload", all_val[vi], val_addrs[vi])
            ]}
            alu_ops = []
            if vi + 1 < n_vectors:
                base_next = self.const_map[(vi + 1) * VLEN]
                alu_ops.extend([
                    ("+", idx_addrs[vi+1], self.scratch["inp_indices_p"], base_next),
                    ("+", val_addrs[vi+1], self.scratch["inp_values_p"], base_next)
                ])
            if alu_ops:
                bundle["alu"] = alu_ops
            if vb_idx < len(pending_vbroadcasts):
                bundle["valu"] = pending_vbroadcasts[vb_idx:vb_idx+6]
                vb_idx += 6
            self.add_bundle(bundle)

        while vb_idx < len(pending_vbroadcasts):
            self.add_bundle({"valu": pending_vbroadcasts[vb_idx:vb_idx+6]})
            vb_idx += 6

        # Broadcast tree[0] only - round 1 prep will be overlapped with round 0
        self.add_bundle({"valu": [
            ("vbroadcast", tree_v_0, tree_val_0),
        ]})

        # Allocate temps for round 1 prep
        tree_v_1 = self.alloc_scratch(None, VLEN)
        tree_v_2 = self.alloc_scratch(None, VLEN)

        # Queue round 1 prep ops to be overlapped with round 0
        round1_prep_ops = [
            ("vbroadcast", tree_v_1, tree_val_1),
            ("vbroadcast", tree_v_2, tree_val_2),
        ]

        self.add("flow", ("pause",))

        # ===== ROUND 0: All indices are 0 =====
        # tree[0] is in tree_v_0, all indices start at 0
        # Process in waves to avoid temp conflicts
        def emit_hash_index_ops(v_val, v_idx, v_node, v_tmp1, v_tmp2):
            """Return list of (op, dest, src1, src2, [src3]) tuples for hash+index update."""
            ops = []
            ops.append(("^", v_val, v_val, v_node))
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                v1, v3 = hash_consts_v[hi]
                mult_v = hash_mult_v[hi]
                if mult_v is not None:
                    ops.append(("multiply_add", v_val, v_val, mult_v, v1))
                else:
                    ops.append((op1, v_tmp1, v_val, v1))
                    ops.append((op3, v_tmp2, v_val, v3))
                    ops.append((op2, v_val, v_tmp1, v_tmp2))
            ops.append(("&", v_tmp1, v_val, one_v))
            ops.append(("+", v_tmp1, v_tmp1, one_v))
            ops.append(("multiply_add", v_idx, v_idx, two_v, v_tmp1))
            ops.append(("<", v_tmp1, v_idx, n_nodes_v))
            ops.append(("*", v_idx, v_idx, v_tmp1))
            return ops

        # Collect all round 0 VALU bundles
        round0_bundles = []
        for wave_start in range(0, n_vectors, GROUP_SIZE):
            wave_end = min(wave_start + GROUP_SIZE, n_vectors)

            # Collect all ops for this wave, organized by dependency phase
            all_ops = []
            for i, vi in enumerate(range(wave_start, wave_end)):
                slot = i  # Use slot within wave
                ops = emit_hash_index_ops(all_val[vi], all_idx[vi], tree_v_0,
                                          spec_tmp1[slot], spec_tmp2[slot])
                all_ops.append(ops)

            # Emit phase by phase to respect dependencies
            max_ops = max(len(ops) for ops in all_ops) if all_ops else 0
            for phase in range(max_ops):
                phase_ops = []
                for ops in all_ops:
                    if phase < len(ops):
                        phase_ops.append(ops[phase])
                # Batch into bundles of 6
                for i in range(0, len(phase_ops), 6):
                    round0_bundles.append(phase_ops[i:i+6])

        # Overlap round 0 with round 1 prep
        # Round 1 prep needs: broadcast tree_v_1, tree_v_2, then compute round1_A, round1_B
        # These are few ops that can be interleaved with round 0
        # Since round 0 uses 6 VALU slots per bundle, we can't add more to same bundle
        # But we can insert prep ops in separate bundles that don't block round 0

        # Emit first part of round0, then insert prep ops, then continue round0
        # The prep ops: vbroadcast tree_v_1, tree_v_2 (can be in one bundle with round0 if slots available)
        # Then: round1_A = tree_v_1 * two_v, round1_B = tree_v_2 - tree_v_1
        # Then: round1_A = round1_A - tree_v_2

        # Insert round1 prep as early as possible while round0 is running
        prep_phase = 0
        for bi, bundle in enumerate(round0_bundles):
            if prep_phase == 0 and len(bundle) <= 4:
                # Add vbroadcast ops to this bundle
                bundle.extend(round1_prep_ops)
                prep_phase = 1
            elif prep_phase == 1 and len(bundle) <= 4:
                # Add multiply and subtract
                bundle.extend([
                    ("*", round1_A, tree_v_1, two_v),
                    ("-", round1_B, tree_v_2, tree_v_1),
                ])
                prep_phase = 2
            elif prep_phase == 2 and len(bundle) <= 5:
                # Add final subtract
                bundle.append(("-", round1_A, round1_A, tree_v_2))
                prep_phase = 3
            self.add_bundle({"valu": bundle})

        # If prep wasn't fully inserted (all bundles were full), emit remaining prep
        if prep_phase == 0:
            self.add_bundle({"valu": round1_prep_ops})
            prep_phase = 1
        if prep_phase == 1:
            self.add_bundle({"valu": [
                ("*", round1_A, tree_v_1, two_v),
                ("-", round1_B, tree_v_2, tree_v_1),
            ]})
            prep_phase = 2
        if prep_phase == 2:
            self.add_bundle({"valu": [("-", round1_A, round1_A, tree_v_2)]})

        # ===== ROUND 1: Indices are 1 or 2 =====
        # node = round1_A + idx * round1_B (using multiply_add)
        # Collect all round 1 bundles first, then overlap first group addresses with later waves
        round1_bundles = []
        for wave_start in range(0, n_vectors, GROUP_SIZE):
            wave_end = min(wave_start + GROUP_SIZE, n_vectors)

            # Phase 1: Compute node = A + idx * B for all vectors in wave
            node_ops = []
            for i, vi in enumerate(range(wave_start, wave_end)):
                slot = i
                # multiply_add: spec_node[slot] = all_idx[vi] * round1_B + round1_A
                node_ops.append(("multiply_add", spec_node[slot], all_idx[vi], round1_B, round1_A))
            for i in range(0, len(node_ops), 6):
                round1_bundles.append(node_ops[i:i+6])

            # Phase 2+: Hash and index update
            all_ops = []
            for i, vi in enumerate(range(wave_start, wave_end)):
                slot = i
                ops = emit_hash_index_ops(all_val[vi], all_idx[vi], spec_node[slot],
                                          spec_tmp1[slot], spec_tmp2[slot])
                all_ops.append(ops)

            max_ops = max(len(ops) for ops in all_ops) if all_ops else 0
            for phase in range(max_ops):
                phase_ops = []
                for ops in all_ops:
                    if phase < len(ops):
                        phase_ops.append(ops[phase])
                for i in range(0, len(phase_ops), 6):
                    round1_bundles.append(phase_ops[i:i+6])

        # ===== ROUNDS 2-15: Pipelined scatter-gather =====
        # Now indices can be anywhere, use standard approach

        start_round = 2
        remaining_rounds = rounds - start_round
        total_iters = remaining_rounds * n_vectors
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
                    ops.append(("^", v_val, v_val, s_node[buf][slot][0]))
            elif phase.startswith("hash_"):
                hi = int(phase.split("_")[1])
                part = int(phase.split("_")[2])
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                v1, v3 = hash_consts_v[hi]
                mult_v = hash_mult_v[hi]
                if mult_v is not None:
                    if part == 0:
                        for slot, v_idx, v_val in iters:
                            ops.append(("multiply_add", v_val, v_val, mult_v, v1))
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
                    if step == 0: ops.append(("&", v_tmp1[buf][slot], v_val, one_v))
                    elif step == 1: ops.append(("+", v_tmp1[buf][slot], v_tmp1[buf][slot], one_v))
                    elif step == 2: ops.append(("multiply_add", v_idx, v_idx, two_v, v_tmp1[buf][slot]))
                    elif step == 3: ops.append(("<", v_tmp1[buf][slot], v_idx, n_nodes_v))
                    elif step == 4: ops.append(("*", v_idx, v_idx, v_tmp1[buf][slot]))
            return ops

        valu_phases = ["xor"]
        for hi in range(6):
            valu_phases.append(f"hash_{hi}_0")
            if hash_mult_v[hi] is None:
                valu_phases.append(f"hash_{hi}_1")
        for step in range(5):
            valu_phases.append(f"idx_{step}")

        # Compute addresses for first group of round 2+ (vectors 0-5)
        # Overlap with round 1 waves 1+ since wave 0 updates vectors 0-5
        first_group_iters = get_group_iters(0, 0)
        first_addr_ops = []
        for slot, v_idx, v_val in first_group_iters:
            for j in range(VLEN):
                first_addr_ops.append(("+", s_addr[0][slot][j], forest_p, v_idx + j))

        # Count bundles in wave 0 (first GROUP_SIZE vectors of round 1)
        # Wave 0 has: 1 bundle for node_ops + (len(emit_hash_index_ops result) / GROUP_SIZE) bundles
        # emit_hash_index_ops produces ~15 ops per vector -> ~15 bundles for wave
        # Total wave 0 bundles = 1 + ~15 = ~16
        # Be conservative: find where wave 0 ends in round1_bundles
        wave0_end = 0
        ops_per_wave = 1 + len(emit_hash_index_ops(all_val[0], all_idx[0], spec_node[0], spec_tmp1[0], spec_tmp2[0]))
        wave0_end = ops_per_wave  # bundles for wave 0

        # Emit round 1: first wave0_end bundles as VALU only, then overlap with first_addr_ops
        for bi, bundle in enumerate(round1_bundles):
            if bi < wave0_end:
                # Wave 0: emit VALU only (indices 0-5 being updated)
                self.add_bundle({"valu": bundle})
            else:
                # Waves 1+: can overlap with first group address computation
                combined = {"valu": bundle}
                addr_start = (bi - wave0_end) * 12
                if addr_start < len(first_addr_ops):
                    combined["alu"] = first_addr_ops[addr_start:addr_start+12]
                self.add_bundle(combined)

        # Emit any remaining address ops not yet overlapped
        emitted_addr = (len(round1_bundles) - wave0_end) * 12
        for i in range(emitted_addr, len(first_addr_ops), 12):
            self.add_bundle({"alu": first_addr_ops[i:i+12]})

        prev_buf = None
        prev_iters = None
        prev_phase_idx = 0

        for group in range(n_groups):
            buf = group % BUFFERS
            iters = get_group_iters(group, buf)

            next_addr_alu = []
            if group + 1 < n_groups:
                next_buf = (group + 1) % BUFFERS
                next_iters = get_group_iters(group + 1, next_buf)
                for slot, v_idx, v_val in next_iters:
                    for j in range(VLEN):
                        next_addr_alu.append(("+", s_addr[next_buf][slot][j], forest_p, v_idx + j))

            all_loads = emit_loads_for_group(buf, iters)

            load_idx = 0
            valu_op_offset = 0
            next_addr_idx = 0

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

                if next_addr_idx < len(next_addr_alu):
                    bundle["alu"] = next_addr_alu[next_addr_idx:next_addr_idx+12]
                    next_addr_idx += 12

                self.add_bundle(bundle)

            while load_idx < len(all_loads):
                bundle = {}
                bundle["load"] = all_loads[load_idx:load_idx+2]
                load_idx += 2

                if next_addr_idx < len(next_addr_alu):
                    bundle["alu"] = next_addr_alu[next_addr_idx:next_addr_idx+12]
                    next_addr_idx += 12

                self.add_bundle(bundle)

            while prev_buf is not None and prev_phase_idx < len(valu_phases):
                valu_ops = emit_valu_for_group(prev_buf, prev_iters, valu_phases[prev_phase_idx])
                if valu_ops:
                    for i in range(valu_op_offset, len(valu_ops), 6):
                        self.add_bundle({"valu": valu_ops[i:i+6]})
                prev_phase_idx += 1
                valu_op_offset = 0

            while next_addr_idx < len(next_addr_alu):
                self.add_bundle({"alu": next_addr_alu[next_addr_idx:next_addr_idx+12]})
                next_addr_idx += 12

            prev_buf = buf
            prev_iters = iters
            prev_phase_idx = 0

        # Store results - pre-allocate addresses
        store_idx_addrs = [self.scratch["inp_indices_p"]] + [self.alloc_scratch() for _ in range(n_vectors - 1)]
        store_val_addrs = [self.scratch["inp_values_p"]] + [self.alloc_scratch() for _ in range(n_vectors - 1)]

        store_addr_alu = []
        for vi in range(1, n_vectors):
            base = self.const_map[vi * VLEN]
            store_addr_alu.append(("+", store_idx_addrs[vi], self.scratch["inp_indices_p"], base))
            store_addr_alu.append(("+", store_val_addrs[vi], self.scratch["inp_values_p"], base))

        # Final group VALU overlapped with store address computation
        final_valu_bundles = []
        while prev_buf is not None and prev_phase_idx < len(valu_phases):
            valu_ops = emit_valu_for_group(prev_buf, prev_iters, valu_phases[prev_phase_idx])
            if valu_ops:
                for i in range(0, len(valu_ops), 6):
                    final_valu_bundles.append(valu_ops[i:i+6])
            prev_phase_idx += 1

        # Emit final VALU overlapped with store ALU
        valu_idx = 0
        alu_idx = 0
        while valu_idx < len(final_valu_bundles) or alu_idx < len(store_addr_alu):
            bundle = {}
            if valu_idx < len(final_valu_bundles):
                bundle["valu"] = final_valu_bundles[valu_idx]
                valu_idx += 1
            if alu_idx < len(store_addr_alu):
                bundle["alu"] = store_addr_alu[alu_idx:alu_idx+12]
                alu_idx += 12
            if bundle:
                self.add_bundle(bundle)

        for vi in range(n_vectors):
            self.add_bundle({"store": [
                ("vstore", store_idx_addrs[vi], all_idx[vi]),
                ("vstore", store_val_addrs[vi], all_val[vi])
            ]})

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

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
