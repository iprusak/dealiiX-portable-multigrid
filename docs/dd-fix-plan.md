# Domain-decomposition (BNN / BDDC) fix plan

Plan from a review of the DD code on `bddc-preconditioner` @ `090160e` (2026-10-07).

**Why:** no test compares the DD solutions against an independent global solve, so
bugs that shift the answer went unnoticed. The 2D BDDC results converge at O(h)
towards the exact value ||u||_L2 = 0.0412615 (3D on `main`: O(h^2) towards
0.0249871), where Q_p elements should be essentially exact.

**Status:** Phase 1 item 1 is done and verified: the 2D BDDC driver gives
||u||_L2 = 0.0412615 on every mesh with 4 and 9 subdomains, against
0.0376-0.0412 before the fix. Nothing else has started.

Phases run in order. Suggested commit grouping: (0 + 1), (2), (3 + 4), so the
refactor never mixes with bug fixes. Re-check that each finding still applies
before fixing it.

## Phase 0: reference test

New `correctness_tests/check_correctness_dd_vs_global/`:

- reference = plain global matrix-free CG (`include/operators/portable_laplace_operator.h`);
- solvers under test: BNN-enhanced, BNN-vanilla (`solve_dd`), BDDC-Pi, BDDC with
  static condensation;
- 2D and 3D, 2x2, 3x3 and 3x3x3 subdomains, degrees 1-4, tolerance about 1e-8;
- at least 3x3 subdomains are needed to trigger item 3 below.

## Phase 1: wrong-answer bugs

1. **Schur RHS** (`portable_schur_interface_operator.h`, `assemble_rhs_schur`).
   The Dirichlet operator has identity rows on the interface, so the solve
   returns `z_G = F_G`. The next `vmult_interface_cell_range` then adds an extra
   `-A_GG F_G` term. Fix: zero the interface entries of `z` before applying `A`,
   as `vmult()` already does. **Done.**
2. **`solve_dd` with BNN converges to `S^-1 b - Q b`**
   (`portable_solver_projected_cg.h`, `portable_bnn_preconditioner.h`).
   `project_initial_residual(r)` balances `r` but never adds `Q r` to `x`.
   Replace it with `DDPreconditionerBase::initialize_iterate(x, r)`: BNN computes
   `c = Q r`, then `x += c` and `r -= S c`; BDDC does nothing. Update all callers.
3. **BNN `relevant_coarse_indices` (stored `S*phi_j` columns).** These are chosen
   from MPI-partitioner neighbours (`ghost_targets()`/`import_targets()`), but the
   true support is the 2-hop neighbourhood of subdomains that share a DoF. With
   hashed ownership, two subdomains that share only a corner owned by a third
   rank are not partitioner neighbours. A model shows missing columns from 3x3
   subdomains up. Fix: build the set from the sharing sets in
   `SubdomainDoFHandler`, and add a debug check comparing `S*z0` from the stored
   columns with `interface_operator->vmult(z0)`.
4. **Uninitialized `v` in `solve_enhanced`.** The first iteration runs
   `v.sadd(0, 1, s_tilde)` on a vector deal.II allocated without initialization,
   so leftover NaN values survive (0 * NaN = NaN). Use `v = s_tilde` when
   `it == 1`. Check `solve_projected` and `SolverBlockCG` for the same pattern.

**Verification:** the Phase 0 test passes for all four solvers, and the solution
norm matches the exact value from the first mesh on.

## Phase 2: latent problems

5. **Variant mismatch.** `BDDCPreconditioner` builds its internal
   `subdomain_bddc_operator` without passing `variant`, so it defaults to
   `corner_edge_face`; so do the driver's `level_subdomain_bddc_matrices`. With
   any other variant, the coarse space and the fine space would not cover each
   other. Pass `variant` through, and assert that both sides have the same
   number of local coarse DoFs.
6. **Vacuous Pi vs. static-condensation check** in
   `source/bddc_preconditioner/program.cc`. The loop runs only `{true}`, so
   `solution_pi` stays empty. Make both runs a runtime option, and only compare
   when both solves actually ran.
7. **Corner-pinned A_RR can be singular** when a subdomain with no physical
   boundary has no corners. Only call `compute_local_edge_face_schur_complement()`
   when static condensation is enabled, and fail with a clear message in that case.
8. **Assumptions inherited from `main`.** Re-check each on this branch, then guard it:
   - subdomain and global cells are matched in lockstep in `fill_dof_info`
     (add a cell-centre assertion);
   - `interface_id = 100 + rank` can clash with real boundary ids;
   - interface cells are only cells with an interface face, which is wrong for
     non-box partitions;
   - physical boundaries are assumed homogeneous Dirichlet;
   - a single rank leaves the interface partitioner null.

## Phase 3: performance (measure before and after)

9. **BDDC `vmult_coarse_correction`:** store Phi as one block and use one reduction
   kernel and one prolongation kernel, instead of 2 x n_local launches and n_local
   host syncs per apply.
10. **BNN coarse setup:** replace the P global Schur applies with local Schur
    applies on neighbours' `phi_j`, and assemble the coarse matrix sparsely, as
    BDDC does.
11. **Document only (no change now):** the all-gathers in `SubdomainDoFHandler`
    and the coarse problems replicated on every rank. Revisit beyond about 1k ranks.

## Phase 4: modularity

12. A shared `PrimalConstraintSet` (offsets, DoF lists, weights, local-to-global
    map, variant fallback) used by `SubdomainBDDCOperator`, `BDDCPreconditioner`
    and the projection kernels. This removes the duplicated
    `setup_primal_constraint_views()`.
13. Split `SubdomainLaplaceOperatorBase` into a core interface plus optional block
    and masked extensions, and move the DD index views into `SubdomainDoFHandler`.
14. `SolverProjectedCG`: keep one `solve_dd` with an `if constexpr` trait for the
    `S*z` reuse; remove `solve_bnn`, the old `solve` and the runtime `std::is_same`
    branches.
15. BDDC constructor: bundle the level hierarchies into one setup struct and
    remove the `static_cast`s.
16. Remove dead code: the BDDC versions of `global_interface_to_coarse`,
    `coarse_to_global_interface` and `project_to_homogeneous_constraints_*`;
    `coarse_problem_rank`; `solve_fine_correction`; the `#if 0` blocks.
