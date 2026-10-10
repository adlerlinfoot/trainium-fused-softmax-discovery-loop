# Prepared contribution to trainium-agent-labs

Target: `yahavb/trainium-agent-labs`, base branch **master**.
Proposed branch: **discover-trainium-tp4**.
PR title/body: [PR_DESCRIPTION.md](PR_DESCRIPTION.md).

## Additive transfer

| impact2 path | Target path/action |
| --- | --- |
| `trainium-softmax/` | Copy tracked source, documentation and evidence at the same path |
| `discover-reference/` | Copy tracked snapshots and MIT license at the same path |
| `docs/archive/` | Copy under `docs/discover-trainium/archive/` |
| `docs/REPORT.md` | Copy to `docs/discover-trainium/REPORT.md`; fix links for its new depth |
| `docs/HANDOFF_VALIDATION.json` | Copy to `docs/discover-trainium/HANDOFF_VALIDATION.json` |
| `docs/PR_DESCRIPTION.md` | Local PR body; no need to include in target commit |
| `README.md` | Append a Discover section to the existing lab README; preserve its contents |
| `.gitignore` | Append missing exclusions; preserve existing patterns |
| `.gitattributes` | Merge vendor/snapshot/patch whitespace rules; preserve existing attributes |

Archive document links need one additional `../` in the target because the
archive moves beneath `docs/discover-trainium/`.
Keep `discover-reference/` at repository root: the RL parity tests resolve
these snapshots relative to the project root.

The prepared local target checkout is `/workspace/impact2-handoff`, on branch
`discover-trainium-tp4`. Its staged additions are reviewable with
`git diff --cached`. `/tmp/impact2-proposed-pr.patch` is the corresponding
portable patch against the inspected target master revision. These are local
artifacts, not a GitHub fork or published PR. Hardware runs are not repeated
as part of this handoff.

The prepared PR body is `/tmp/impact2-pr-body.md`, with the title heading
removed and report links rewritten for the target layout. The complete
write-up is [REPORT.md](REPORT.md).

## Readiness validation

All **20 tests** passed in the prepared target checkout: seven evaluator
tests and thirteen RL/TP4/partial-model tests. JSON parsing, local documentation
links, staged whitespace checks and reverse patch applicability passed. The
copied code/evidence matches impact2 byte for byte; the target's existing
README and ignore rules retain their original contents. No weights, compiler
binaries or training checkpoint files are in the proposed diff. See
[HANDOFF_VALIDATION.json](HANDOFF_VALIDATION.json). The checks used the existing
isolated environment; hardware experiments were not repeated.

## Publish when requested

1. Refresh the target's `master` and check the prepared changes for conflicts.
2. Fork the target into the authorized GitHub account, or use an existing fork.
3. Point the prepared checkout's push remote at that fork. Keep the upstream
   repository as the read-only base remote.
4. Commit and push `discover-trainium-tp4` to the fork.
5. Open a PR targeting `yahavb/trainium-agent-labs:master`, using the draft's
   first heading as title and the remaining text as body.

Use a body file for GitHub CLI submission to preserve formatting. The local
integration branch is based on the target repository rather than impact2's
unrelated history, so it can form a normal upstream PR.
