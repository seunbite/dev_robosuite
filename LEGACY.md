# Legacy and archive policy

The complete pre-cleanup workspace is preserved on branch
`codex/pre-paper-cleanup-20261118` at commit `8ac9365`.

The release branch supports the final Motion Program pipeline and paper data.
Earlier Pilot-40/Pilot-90 protocols, Google-Robot naming migrations, persona and
context experiments, H4/correction pilots, Go2 experiments, DROID/Franka
prototypes, and cluster-specific launch wrappers are historical and are not
part of the supported reproduction path.

Generated render trees, review decks, caches, and intermediate LaTeX/PDF
previews are intentionally absent. Retrieve historical source from the archive
branch; regenerate media from versioned programs rather than adding scratch
outputs back to the default branch.
