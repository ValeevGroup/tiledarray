# Recommended Workflow Elements  {#Workflow-Notes}

## `pre-commit` git hooks

It is recommended to use the [pre-commit hook manager](https://pre-commit.com/) to enforce coding conventions, perform static code analysis, and manage TiledArray-specific infrastructure. Simply install `pre-commit` as described [here](https://pre-commit.com/#installation). Then run `pre-commit install` in the TiledArray source directory. File `.pre-commit-config.yaml` describes the hook configuration used by TiledArray; feel free to PR additional hooks.

Each time you try to commit a changeset in a repo in which `pre-commit` hooks have been installed each hook will be executed on each file added or changed in the changeset. Some hooks are designed to simply prevent nonconformant source code, documentation, infrastructure files, etc. from being committed, whereas other hooks will change the files to make them conformant. In either case, the commit will fail if any changes are needed. You will need to update the changeset (by amending the commit with the changes performed by the hooks and/or any changes you performed manually) and try again.

N.B. Changes in files performed by the `pre-commit` hooks are not instantly "seen" by the IDE, so it is recommended to manually run `git status` after a failed commit.

Among the 2 most important use cases for `pre-commit` hooks are:
- invoking clang-format automatically on each file added or changed in a commit, and
- keeping IDs (versions, git tags, etc.) of dependencies syncronized across TiledArray source.

### pre-commit git hook: clang-format

This hook runs `clang-format` to enforce the TiledArray code formatting conventions.

### pre-commit git hook: dependency-version-update

Whenever the ID of a dependency changes there are several places in the source tree where it needs to be updated: CMake code, documentation, and potentially even in the source code itself. This used to be done manually or even not done at all, i.e. we assumed that the user provided the appropriate version of MADNESS, with only minimal sufficiency checks via the CMake code located in cmake/modules ).

This hook will perform propagate any changes to the dependency IDs automatically whenever
the IDs are changed in `external/versions.cmake`. Thus, whenever dependency IDs need to be changed you should *only* change them in this file and then let the hook handle the rest.

`versions.cmake` file encodes dependency IDs as follows:
```
set(TA_TRACKED_MADNESS_TAG fe5fff9f61fd0780af64608a8f87f14afa72228d)
set(TA_TRACKED_MADNESS_PREVIOUS_TAG ab3476487e58792e8c77a60bd62ca0420b6ebc1a)
```
When you need to change the required version of MADNESS you should replace the previous version with the current version and update the current version with the new value. When you try to commit the updated `versions.cmake` file the `pre-commit` hook will update the necessary files with the new MADNESS version and block the commit. You will need to add
the updated files to the index and try again.

N.B. The hook code is located in `bin/admin/dependency-versions-update-hook.py`. It updates `git` *revision numbers* directly, without checking context, since revision numbers have high entropy. Replacement of dependency *versions* is more error prone and must be done in a context-sensitive manner. Significant changes to the files that include dependency versions (e.g. INSTALL.md) may require changes to the hook script.
