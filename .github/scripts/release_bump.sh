#!/usr/bin/env bash
# Decide which semver part a push to main bumps, from Conventional Commits.
#
#   <type>!: ... or a "BREAKING CHANGE:" footer -> major
#   feat: ...                                    -> minor
#   fix: ... / perf: ...                         -> patch
#   anything else (refactor, docs, ci, chore,
#   test, build, style, revert, ...)             -> none (no release)
#
# Only commit subjects decide the type: GitHub squash-merge bodies list the
# PR's individual commits ("* feat: ..."), which must not trigger a release.
# A breaking-change footer only counts at the start of a body line.
#
# Usage: release_bump.sh <revision range>   (e.g. "before..after", or "-1 HEAD")
# Prints one of: major, minor, patch, none.
set -euo pipefail

scope='(\([^)]*\))?'
bump=none
rank() { case "$1" in major) echo 3 ;; minor) echo 2 ;; patch) echo 1 ;; *) echo 0 ;; esac; }
raise() { if [ "$(rank "$1")" -gt "$(rank "$bump")" ]; then bump=$1; fi; }

while IFS= read -r -d '' message; do
  subject=${message%%$'\n'*}
  if [[ $subject =~ ^[a-z]+${scope}!:\  ]]; then
    raise major
  elif grep -qE '^BREAKING[ -]CHANGE: ' <<<"$message"; then
    raise major
  elif [[ $subject =~ ^feat${scope}:\  ]]; then
    raise minor
  elif [[ $subject =~ ^(fix|perf)${scope}:\  ]]; then
    raise patch
  fi
done < <(git log -z --format=%B "$@")

echo "$bump"
