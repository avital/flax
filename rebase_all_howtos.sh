#!/bin/bash
# Rebases all HOWTO branches to the main branch. It should be ran before 
# generating the Markdown files, to avoid unrelated changes in the main branch 
# to show up in the HOWTO diffs.

main_branch="prerelease"
howto_prefix="howto-"

git checkout "${main_branch}"
for howto_branch in $(git branch | grep "${howto_prefix}"); do
  git rebase "${main_branch}" "${howto_branch}"
done

git push --all origin --force-with-lease