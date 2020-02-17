#!/bin/bash

# This script rebases all HOWTO branches to the main branch. It shoul be ran
# before generating the Markdown files, to avoid unrelated changes in the main
# branch to show up in the HOWTO diffs.

local main_branch="prerelease"
local howto_prefix="howto-"

git branch | grep "${howto_prefix}"