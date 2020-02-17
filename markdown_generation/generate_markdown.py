"""Generates Markdown from templates.

Generates Markdown from all templates found in the same directory (and
recursive subdirectories) of this script. Will store the Markdown into
--repo_dir.

Templates can contain the following annotations:

ANNOTATION                 REPLACEMENT
@insert_branch_diff <b1>:  Output of `git diff <b0> <b1>, where <b0> and <b1>
                           are branches, and <b0> can be set with --head_branch.

@insert_code <f> <lang>:   Markdown view of code found in file <f> using <lang>.
                           <lang> is optional and defaults to 'py' (Python).

"""
import os

from absl import app
from absl import flags

from git import Repo

FLAGS = flags.FLAGS

flags.DEFINE_string('repo_dir', '../',
                    'Relative dir to repository. The final Markdown will be '
                    'written here.')

flags.DEFINE_string('head_branch', 'prerelease',
                    'Name of the "HEAD" branch to run diffs against')

flags.DEFINE_bool('strip_copyright_from_code', True,
                  'If true, removes copyright header from code.')

flags.DEFINE_string('template_path', 'markdown_templates',
                    'Relative path to the directory containing templates.')

flags.DEFINE_string('input_suffix', '.template.md',
                    'Files ending with this suffix are treated as templates.')

flags.DEFINE_string('output_suffix', '.md',
                    'Replace @input_suffix with this when creating the final '
                    'Markdown file.')

flags.DEFINE_bool('exclude_tests', True, 'If true, do not show test files in '
                   'diffs.')


def wrap_in_code(lines, lang):
  return [f'```{lang}'] + lines + ['```']


def insert_branch_diff_fn(target_branch, source_branch='prerelease'):
  git = Repo(FLAGS.repo_dir).git
  diff_text = git.diff(source_branch, target_branch, '--', ':(exclude)*.md')
  return wrap_in_code(diff_text.split('\n'), 'diff')


def insert_py_code_fn(file_path, lang='py'):
  file_path = os.path.join(FLAGS.repo_dir, file_path)
  lines = open(file_path, encoding='utf8').read().splitlines()
  if FLAGS.strip_copyright_from_code:
    while len(lines) and lines[0].startswith('#'):
      lines = lines[1:]
  # Strips empty lines from start and aned.
  while len(lines) and not lines[0].strip():
    lines = lines[1:]
  while len(lines) and not lines[-1].strip():
    lines = lines[:-1]

  return wrap_in_code(lines, lang=lang)


def generate_markdown_content(file_path):
  output_lines = []
  operations = {
    '@insert_branch_diff': insert_branch_diff_fn,
    '@insert_code': insert_py_code_fn
  }
  for line in open(file_path, encoding='utf8').read().splitlines():
    words = line.split()
    if len(words) < 2 or words[0] not in operations:
      output_lines.append(line)
      continue
    output_lines += operations[words[0]](*words[1:])
  return '\n'.join(output_lines)


def prepare_output_path(root, filename):
  output_dir = os.path.join(FLAGS.repo_dir, root)
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  # This replaces all occurrences, so we assume the input suffix only occurs
  # once in the input filename.
  out_filename = filename.replace(FLAGS.input_suffix, FLAGS.output_suffix)
  return os.path.join(output_dir, out_filename)


def generate_markdown_recursively():
  print('Generating Markdown files...')
  for root, _, files in os.walk('.'):
    for filename in files:
      if not filename.endswith(FLAGS.input_suffix):
        continue
      input_path = os.path.join(root, filename)
      output_path = prepare_output_path(root, filename)
      with open(output_path, 'w', encoding='utf8') as f:
        f.write(generate_markdown_content(input_path))
      print(f'  {output_path} (from {input_path})')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  generate_markdown_recursively()


if __name__ == '__main__':
  app.run(main)
