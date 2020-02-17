"""Test."""
from absl import app
from absl import flags

from git import Repo

FLAGS = flags.FLAGS

flags.DEFINE_string('repo_path', '.', 'Relative path to the Github repository.')

flags.DEFINE_bool('strip_copyright_from_code', True, 'If true, removes copyright header from code.')

# TODO(marcvanzee): Currently this only works for README.md. Think of a more 
# general solution.

def wrap_in_code(lines, lang):
  return [f'```{lang}'] + lines + ['```']


def insert_branch_diff_fn(target_branch, source_branch='prerelease'):
  git = Repo(FLAGS.repo_path).git
  diff_text = git.diff(source_branch, target_branch, '--', ':(exclude)*.md')
  return wrap_in_code(diff_text.split('\n'), 'diff')


def insert_py_code_fn(file_path, lang='py'):
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


operations = {
  '@insert_branch_diff': insert_branch_diff_fn,
  '@insert_code': insert_py_code_fn
}

def generate_template(input_path, output_path):
  output_lines = []
  for line in open(input_path, encoding='utf8').read().splitlines():
    words = line.split()
    if len(words) < 2 or words[0] not in operations:
      output_lines.append(line)
      continue
    output_lines += operations[words[0]](*words[1:])

  with open(output_path, 'w', encoding='utf8') as output_f:
    output_f.write('\n'.join(output_lines))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  generate_template('.README.template.md', 'README.md')


if __name__ == '__main__':
  app.run(main)
