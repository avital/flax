"""Test."""
from absl import app
from absl import flags

from git import Repo

FLAGS = flags.FLAGS

flags.DEFINE_string('repo_path', '.', 'Relative path to the Github repository.')

# TODO(marcvanzee): Currently this only works for README.md. Think of a more 
# general solution.

def wrap_in_code(text, type='py'):
  return [f'```{type}'] + text.split('\n') + ['```']

def insert_branch_diff_fn(target_branch, source_branch='prerelease'):
  git = Repo(FLAGS.repo_path).git
  diff_text = git.diff(source_branch, target_branch, '--', ':(exclude)*.md')
  return wrap_in_code(diff_text, 'diff')


def insert_py_code_fn(file_path, lang='py'):
  lines = open(file_path, encoding='utf8').readlines()
  return wrap_in_code(lines)


operations = {
  '@insert_branch_diff': insert_branch_diff_fn,
  '@insert_py_code': insert_py_code_fn
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
