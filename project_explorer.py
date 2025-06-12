import os
import fnmatch

# Maximum number of characters from each file to display
MAX_CHARACTERS = 500000000

def display_project_structure_with_content(startpath='.'):
    # Directories to exclude from traversal
    excluded_dirs = ['.git', '__pycache__']

    # File patterns considered text files
    text_file_patterns = ['*.py', '*.json', '*.yaml', '*.yml', '*.md','*.ipynb','*.rs','*.toml']

    for root, dirs, files in os.walk(startpath):
        # Exclude certain directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        # Determine the indentation based on directory depth
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        # Print directory name
        print('{}{}/'.format(indent, os.path.basename(root)))

        subindent = ' ' * 4 * (level + 1)
        for filename in files:
            # Check if the file matches any of our text patterns
            if any(fnmatch.fnmatch(filename, pattern) for pattern in text_file_patterns):
                filepath = os.path.join(root, filename)
                print(f"{subindent}{filename}:")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # not Limiting the content length
                        shortened_content = content
                        # Remove leading/trailing whitespace and print line by line
                        stripped_content = shortened_content.strip()
                        content_lines = stripped_content.split('\n')
                        for line in content_lines:
                            print(f"{subindent*2}{line}")
                except UnicodeDecodeError:
                    print(f"{subindent*2}[Binary or undecodable file]")
                except Exception as e:
                    print(f"{subindent*2}[Error reading file: {e}]")
                print("-" * 20)  # Separator between files

if __name__ == "__main__":
    display_project_structure_with_content()