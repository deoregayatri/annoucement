import os
import re

# Mapping of common formatting characters to safe ASCII
ASCII_REPLACEMENTS = {
    '-': '-',  # em dash
    '-': '-',  # en dash
    '"': '"',  # smart quote left
    '"': '"',  # smart quote right
    ''': "'",  # smart single quote left
    ''': "'",  # smart single quote right
    '...': '...',# ellipsis
    '[START]': '[START]',
    '[REPORT]': '[REPORT]',
    '[OK]': '[OK]',
    '[FAILED]': '[FAILED]',
    '[WARNING]': '[WARNING]',
    '[HOT]': '[HOT]',
}

def clean_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filepath}: Could not read as utf-8 ({e})")
        return False

    original_content = content
    
    # Replace known mappings
    for bad_char, good_char in ASCII_REPLACEMENTS.items():
        content = content.replace(bad_char, good_char)
        
    # Find any remaining non-ASCII characters and remove them
    # Ascii range is \x00 to \x7F
    # We will log if we strip any.
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    
    if non_ascii_pattern.search(content):
        # We found other non-ascii chars. Let's just strip them
        content = non_ascii_pattern.sub('', content)

    if content != original_content:
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned non-ASCII characters from: {filepath}")
        return True
    return False

def scan_directory(directory):
    cleaned_count = 0
    for root, _, files in os.walk(directory):
        # Skip hidden dirs, virtual envs, etc
        basename = os.path.basename(root)
        if basename.startswith('.') and basename != '.':
            continue
        if 'venv' in root or '__pycache__' in root or 'node_modules' in root:
            continue
            
        for file in files:
            if file.endswith('.py') or file.endswith('.sh') or file.endswith('.md'):
                filepath = os.path.join(root, file)
                if clean_file(filepath):
                    cleaned_count += 1
                    
    print(f"\nOptimization Complete! Cleaned {cleaned_count} files.")

if __name__ == "__main__":
    import sys
    # Default to the current directory if no path is provided
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Scanning directory: {os.path.abspath(target_dir)}")
    scan_directory(target_dir)
