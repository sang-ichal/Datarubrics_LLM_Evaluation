import os
import re

total_nosection = 0
error_case_counter = 0


def process_file(input_file, output_file):
    global total_nosection, error_case_counter

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Use regex to find all occurrences of text between 'References' and closing sections
    opening_sections = ["References", "Bibliography"]
    closing_sections = [
        "Appendix",
        "Supplementary",
        "Related Works",
        "Content",
        "Methods",
        "Datasheet",
        "Appendices",
        "Table 4:",
        "Data curation models",
        "A Details of Data Collecting",
        'A Datasheets',
        'APPENDIX',
        'A Plotting results',
        'Related Work',
        'A Detailed Statistics',
        'Table 3: IaC-Eval dataset columns'
    ]
    pattern = (
        r"("
        + "|".join(opening_sections)
        + r")(.*?)(?="
        + "|".join(f"(?:{section})" for section in closing_sections)
        + ")"
    )
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if matches:
        # Get the last match (as per requirement)
        last_match = matches[0]

        # Split the content into three parts:
        start_pos = last_match.start()
        end_pos = last_match.end()

        content_before = content[:start_pos]
        content_after = content[end_pos:]

        # Verify the content before References is unchanged
        original_before = content[:start_pos]
        if content_before != original_before:
            print(f"Warning: Content before 'References' was modified in {input_file}")

        # Verify the content after the pattern is unchanged
        original_after = content[end_pos:]
        if content_after != original_after:
            print(f"Warning: Content after References was modified in {input_file}")

        # Combine the parts, excluding the References section
        trimmed_content = content_before + content_after
    else:
        # Fallback: If no pattern found, find last 'References' and trim from there to end
        last_ref_pos = content.rfind("References")
        if last_ref_pos != -1:
            # Save the problematic section to error cases folder
            error_dir = "./txt/error_cases"
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, f"{error_case_counter}_error.txt")

            # Extract the content from References to the end
            error_content = content[last_ref_pos:]

            # Save the error case
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"Original file: {input_file}\n")
                f.write("Content from 'References' to end:\n")
                f.write("=" * 50 + "\n")
                f.write(error_content)

            error_case_counter += 1

            # Keep everything before the last 'References'
            trimmed_content = content[:last_ref_pos]
            print(
                f"No closing section found in {input_file}, trimmed from last 'References' to end"
            )
            print(f"Error case saved to: {error_file}")
        else:
            # If no 'References' found at all, keep original content and save to other folder
            trimmed_content = content
            
            # Save to no_references folder
            no_ref_dir = "./txt/no_references"
            os.makedirs(no_ref_dir, exist_ok=True)
            no_ref_file = os.path.join(no_ref_dir, f"{os.path.basename(input_file)}")
            with open(no_ref_file, "w", encoding="utf-8") as f:
                f.write(content)
                
            print(f"No 'References' found in {input_file}, keeping original content")
            print(f"File saved to: {no_ref_file}")

        total_nosection += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(trimmed_content)


def main():
    input_dir = "./txt/selected"
    output_dir = "./txt/selected_no_ref"

    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                print(f"Processing: {file}")
                process_file(input_path, output_path)

    print(f"Total files with no section: {total_nosection}")
    print(f"Total error cases saved: {error_case_counter}")


if __name__ == "__main__":
    main()
