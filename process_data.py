# Open the file for reading
with open('/home/jiachen/PromptIR/data_dir/noisy/denoise.txt', 'r') as file:
    lines = file.readlines()

# Process each line, removing "a" if it starts with "a"
processed_lines = [line[1:] if line.startswith('a') else line for line in lines]

# Write the processed lines back to the file
with open('denoise.txt', 'w') as file:
    file.writelines(processed_lines)