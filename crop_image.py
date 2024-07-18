import cv2
import numpy as np

def process_image(image_path, output_path=None):
    img = cv2.imread(image_path)

    #x1, y1= 270, 70 # for rainy_070.png
    x1, y1= 180, 70 # for rainy_024.png


    x2, y2 = x1+80, y1+80
    cropped_area = img[y1:y2, x1:x2]
    
    enlarged_area = cv2.resize(cropped_area, (0, 0), fx=2, fy=2)
    eh, ew, _ = enlarged_area.shape
    img_height, img_width, _ = img.shape
    start_x = img_width - ew
    start_y = img_height - eh
    img[start_y:start_y+eh, start_x:start_x+ew] = enlarged_area
    
    # Draw the red box around the original area
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw the green box around the enlarged area
    cv2.rectangle(img, (start_x, start_y), (start_x+ew, start_y+eh), (0, 255, 0), 2)
    
    # Save the result instead of displaying it


    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")

# Example usage
output_dir = "/home/jiachen/PromptIR/visualization"

# for 070 image deraining
# input_image_path  = "/data/jiachen/all_in_one/Test/derain/Rain100L/input/070.png"
# target_image_path = "/data/jiachen/all_in_one/Test/derain/Rain100L/target/070.png"
# promptir_image_path = "/home/jiachen/PromptIR/output_promptir/derain/070.png"
# promptxrestormer_image_path = "/home/jiachen/PromptIR/output_promptxrestormer_epoch47/derain/070.png"    


# output_path = f"{output_dir}/rainy_070.png"
# process_image(input_image_path, output_path)

# output_path = f"{output_dir}/clean_070.png"
# process_image(target_image_path, output_path)

# output_path = f"{output_dir}/promptir_070.png"
# process_image(promptir_image_path, output_path)

# output_path = f"{output_dir}/promptxrestormer_070.png"
# process_image(promptxrestormer_image_path, output_path)

# for 024 image deraining
input_image_path  = "/data/jiachen/all_in_one/Test/derain/Rain100L/input/024.png"
target_image_path = "/data/jiachen/all_in_one/Test/derain/Rain100L/target/024.png"
promptir_image_path = "/home/jiachen/PromptIR/output_promptir/derain/024.png"
promptxrestormer_image_path = "/home/jiachen/PromptIR/output_promptxrestormer_epoch47/derain/024.png"

output_path = f"{output_dir}/rainy_024.png"
process_image(input_image_path, output_path)

output_path = f"{output_dir}/clean_024.png"
process_image(target_image_path, output_path)

output_path = f"{output_dir}/promptir_024.png"
process_image(promptir_image_path, output_path)

output_path = f"{output_dir}/promptxrestormer_024.png"
process_image(promptxrestormer_image_path, output_path)

