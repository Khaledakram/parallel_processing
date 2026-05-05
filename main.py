import cv2
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

SOURCE_DIR = 'input_photos'    
RESULT_DIR = 'processed_photos'   
Compute_Cores = max(1, int(cpu_count() * 0.75))
Image_size = (300, 300) 

report = {
    "sequential": None,
    "parallel": None
}

def convolution(image_data):
    
    height, width, channels = image_data.shape
    output_image = np.zeros_like(image_data)
    
    #(3x3 Blur Kernel)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0

    for c in range(channels):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                #Extracting the current pixel's neighbors (3x3)
                pixel_neighborhood = image_data[y-1:y+2, x-1:x+2, c]
                #(Convolution process) 
                pixel_value = np.sum(pixel_neighborhood * kernel)
                output_image[y, x, c] = pixel_value
    return output_image

def prepare_test_images():
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"[*] Generating {Image_size} test images...")
        print("-" * 50)
        for i in range(10):
            # Creating random data and converting it into an image
            random_pixels = np.random.randint(0, 255, (Image_size[0], Image_size[1], 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(SOURCE_DIR, f"image_{i}.jpg"), random_pixels)
    os.makedirs(RESULT_DIR, exist_ok=True)

def image_processing_unit(image_name):
    input_path = os.path.join(SOURCE_DIR, image_name)
    img = cv2.imread(input_path)
    if img is None: return False
    
    #apply convolution to the image
    processed_img = convolution(img)
    
    # Save the processed image
    output_path = os.path.join(RESULT_DIR, f"blurred_{image_name}")
    cv2.imwrite(output_path, processed_img)
    return True

def sequential_process(task_list):
    print(f"\n>>> Starting SERIAL processing...")
    start = time.time()
    for task in task_list:
        image_processing_unit(task)
        print(f"    [Processed] {task}")
        duration = time.time() - start
        report["sequential"] = duration
    return duration

def parallel_process(task_list):
    print(f"\n>>> Starting PARALLEL processing...")
    start = time.time()
    with Pool(processes=Compute_Cores) as pool:
        pool.map(image_processing_unit, task_list)
    duration = time.time() - start
    report["parallel"] = duration
    return duration

def show_comparison():
    print("\n" + "="*40)
    print("      PERFORMANCE COMPARISON REPORT      ")
    print("="*40)
    s = report["sequential"]
    p = report["parallel"]
    
    print(f"Sequential Time:   {f'{s:.2f}s' if s else 'N/A'}")
    print(f"Parallel Time: {f'{p:.2f}s' if p else 'N/A'}")
    
    if s and p:
        speedup = s / p
        print("-" * 40)
        print(f"Speedup Factor: {speedup:.2f}x faster")
        print(f"Efficiency:     {(speedup/Compute_Cores)*100:.1f}%")
    else:
        print("\n[!] Please run both modes to see full comparison.")
    print("="*40 + "\n")

if __name__ == "__main__":
    prepare_test_images()
    image_queue = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.jpg')]

    while True:
        print("1. SEQUENTIAL PROCESSING ")
        print("2. PARALLEL PROCESSING ")
        print("3. SHOW COMPARISON REPORT")
        print("4. Exit System")
        
        choice = input("\nSelect Execution Mode: ")

        if choice == '1':
            duration = sequential_process(image_queue)
            print(f"\n>> Total Sequential Time: {duration:.2f} seconds")
            print("-" * 50)
        elif choice == '2':
            duration = parallel_process(image_queue)
            print(f"\n>> Total Parallel Time: {duration:.2f} seconds")
            print("-" * 50)
        elif choice == '3':
            show_comparison()
        elif choice == '4':
            print("Exiting System...")
            print("-" * 50)
            break