import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2

def load_data(image_dir, label_dir):
    data = []
    # Sort image files
    image_files = sorted(os.listdir(image_dir))
    label_files = {os.path.splitext(f)[0]: f for f in sorted(os.listdir(label_dir))}
    
    for img_file in image_files:
        # Full path of the image
        image_path = os.path.join(image_dir, img_file)

        # Identify the corresponding label file
        base_name = os.path.splitext(img_file)[0]
        label_file = label_files.get(base_name)
        if not label_file:
            print(f"Label for {img_file} not found.")
            continue
        label_path = os.path.join(label_dir, label_file)

        # Load image as an array
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image_array = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        # Load label
        try:
            with open(label_path, 'r') as f:
                label = f.read().strip()
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            continue

        # Add to data
        data.append({'image': image_array, 'label': label})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def split_image_in_chars(image, num_chars=6):
    """
    Splits the full captcha image into num_chars columns (characters).
    Assuming uniform horizontal division.
    """
    width = image.shape[1]
    char_width = width // num_chars  # Uniform division
    sub_images = []
    
    for i in range(num_chars):
        # Crop column i
        sub_img = image[:, i*char_width:(i+1)*char_width]
        sub_images.append(sub_img)
        
    return sub_images

def plot_image(df, i, column='image'):
    image = df.iloc[i][column]
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(f"Row {i} - Image")
    plt.show()

def plot_result(correct_chars_arr):
    # Get unique values of correctly recognized characters sorted
    thresholds = sorted(correct_chars_arr.unique())
    print(correct_chars_arr.value_counts().sort_index())
    # Calculate the count and convert to percentage for each threshold
    percentages = [
        (correct_chars_arr >= threshold).mean()
        for threshold in thresholds
    ]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, percentages, marker='o', linestyle='-', color='b', label='Percentage')
    plt.xlabel('Minimum number of recognized characters per captcha')
    plt.ylabel('Recognition rate')
    plt.title('Result')
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid(True)
    plt.legend()
    plt.show()

def expand_row(row, num_chars=6):
    """
    Given a DF row with 'image' and 'label',
    returns a dictionary with columns:
      'image_1', 'image_2', ..., 'image_6'
      'label_1', 'label_2', ..., 'label_6'
    """
    image = row["image"]
    label = row["label"]
    
    # Split the image into 6 sub-images
    sub_images = split_image_in_chars(image, num_chars=num_chars)
    
    # Split the label into 6 characters
    # Example: "JW4CZF" -> ['J','W','4','C','Z','F']
    chars = list(label)
    
    # Create output dictionary
    output = {}
    for i in range(num_chars):
        output[f"image_{i+1}"] = sub_images[i]
        output[f"label_{i+1}"] = chars[i]
        
    return output

def expand_df(df, num_chars=6):
    # For each row, obtain a dictionary with extra columns
    expanded_rows = df.apply(lambda row: expand_row(row, num_chars=num_chars), axis=1)
    
    # expanded_rows will be a Series of dictionaries; convert to DataFrame
    df_exp = pd.DataFrame(expanded_rows.tolist())
    
    # Concatenate original columns (if you want to keep 'image' and 'label')
    # with the generated columns:
    df_final = pd.concat([df.reset_index(drop=True), df_exp], axis=1)
    
    return df_final

def filter_df(df, num_chars=6):
    # Filter rows where the label has exactly 6 characters and no '?' symbol
    df_filtered = df[df["label"].apply(lambda x: '?' not in x) & df["label"].apply(lambda x: len(x) == 6)].copy()
    return df_filtered

def preprocess_img(img):
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img     = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    _, img  = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    return img

def preprocess_df(df):
    # Apply preprocessing to each image
    df["image"] = df["image"].apply(preprocess_img)
    return df

def prepare_data(data):  
    X = []
    y = []

    for idx, row in data.iterrows():
        image = row["image"]
        label = row["label"]

        sub_images = split_image_in_chars(image)
        characters = list(label)

        for sub_img, ch in zip(sub_images, characters):
            X.append(sub_img)
            y.append(ch)
    
    return np.array(X), np.array(y)
