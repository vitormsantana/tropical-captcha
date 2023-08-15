import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools

# Path to the labeled test set folder
test_folder = r"C:\Users\visantana\Documents\tropical-captcha\labeled_testSet"

# Create a dictionary to store counts of character sequences in different positions
sequence_counts = defaultdict(lambda: defaultdict(int))

# Length of character sequences to analyze
sequence_length = 2  # Change to 3 for triplets, etc.

# Set of character combinations to analyze (vowels and consonants)
characters = 'aeiou'
all_sequences = [seq for seq in itertools.product(characters, repeat=sequence_length + 1)]

# Initialize counts for all sequences in each position
for position in range(1, sequence_length + 1):
    sequence_counts[position] = {seq: 0 for seq in all_sequences}

# Iterate through each image in the folder
for image_filename in os.listdir(test_folder):
    if image_filename.endswith(".png"):
        # Extract positions and characters from the filename
        characters = image_filename.split(".")[0]  # Remove extension
        positions = characters[:-1]
        characters = characters[-1]

        # Extract character sequences and update counts
        for i in range(len(positions) - sequence_length):
            sequence = characters[i:i + sequence_length]
            next_index = i + sequence_length
            if next_index < len(characters):
                next_char = characters[next_index]
                position = int(positions[i])  # Convert position to integer
                if next_char in characters:  # Limit analysis to specific characters
                    sequence_counts[position][sequence + next_char] += 1

# Visualize the sequence patterns using a heatmap
plt.figure(figsize=(30, 10))
sns.heatmap(data=[[sequence_counts[position][sequence] for sequence in all_sequences] for position in range(1, sequence_length + 1)],
            xticklabels=all_sequences, yticklabels=range(1, sequence_length + 1),
            cmap='YlGnBu', annot=True, fmt='d')
plt.title(f"Character Sequence Distribution in Different Positions (Length {sequence_length + 1})")
plt.xlabel("Character Sequences")
plt.ylabel("Position")
plt.show()
