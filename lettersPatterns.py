import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Path to the labeled test set folder
test_folder = r"C:\Users\visantana\Documents\tropical-captcha\labeled_testSet\AlreadyProccedLabeled"

# Create dictionaries to store counts of characters in different positions
positions = [str(i) for i in range(1, 7)]
position_counts = {pos: defaultdict(int) for pos in positions}
vowel_counts = {pos: 0 for pos in positions}
consonant_counts = {pos: 0 for pos in positions}

vowels = 'aeiouAEIOU'
consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'

# Iterate through each image in the folder
for image_filename in os.listdir(test_folder):
    if image_filename.endswith(".png"):
        characters = image_filename.split(".")[0]  # Remove extension
        
        for pos, char in enumerate(characters, start=1):
            position_counts[str(pos)][char] += 1
            if char in vowels:
                vowel_counts[str(pos)] += 1
            elif char in consonants:
                consonant_counts[str(pos)] += 1

# Print the character distribution in different positions
for pos in positions:
    print(f"Position {pos}: {position_counts[pos]}")
    print(f"Vowels in Position {pos}: {vowel_counts[pos]}")
    print(f"Consonants in Position {pos}: {consonant_counts[pos]}")
    print()

# Visualize the patterns using heatmaps and bar plots
plt.figure(figsize=(15, 10))

# Character distribution heatmaps
for pos in positions:
    plt.subplot(3, 2, int(pos))
    sns.heatmap(data=[[position_counts[pos][char] for char in '123456789abcdefghijklmnopqrstuvwxyz'] ],
                xticklabels=list('123456789abcdefghijklmnopqrstuvwxyz'), yticklabels=[],
                cmap='YlGnBu', annot=True, fmt='d')
    plt.title(f"Character Distribution in Position {pos}")

# Vowels vs. Consonants bar plots
plt.subplot(3, 2, 6)
plt.bar(positions, list(vowel_counts.values()), label='Vowels', color='b')
plt.bar(positions, list(consonant_counts.values()), bottom=list(vowel_counts.values()), label='Consonants', color='r')
plt.title("Vowels vs. Consonants in Positions")
plt.xlabel("Position")
plt.ylabel("Occurrences")
plt.legend()

plt.tight_layout()
plt.show()
