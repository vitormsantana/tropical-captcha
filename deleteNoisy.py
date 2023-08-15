import os

relegated_folder = r"C:\Users\visantana\Documents\tropical-captcha\relegated"
letters_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"

def main():
    try:
        for relegated_file in os.listdir(relegated_folder):
            print(relegated_file)
            if relegated_file.endswith(".png"):
                last_char = relegated_file[-5]
                #if is_valid_character(last_char):
                print(last_char)
                process_relegated_file(last_char)
    except Exception as e:
        print(f"An error occurred: {e}")

def is_valid_character(char):
    return char.isalnum() and (char.isalpha() or char.isdigit())

def process_relegated_file(char_to_delete):
    specific_char_folder_path = os.path.join(letters_folder, char_to_delete)
    if os.path.exists(specific_char_folder_path):
        for file in os.listdir(specific_char_folder_path):
            if file.startswith("4lines") and file.endswith(f"_{char_to_delete}.png"):
                specific_file_path = os.path.join(specific_char_folder_path, file)
                if os.path.exists(specific_file_path):
                    os.remove(specific_file_path)
                    print(f"Deleted: {specific_file_path}")
                else:
                    print(f"File not found: {specific_file_path}")
    else:
        print(f"Folder not found: {specific_char_folder_path}")

if __name__ == "__main__":
    main()


