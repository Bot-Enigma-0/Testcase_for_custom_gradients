# Define the input file name
input_file = "of_grad.dat"

# Define the normalization factor
normalization_factor = 1  # You can set this to any desired factor

# Initialize an empty list to store the normalized values
DV_VALUE_new = []

# Function to check if a string can be converted to a float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Open the file and read it line by line
with open(input_file, 'r') as file:
    for line in file:
        # Strip any extra whitespace
        value = line.strip()
        
        # Check if the line contains a number
        if is_number(value):
            # Convert the string to a float, normalize it, and convert it back to a string
            normalized_value = str(float(value) * normalization_factor)
            
            # Append the normalized value to the DV_VALUE_new list
            DV_VALUE_new.append(normalized_value)

# Convert the list into a single row of comma-separated values
DV_VALUE_new_str = ", ".join(DV_VALUE_new)

# Output the result
print("DV_VALUE =", DV_VALUE_new_str)


