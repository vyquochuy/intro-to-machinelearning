# Import thư viện cần thiết
import numpy as np

# Dữ liệu từ bảng
data = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]

# Tổng số mẫu
total_samples = len(data)

# Tính Gini impurity của tập dữ liệu ban đầu
num_yes = sum(1 for row in data if row[-1] == "Yes")
num_no = total_samples - num_yes

gini_D = 1 - ((num_yes / total_samples) ** 2 + (num_no / total_samples) ** 2)

# Hàm tính Gini của một thuộc tính sau khi chia
def gini_attribute(data, attribute_index):
    subsets = {}
    for row in data:
        key = row[attribute_index]
        if key not in subsets:
            subsets[key] = {"Yes": 0, "No": 0}
        subsets[key][row[-1]] += 1
    
    # Tính Gini của thuộc tính đó
    gini = 0
    for subset in subsets.values():
        total = subset["Yes"] + subset["No"]
        if total == 0:
            continue
        prob_yes = subset["Yes"] / total
        prob_no = subset["No"] / total
        gini_subset = 1 - (prob_yes ** 2 + prob_no ** 2)
        gini += (total / total_samples) * gini_subset
    
    return gini

# Tính Gini cho từng thuộc tính
attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
gini_values = {attr: gini_attribute(data, i) for i, attr in enumerate(attributes)}

# Chọn thuộc tính có Gini nhỏ nhất (tốt nhất)
best_attribute = min(gini_values, key=gini_values.get)

gini_D, gini_values, best_attribute

# Kết quả
# Gini impurity của tập dữ liệu ban đầu
# Gini impurity của từng thuộc
print("Gini impurity của tập dữ liệu ban đầu:", gini_D)
print("Gini impurity của từng thuộc tính:", gini_values)
print("Thuộc tính tốt nhất để chia:", best_attribute)
# vẽ cây
import matplotlib.pyplot as plt
import networkx as nx
