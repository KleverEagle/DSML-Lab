import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 8, 7]

# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o')
plt.title("Line Plot")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
plt.bar(x, y)
plt.title("Bar Chart")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

# Pie Chart
labels = ["A", "B", "C", "D", "E"]
plt.figure(figsize=(5, 5))
plt.pie(y, labels=labels, autopct="%1.1f%%")
plt.title("Pie Chart")
plt.show()

# Histogram
data = [10, 12, 15, 18, 18, 19, 20, 22, 24, 25, 27, 28, 30]
plt.figure(figsize=(6, 4))
plt.hist(data, bins=5)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
