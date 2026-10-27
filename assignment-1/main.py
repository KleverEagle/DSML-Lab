# Question 1 - List
list = [1,2,"Apple", "Samsung", "Oppo", "Xiaomi",2]

list.pop()
print(list)
list.append("Vivo")
print(list)
list.remove(1)
print(list)
list.insert(2, "Motorala")
print(list)

# Question 2 - Dictionaries
dict = {
    "name": "Gharshit",
    "age": 18,
    "sid":25105036,
    "branch":"ECE",
    "college": "PEC"
}
print(f' My name is {dict["name"]}.')
print(f' My age is {dict.get("age")}.')
dict["sid"] = 25103128
dict["branch"] = "CSE"

# Question 3 - Sets
set1 = {1,2,3,4}
set2 = {2,3,5}
print(set1.union(set2))
print(set1.intersection(set2))
print(set1.difference(set2))
print(set1.symmetric_difference(set2))

# update changes the original set
print(set1)
set1.difference_update(set2)
print(set1)
