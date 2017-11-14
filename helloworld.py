import random
import sys
import os

#the below is for strings
print ("Hi")

name = 'Dj'

print (name)

#this is just a comment

'''this could also be a comment'''

quote = "\"It is time to move on"
print quote
print (quote)

print ("%s" %quote)
print "done with strings***\n"

#the below is for lists
print "---beginning lists"
lists = ['what','is','for','lunch']
print lists[0]

print lists[1:3]

extra = ['could', 'be' , 'mayo', 'southwest', 'bbq']

menu = [lists,extra]
print menu

print menu[0][3]

extra.remove("bbq")
print menu
extra.insert(4,'sweet onion')
print menu
menu.sort()
print menu
menu.reverse()
print menu
print (len(menu))


#tuple - a list that cannot be changed. shouldnt be changed
tuple_pi = (1,2,3,4,5)
list_tuple = list(tuple_pi)
print list_tuple
new_tuple = tuple(tuple_pi)
print new_tuple
list_tuple.insert(6,6)

#dictionary
dic_list = {1:3, 4:2, 6:5, 2:3, 1:7}
print dic_list
print dic_list.keys()
print dic_list.values()
print dic_list[1]
print dic_list.get(3)

#conditionals
age = 5
if age==12 :
    print "nice"
elif age == 13 :
    print "bad"
else : print "what !!"

#combining conditionals, AND/OR/NOT

#looping
for x in range(0,10):
    print (x,"yeay")

for y in [1,3,5,7,9]:
    print y

#functions
def printNum(num, text):
    print num, text

printNum(5,"hi")

#input console
print ("What is the time")

name = sys.stdin.readline()
print "you are right.", name

#print strings
longname =  ("%c is what i am called but my real name is %s, i like %d and %.4f") % ('d', "Dhivya", 8, 9.02)	
print longname
print longname.capitalize()
print longname.find("l")
print longname.strip()


