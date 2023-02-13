from lfs import *
class Dog:

	def __init__(self, name, height):
		self.name=name
		self.height=height


import heapq
a = [3, 5, 1, 2, 6, 8, 7]
heapq.heapify(a)

print(a)



if __name__ == '__main__':

	tuples = list(zip([1,2,3,4,5],LFs_running_example))
	for t in tuples:
		print(t)
	dog_heap = []

	heapq.heapify(tuples)
	print(tuples)