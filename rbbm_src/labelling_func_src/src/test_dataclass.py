from dataclasses import dataclass

class Dog:
	def __init__(self, name):
		self.name=name


@dataclass(eq=True)
class Repair:
	new_lf : Dog = None
	dog_name : str = None

	def __hash__(self):
		return hash(self.new_lf) + hash(self.dog_name)



if __name__  == "__main__":

	d1 = Dog('1')
	d2 = Dog('2')

	r1 = Repair(new_lf=d1, dog_name='blah1')
	r2 = Repair(new_lf=d1, dog_name='blah1')
	r3 = Repair(new_lf=d1, dog_name='blah2')

	l1 = [r1, r2]
	l2 = [r1, r3]

	print(set(l1))
	print(set(l2))