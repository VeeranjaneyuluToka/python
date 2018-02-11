# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 18:09:07 2018

@author: Veeranjaneyulu Toka
"""

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x== pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left)+middle+quick_sort(right)

def basic_data_types():
    
    print("numbers")
    def numbers():
        x = 3
        print(type(x))
        print(x)
        print(x+1)
        print(x-1)
        print(x*2)
        print(x**2)
        
        x+= 1
        print(x)
        
        x*=2
        print(x)
        
        y = 2.5
        print(type(y))
        print(y, y+1, y-1, y*2, y**2)
        
    numbers()
    
    print("Booleans")
    def booleans():
        t = True
        f = False
        print(type(t))
        print(t and f)
        print(t or f)
        print(not t)
        print(t != f)
        
    booleans()
    
    print("strings")
    def strings_v():
        hello = 'hello'
        world = 'world'
        print(hello)
        print(len(hello))
        hw = hello + ' ' + world
        print(hw)
        hw12 = '%s %s %s'%(hello, world, 12)
        print(hw12)
        
    strings_v()
    
    print("string functions")
    def string_func():
        s = 'hello'
        print(s.capitalize())
        print(s.upper())
        print(s.rjust(7))
        print(s.center(7))
        print(s.replace('l', '(ell)'))
        print('world '.strip())
        
    string_func()
    
def containers_p():
    print('lists in python')
    def lists_p():
        xs = [3, 1, 2]
        print(xs, xs[2])
        print(xs[-1])
        xs[2] = 'foo'
        print(xs)
        xs.append('bar')
        print(xs)
        x = xs.pop()
        print(x, xs)
        
        print('slicing')
        def slicing_lists_p():
            nums = list(range(5))
            print(nums)
            print(nums[2:4])
            print(nums[2:])
            print(nums[:2])
            print(nums[:])
            print(nums[:-1])
            nums[2:4]=[8,9]
            print(nums)
        slicing_lists_p()
        
        print('loops')
        def loops_lists_p():
            animals = ['cat', 'dog', 'monkey']
            for animal in animals:
                print(animal)
                
            for idx, animal in enumerate(animals):
                print('#%d:%s'%(idx+1, animal))
                
        loops_lists_p()
        
        print("list comprehensions")
        def list_comprehensions_p():
            nums = [0, 1, 2, 3, 4]
            squares = []
            for x in nums:
                squares.append(x**2)
            print(squares)
            
            squares = [x ** 2 for x in nums]
            print(squares)
            
            even_squares = [x**2 for x in nums if x%2 == 0]
            print(even_squares)
            
        list_comprehensions_p()
        
    lists_p()
    
    print("dictionaries in python")
    def dict_p():
        d = {'cat': 'cute', 'dog': 'furry'}
        print(d['cat'])
        print('cat' in d)
        d['fish'] = 'wet'
        print(d['fish'])
        print(d.get('monkey', 'N/A'))
        print(d.get('fish', 'N/A'))
        del d['fish']
        print(d.get('fish','N/A'))
        
        def dict_loops_p():
            d = {'person': 2, 'cat':4, 'spider':8}
            for animal in d:
                legs = d[animal]
                print('A %s has %d legs' %(animal, legs))
                
            for animal, legs in d.items():
                print('A %s has %d legs' %(animal, legs))
                
        dict_loops_p()
        
        def dict_compreh_p():
            nums = [0, 1, 2, 3, 4]
            even_num_to_square = {x:x**2 for x in nums if x%2 == 0}
            print(even_num_to_square)
            
        dict_compreh_p()
        
    dict_p()

    print("sets in python")
    def sets_p():
        animals = {'cat', 'dog'}
        print('cat' in animals)
        print('fish' in animals)
        animals.add('fish')
        print('fish' in animals)
        print(len(animals))
        animals.add('cat')
        print(len(animals))
        animals.remove('cat')
        print(len(animals))
        
        def sets_loops_p():
            animals = {'cat', 'dog', 'fish'}
            for idx, animal in enumerate(animals):
                print("%d :%s" %(idx+1, animal))
                
        sets_loops_p()
        
    sets_p()
    
    print("touples in python")
    def tuples_p():
        d = {(x, x+1): x for x in range(10)}
        t = (5, 6)
        print(type(t))
        print(d[t])
        print(d[(1, 2)])
        
    tuples_p()
    
def sign_func(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'
        

class Generator(object):
    def __init__(self, name):
        self.name = name
    
    def greet(self, loud=False):
        if loud:
            print("HELLO, %s!" % self.name.upper())
        else:
            print("Hello, %s" %self.name)
            
            
def main():
    print(quick_sort([3, 6, 8, 10, 1, 2, 1]))
    
    basic_data_types()
    
    containers_p()
    
    #functions
    for x in [-1, 0, 1]:
        print(sign_func(x))
        
    #classes
    g = Generator('Fred')
    g.greet()
    g.greet(loud=True)
    
if __name__ == "__main__":
    main()

